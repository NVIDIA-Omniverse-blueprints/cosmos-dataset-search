# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VastDB-backed Haystack document store for Cosmos Dataset Search.

Mirrors the VastDocumentStore surface used by CDS collection CRUD, indexing,
and search pipelines. Vectors are stored as fixed-size float lists in VastDB
and searched via the VAST Query Engine ADBC driver
(``array_cosine_distance`` / ``array_distance`` / ``array_negative_inner_product``).

Collection metadata lives in a dedicated ``_collection_registry`` table (JSON
blob per collection), matching the VastDB registry pattern so ``db_models``
continues to work without Postgres.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from src.haystack.components.schema_utils import MetadataConfig
from src.haystack.serializer import SerializerMixin
from src.visual_search.exceptions import InputValidationError

# Apply SDK select() patch for tables that include vector columns.
from src.haystack.components.vastdb import vastdb_patch as _vastdb_patch  # noqa: F401

logger = logging.getLogger(__name__)

ID_FIELD = "id"
CONTENT_FIELD = "content"
META_FIELD = "meta_json"
VECTOR_FIELD = "embedding"
COLL_META = "_collection_registry"

_SAFE_TABLE = re.compile(r"^[A-Za-z0-9_]+$")


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if endpoint and "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def _embedding_type(dim: int) -> pa.DataType:
    return pa.list_(pa.field("item", pa.float32(), nullable=False), dim)


def _vector_table_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            (ID_FIELD, pa.string()),
            (CONTENT_FIELD, pa.string()),
            (META_FIELD, pa.string()),
            (VECTOR_FIELD, _embedding_type(dim)),
        ]
    )


def _registry_schema() -> pa.Schema:
    return pa.schema(
        [
            (ID_FIELD, pa.string()),
            ("data", pa.string()),
        ]
    )


class VastDocumentStore(SerializerMixin):
    """
    VastDB vector document store compatible with CDS VastDB call sites.

    Args:
        endpoint: VastDB VIP / Query Engine endpoint (or set ``VASTDB_ENDPOINT``).
        access_key / secret_key: S3-style credentials for VastDB.
        bucket / schema_name: VastDB bucket and schema that hold CDS tables.
        embedding_dim: Vector width (Cosmos-Embed1 uses 256).
        similarity: ``IP`` / ``dot_product``, ``cosine``, or ``l2`` / ``euclidean``.
        adbc_driver_path: Path to ``libadbc_driver_vastdb.so``.
        ssl_verify: TLS verification for the VastDB SDK session.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        *,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        schema_name: Optional[str] = None,
        embedding_dim: int = 256,
        similarity: str = "IP",
        adbc_driver_path: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        # Accepted for pipeline parity with VastDocumentStore (ignored).
        uri: Optional[str] = None,
        api_key: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.endpoint = _normalize_endpoint(
            endpoint
            or os.getenv("VASTDB_ENDPOINT")
            or uri
            or ""
        )
        if not self.endpoint:
            raise ValueError(
                "VastDB endpoint is required (constructor arg or VASTDB_ENDPOINT)"
            )

        self.access_key = (
            access_key
            or os.getenv("VAST_ACCESS_KEY")
            or os.getenv("VASTDB_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        self.secret_key = (
            secret_key
            or os.getenv("VAST_SECRET_KEY")
            or os.getenv("VASTDB_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        if not self.access_key or not self.secret_key:
            raise ValueError("VastDB access_key and secret_key are required")

        self.bucket = bucket or os.getenv("VASTDB_BUCKET", "cds-vastdb")
        self.schema_name = (
            schema_name or os.getenv("VASTDB_SCHEMA", "cds")
        )
        self.embedding_dim = int(
            embedding_dim or int(os.getenv("VASTDB_EMBEDDING_DIM", "256"))
        )
        self.similarity = similarity
        self.metric_type = (
            "COSINE"
            if similarity.lower() == "cosine"
            else "L2"
            if similarity.lower() in ("l2", "euclidean")
            else "IP"
        )
        self.adbc_driver_path = adbc_driver_path or os.getenv(
            "VAST_ADBC_DRIVER_PATH",
            "/opt/adbc-driver/libadbc_driver_vastdb.so",
        )
        if ssl_verify is None:
            ssl_verify = os.getenv("VASTDB_SSL_VERIFY", "false").lower() in (
                "1",
                "true",
                "yes",
            )
        self.ssl_verify = ssl_verify
        # Retained for SerializerMixin / VastDB pipeline parity (unused by VastDB).
        self.uri = uri or self.endpoint
        self.api_key = api_key
        self.database = database
        self.dummy_data = [0.0] * self.embedding_dim

        # Compatibility shim so legacy code that touches ``store.client`` keeps working.
        self.client = self

        import vastdb

        logger.warning(
            "Connecting to VastDB at %s (bucket=%s schema=%s dim=%s metric=%s)",
            self.endpoint,
            self.bucket,
            self.schema_name,
            self.embedding_dim,
            self.metric_type,
        )
        self._session = vastdb.connect(
            endpoint=self.endpoint,
            access=self.access_key,
            secret=self.secret_key,
            ssl_verify=self.ssl_verify,
        )
        self._ensure_collection_meta_exists()
        logger.warning("VastDB metadata registry ready")

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------
    def _validate_table_name(self, name: str) -> str:
        if not name or not _SAFE_TABLE.match(name):
            raise InputValidationError(
                f"Invalid VastDB table/collection name: {name!r}"
            )
        return name

    def _table_ref(self, table_name: str) -> str:
        table_name = self._validate_table_name(table_name)
        return f'"{self.bucket}/{self.schema_name}"."{table_name}"'

    def _prepare_table(self, tx, table_name: str, columns: pa.Schema):
        bucket = tx.bucket(self.bucket)
        schema = bucket.schema(self.schema_name, fail_if_missing=False)
        if schema is None:
            schema = bucket.create_schema(self.schema_name, fail_if_exists=False)
        table = schema.table(table_name, fail_if_missing=False)
        if table is None:
            schema.create_table(table_name, columns=columns)
            table = schema.table(table_name, fail_if_missing=False)
        return table

    def _drop_table(self, table_name: str) -> None:
        table_name = self._validate_table_name(table_name)
        with self._session.transaction() as tx:
            bucket = tx.bucket(self.bucket)
            schema = bucket.schema(self.schema_name, fail_if_missing=False)
            if schema is None:
                return
            table = schema.table(table_name, fail_if_missing=False)
            if table is not None:
                table.drop()

    def list_collections(self) -> List[str]:
        """List VastDB tables in the configured schema (VastDB client parity)."""
        with self._session.transaction() as tx:
            bucket = tx.bucket(self.bucket)
            schema = bucket.schema(self.schema_name, fail_if_missing=False)
            if schema is None:
                return []
            return [t.name for t in schema.tables()]

    def list_collection_names(self) -> List[str]:
        return self.list_collections()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def _ensure_collection_meta_exists(self) -> None:
        with self._session.transaction() as tx:
            self._prepare_table(tx, COLL_META, _registry_schema())

    def _registry_rows(self) -> Dict[str, dict]:
        self._ensure_collection_meta_exists()
        rows: Dict[str, dict] = {}
        with self._session.transaction() as tx:
            table = self._prepare_table(tx, COLL_META, _registry_schema())
            reader = table.select(columns=[ID_FIELD, "data"])
            arrow = reader.read_all()
        for i in range(arrow.num_rows):
            cid = arrow.column(ID_FIELD)[i].as_py()
            raw = arrow.column("data")[i].as_py()
            if isinstance(raw, str):
                raw = _json.loads(raw)
            rows[cid] = raw
        return rows

    def register_collection(self, collection_id: str, data: dict) -> None:
        self._ensure_collection_meta_exists()
        if collection_id in self._registry_rows():
            raise ValueError("Collection ID already registered")
        record = pa.table(
            {
                ID_FIELD: [collection_id],
                "data": [_json.dumps(data)],
            },
            schema=_registry_schema(),
        )
        with self._session.transaction() as tx:
            table = self._prepare_table(tx, COLL_META, _registry_schema())
            table.insert(record)

    def update_collection_meta(self, collection_id: str, updates: dict) -> None:
        rows = self._registry_rows()
        if collection_id not in rows:
            raise KeyError("Collection ID not found in metadata store")
        raw = rows[collection_id]
        raw.update(updates)
        self._delete_registry_ids([collection_id])
        record = pa.table(
            {
                ID_FIELD: [collection_id],
                "data": [_json.dumps(raw)],
            },
            schema=_registry_schema(),
        )
        with self._session.transaction() as tx:
            table = self._prepare_table(tx, COLL_META, _registry_schema())
            table.insert(record)

    def fetch_collection_meta(self, collection_ids: List[str]) -> List[dict]:
        rows = self._registry_rows()
        return [rows[cid] for cid in collection_ids if cid in rows]

    def delete_collection(self, collection_id: str) -> None:
        rows = self._registry_rows()
        if collection_id not in rows:
            raise KeyError("Collection ID not found in metadata store")
        self._delete_registry_ids([collection_id])
        try:
            self.delete_index(collection_id)
        except Exception as exc:  # pragma: no cover – defensive
            logger.warning("Failed dropping VastDB table %s: %s", collection_id, exc)

    def _delete_registry_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        drop = set(ids)
        remaining = {
            k: v for k, v in self._registry_rows().items() if k not in drop
        }
        with self._session.transaction() as tx:
            bucket = tx.bucket(self.bucket)
            schema = bucket.schema(self.schema_name, fail_if_missing=False)
            if schema is None:
                return
            table = schema.table(COLL_META, fail_if_missing=False)
            if table is not None:
                table.drop()
            table = self._prepare_table(tx, COLL_META, _registry_schema())
            if remaining:
                arrow = pa.table(
                    {
                        ID_FIELD: list(remaining.keys()),
                        "data": [_json.dumps(v) for v in remaining.values()],
                    },
                    schema=_registry_schema(),
                )
                table.insert(arrow)

    # ------------------------------------------------------------------
    # Index / collection lifecycle
    # ------------------------------------------------------------------
    def create_index(
        self,
        index: str,
        collection_config: Dict = {},
        index_config: Dict = {},
        metadata_config: MetadataConfig = MetadataConfig(),
    ) -> None:
        """
        Create a VastDB table for a CDS collection.

        ``index_config`` / VastDB GPU_CAGRA settings are ignored — VastDB uses
        native vector distance functions at query time. Typed metadata fields
        from ``metadata_config`` are stored inside ``meta_json`` for flexibility.
        """
        del collection_config, index_config, metadata_config  # unused (API parity)
        index = self._validate_table_name(index)
        logger.warning(
            "Creating VastDB vector table %s (dim=%s)", index, self.embedding_dim
        )
        with self._session.transaction() as tx:
            self._prepare_table(tx, index, _vector_table_schema(self.embedding_dim))

    def delete_index(self, index: str) -> None:
        self._drop_table(index)

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------
    def write_documents(
        self,
        collection_name: str,
        documents: List[Document],
        policy: Optional[DuplicatePolicy] = DuplicatePolicy.NONE,
    ) -> int:
        if not documents:
            return 0
        collection_name = self._validate_table_name(collection_name)

        if policy in (DuplicatePolicy.OVERWRITE, DuplicatePolicy.SKIP):
            existing = self.get_existing_ids(
                collection_name, [d.id for d in documents if d.id]
            )
            if policy == DuplicatePolicy.SKIP and existing:
                skip = set(existing)
                documents = [d for d in documents if d.id not in skip]
            elif policy == DuplicatePolicy.OVERWRITE and existing:
                self.delete_documents(collection_name, existing)

        if not documents:
            return 0

        ids, contents, metas, embeddings = [], [], [], []
        for doc in documents:
            embedding = doc.embedding if doc.embedding is not None else self.dummy_data
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            if len(embedding) != self.embedding_dim:
                raise InputValidationError(
                    f"Embedding dimension {len(embedding)} does not match "
                    f"expected {self.embedding_dim}."
                )
            ids.append(doc.id)
            contents.append(doc.content if doc.content is not None else "")
            metas.append(_json.dumps(doc.meta or {}))
            embeddings.append(embedding.tolist())

        arrow = pa.table(
            {
                ID_FIELD: ids,
                CONTENT_FIELD: contents,
                META_FIELD: metas,
                VECTOR_FIELD: pa.array(
                    embeddings, type=_embedding_type(self.embedding_dim)
                ),
            },
            schema=_vector_table_schema(self.embedding_dim),
        )
        with self._session.transaction() as tx:
            table = self._prepare_table(
                tx, collection_name, _vector_table_schema(self.embedding_dim)
            )
            table.insert(arrow)
        logger.debug(
            "Inserted %d docs into VastDB table %s", len(ids), collection_name
        )
        return len(ids)

    def filter_documents(
        self,
        collection_name: str,
        filters: Optional[Union[Dict[str, Any], str]] = None,
        return_embedding: bool = False,
    ) -> List[Document]:
        if not filters:
            raise InputValidationError(
                "Empty filter is not supported for performance reasons."
            )
        collection_name = self._validate_table_name(collection_name)
        where = self._filters_to_sql(filters)
        columns = [ID_FIELD, CONTENT_FIELD, META_FIELD]
        if return_embedding:
            # Vector columns are excluded from SDK select; fetch via ADBC.
            return self._filter_via_adbc(
                collection_name, where, return_embedding=True
            )

        docs: List[Document] = []
        with self._session.transaction() as tx:
            table = self._prepare_table(
                tx, collection_name, _vector_table_schema(self.embedding_dim)
            )
            # SDK select cannot push arbitrary SQL; load and filter in Python for
            # the common id / meta equality cases used by CDS delete paths.
            reader = table.select(columns=columns)
            arrow = reader.read_all()

        for i in range(arrow.num_rows):
            row = {
                ID_FIELD: arrow.column(ID_FIELD)[i].as_py(),
                CONTENT_FIELD: arrow.column(CONTENT_FIELD)[i].as_py(),
                META_FIELD: arrow.column(META_FIELD)[i].as_py(),
            }
            if self._row_matches(row, filters):
                docs.append(self._row_to_document(row, embedding=None))
        return docs

    def delete_documents(self, collection_name: str, document_ids: List[str]) -> None:
        if not document_ids:
            return
        collection_name = self._validate_table_name(collection_name)
        # Rewrite table without the deleted IDs (portable across SDK versions).
        keep_ids = set(self.get_existing_ids(collection_name, document_ids))
        if not keep_ids:
            return
        all_docs = self._scan_documents(collection_name, return_embedding=True)
        remaining = [d for d in all_docs if d.id not in set(document_ids)]
        self._drop_table(collection_name)
        self.create_index(collection_name)
        if remaining:
            self.write_documents(
                collection_name, remaining, policy=DuplicatePolicy.NONE
            )

    def delete_documents_by_filter(
        self,
        collection_name: str,
        filters: Optional[Union[Dict[str, Any], str]] = None,
    ) -> int:
        matched = self.filter_documents(
            collection_name, filters=filters, return_embedding=False
        )
        if not matched:
            return 0
        self.delete_documents(collection_name, [d.id for d in matched])
        return len(matched)

    def get_embedding_count(self, collection_name: str) -> int:
        collection_name = self._validate_table_name(collection_name)
        with self._session.transaction() as tx:
            table = self._prepare_table(
                tx, collection_name, _vector_table_schema(self.embedding_dim)
            )
            reader = table.select(columns=[ID_FIELD])
            return reader.read_all().num_rows

    def get_existing_ids(
        self, collection_name: str, ids: List, timeout: Optional[float] = None
    ) -> List:
        del timeout
        if not ids:
            return []
        id_set = {str(i) for i in ids}
        found: List[str] = []
        with self._session.transaction() as tx:
            table = self._prepare_table(
                tx, collection_name, _vector_table_schema(self.embedding_dim)
            )
            reader = table.select(columns=[ID_FIELD])
            arrow = reader.read_all()
        for i in range(arrow.num_rows):
            cid = arrow.column(ID_FIELD)[i].as_py()
            if cid in id_set:
                found.append(cid)
        return found

    def get_metadata_schema(self, collection_name: str) -> MetadataConfig:
        del collection_name
        return MetadataConfig(allow_dynamic_schema=True, fields=[])

    # ------------------------------------------------------------------
    # Vector search (ADBC)
    # ------------------------------------------------------------------
    def _distance_expr(self, dim: int, vector_sql: str) -> str:
        cast_col = f"{VECTOR_FIELD}::FLOAT[{dim}]"
        cast_q = f"ARRAY[{vector_sql}]::FLOAT[{dim}]"
        if self.metric_type == "L2":
            return f"array_distance({cast_col}, {cast_q})"
        if self.metric_type == "COSINE":
            return f"array_cosine_distance({cast_col}, {cast_q})"
        # IP / inner product — lower negative-IP == higher similarity
        return f"array_negative_inner_product({cast_col}, {cast_q})"

    def _score_from_distance(self, distance: float) -> float:
        if self.metric_type == "COSINE":
            return 1.0 - float(distance)
        if self.metric_type == "L2":
            return -float(distance)
        # IP: distance is negative inner product → negate so higher is better
        return -float(distance)

    def _adbc_connect(self):
        import adbc_driver_manager.dbapi

        return adbc_driver_manager.dbapi.connect(
            driver=self.adbc_driver_path,
            db_kwargs={
                "vast.db.endpoint": self.endpoint,
                "vast.db.access_key": self.access_key,
                "vast.db.secret_key": self.secret_key,
            },
        )

    def _embedding_retrieval(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        filters: Optional[Union[Dict[str, Any], str]] = None,
        top_k: int = 10,
        search_params: Dict = {},
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        del search_params  # VastDB uses native distance; nprobe/CAGRA unused
        collection_name = self._validate_table_name(collection_name)
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if len(query_embedding) != self.embedding_dim:
            raise InputValidationError(
                f"Embedding dimension {len(query_embedding)} does not match "
                f"expected {self.embedding_dim}."
            )
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        dim = self.embedding_dim
        vector_sql = ",".join(repr(float(x)) for x in query_embedding.tolist())
        distance_expr = self._distance_expr(dim, vector_sql)
        table_ref = self._table_ref(collection_name)
        where = ""
        if filters:
            clause = self._filters_to_sql(filters)
            if clause:
                where = f"WHERE {clause}"

        select_cols = f"{ID_FIELD}, {CONTENT_FIELD}, {META_FIELD}"
        if return_embedding:
            select_cols += f", {VECTOR_FIELD}"

        sql = (
            f"SELECT {select_cols}, {distance_expr} AS distance "
            f"FROM {table_ref} {where} "
            f"ORDER BY distance ASC LIMIT {int(top_k)}"
        )
        logger.debug("VastDB vector search SQL: %s", sql)

        docs: List[Document] = []
        with self._adbc_connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                arrow = cursor.fetch_arrow_table()

        for i in range(arrow.num_rows):
            row = {
                ID_FIELD: arrow.column(ID_FIELD)[i].as_py(),
                CONTENT_FIELD: arrow.column(CONTENT_FIELD)[i].as_py(),
                META_FIELD: arrow.column(META_FIELD)[i].as_py(),
            }
            emb = None
            if return_embedding and VECTOR_FIELD in arrow.column_names:
                emb = arrow.column(VECTOR_FIELD)[i].as_py()
            doc = self._row_to_document(row, embedding=emb)
            doc.score = self._score_from_distance(arrow.column("distance")[i].as_py())
            docs.append(doc)
        return docs

    def _filter_via_adbc(
        self, collection_name: str, where: str, return_embedding: bool
    ) -> List[Document]:
        table_ref = self._table_ref(collection_name)
        cols = f"{ID_FIELD}, {CONTENT_FIELD}, {META_FIELD}"
        if return_embedding:
            cols += f", {VECTOR_FIELD}"
        sql = f"SELECT {cols} FROM {table_ref}"
        if where:
            sql += f" WHERE {where}"
        docs: List[Document] = []
        with self._adbc_connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                arrow = cursor.fetch_arrow_table()
        for i in range(arrow.num_rows):
            row = {
                ID_FIELD: arrow.column(ID_FIELD)[i].as_py(),
                CONTENT_FIELD: arrow.column(CONTENT_FIELD)[i].as_py(),
                META_FIELD: arrow.column(META_FIELD)[i].as_py(),
            }
            emb = None
            if return_embedding and VECTOR_FIELD in arrow.column_names:
                emb = arrow.column(VECTOR_FIELD)[i].as_py()
            docs.append(self._row_to_document(row, embedding=emb))
        return docs

    def _scan_documents(
        self, collection_name: str, return_embedding: bool = False
    ) -> List[Document]:
        return self._filter_via_adbc(collection_name, where="", return_embedding=return_embedding)

    # ------------------------------------------------------------------
    # Filter helpers (subset used by CDS APIs)
    # ------------------------------------------------------------------
    def _filters_to_sql(self, filters: Union[Dict[str, Any], str]) -> str:
        if isinstance(filters, str):
            return filters
        if not isinstance(filters, dict):
            return ""

        # Haystack-style: {"field": "id", "operator": "==", "value": "..."}
        if "field" in filters and "operator" in filters:
            return self._single_filter_sql(filters)

        # Nested conditions
        if "conditions" in filters:
            op = filters.get("operator", "AND").upper()
            parts = [
                self._filters_to_sql(c)
                for c in filters["conditions"]
                if self._filters_to_sql(c)
            ]
            if not parts:
                return ""
            return "(" + f" {op} ".join(parts) + ")"
        return ""

    def _single_filter_sql(self, filt: Dict[str, Any]) -> str:
        field = str(filt.get("field", ""))
        operator = str(filt.get("operator", "=="))
        value = filt.get("value")
        # Map Haystack meta.* onto JSON text contains for VastDB.
        if field.startswith("meta."):
            key = field[len("meta.") :]
            # Store is opaque JSON; equality approximated via LIKE on serialized meta.
            literal = _json.dumps(value)
            # Escape single quotes for SQL string literal.
            needle = _json.dumps({key: value})[1:-1].replace("'", "''")
            if operator in ("==", "eq"):
                return f"{META_FIELD} LIKE '%{needle}%'"
            if operator in ("!=", "ne"):
                return f"{META_FIELD} NOT LIKE '%{needle}%'"
            del literal
            return ""

        col = ID_FIELD if field in (ID_FIELD, "id") else field
        if col not in (ID_FIELD, CONTENT_FIELD):
            # Unknown columns are not first-class; skip rather than fail hard.
            return ""

        def _sql_literal(v: Any) -> str:
            if v is None:
                return "NULL"
            if isinstance(v, str):
                return "'" + v.replace("'", "''") + "'"
            return repr(v)

        if operator in ("in", "IN"):
            if not isinstance(value, (list, tuple, set)):
                return ""
            literals = ", ".join(_sql_literal(v) for v in value)
            return f"{col} IN ({literals})" if literals else ""

        lit = _sql_literal(value)
        op_map = {"==": "=", "eq": "=", "!=": "<>", "ne": "<>"}
        sql_op = op_map.get(operator, operator)
        if lit == "NULL":
            return f"{col} IS NULL" if sql_op in ("=", "==") else f"{col} IS NOT NULL"
        return f"{col} {sql_op} {lit}"

    def _row_matches(
        self, row: Dict[str, Any], filters: Optional[Union[Dict[str, Any], str]]
    ) -> bool:
        if filters is None:
            return True
        if isinstance(filters, str):
            # String filters are evaluated only via ADBC path.
            return True
        if "field" in filters and "operator" in filters:
            field = filters["field"]
            op = filters["operator"]
            value = filters["value"]
            if field in (ID_FIELD, "id"):
                actual = row.get(ID_FIELD)
            elif field.startswith("meta."):
                meta = row.get(META_FIELD) or "{}"
                if isinstance(meta, str):
                    meta = _json.loads(meta)
                actual = meta.get(field[len("meta.") :])
            else:
                actual = row.get(field)
            if op in ("==", "eq"):
                return actual == value
            if op in ("!=", "ne"):
                return actual != value
            if op in ("in", "IN"):
                try:
                    return actual in value
                except TypeError:
                    return False
            return False
        if "conditions" in filters:
            op = filters.get("operator", "AND").upper()
            results = [self._row_matches(row, c) for c in filters["conditions"]]
            return all(results) if op == "AND" else any(results)
        return True

    @staticmethod
    def _row_to_document(
        row: Dict[str, Any], embedding: Optional[List[float]]
    ) -> Document:
        meta_raw = row.get(META_FIELD) or "{}"
        if isinstance(meta_raw, str):
            try:
                meta = _json.loads(meta_raw) if meta_raw else {}
            except _json.JSONDecodeError:
                meta = {}
        else:
            meta = meta_raw or {}
        doc = Document(
            id=row.get(ID_FIELD),
            content=row.get(CONTENT_FIELD) or None,
            meta=meta,
        )
        if embedding is not None:
            doc.embedding = embedding
        return doc

    # Bulk-insert APIs are VastDB-specific; stub for duck-typing callers.
    def bulk_insert_files(self, collection_name: str, file_paths: List[str], **storage_opts):
        raise NotImplementedError(
            "VastDB bulk parquet import is not implemented in this connector. "
            "Use the CDS ingest API or a VAST DataEngine writer pipeline."
        )

    def get_bulk_insert_state(self, job_id: int):
        raise NotImplementedError("VastDB does not use VastDB bulk-insert jobs")

    def list_bulk_insert_tasks(self, limit=None, collection_name=None):
        return []
