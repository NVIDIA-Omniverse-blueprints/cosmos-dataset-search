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
Milvus‑backed Haystack document store with a universal metadata registry.

Key points
----------
* `_collection_registry` is a single Milvus collection that stores every
  `Collection` record (JSON blob) irrespective of pipeline.
* All CRUD helpers route through `using=self.client._using`, mirroring the
  pattern used by Milvus bulk‑insert utilities.
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from pymilvus import (
    BulkInsertState,
    Collection,
    DataType,
    MilvusClient,
    connections,
    utility,
)
from pymilvus.client.types import LoadState

from src.haystack.components.milvus.filter_utils import LogicalFilterClause
from src.haystack.components.milvus.schema_utils import MetadataConfig, MetadataField
from src.haystack.serializer import SerializerMixin
from src.visual_search.exceptions import InputValidationError

logger = logging.getLogger(__name__)

ID_FIELD = "id"
VECTOR_FIELD = "embedding"
COLL_META = "_collection_registry"  # global—not pipeline‑scoped


# --------------------------------------------------------------------------- #
# Universal registry helpers (internal)                                       #
# --------------------------------------------------------------------------- #
def _ensure_collection_meta_exists(store: "MilvusDocumentStore") -> None:
    logger.warning("Ensuring collection metadata registry exists in Milvus")
    client = store.client
    if COLL_META in client.list_collections():
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("data", DataType.JSON)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=2)

    client.create_collection(
        collection_name=COLL_META,
        schema=schema,
        consistency_level="Strong",
        using=client._using,
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="FLAT",
        metric_type=store.metric_type,  # Use store's metric_type for consistency
        index_name="dummy_flat",
    )
    client.create_index(
        collection_name=COLL_META,
        index_params=index_params,
        using=client._using,
    )
    client.load_collection(collection_name=COLL_META, using=store.client._using)


def _get_collection_meta_by_id(
    store: "MilvusDocumentStore", coll_ids: List[str]
) -> List[dict]:
    logger.warning("Fetching metadata for collections %s", coll_ids)
    store.client.load_collection(collection_name=COLL_META, using=store.client._using)
    logger.warning(
        "Collection should be loaded",
    )
    collections = store.client.get(
        collection_name=COLL_META, ids=coll_ids, using=store.client._using
    )
    if len(collections) == 0:
        logger.warning("No metadata found for collections %s", coll_ids)
        return []
    if len(collections) > 1:
        result = [collection["data"] for collection in collections]
    elif isinstance(collections, dict) and "data" in collections:
        result = [collections["data"]]
    elif isinstance(collections, list) and len(collections) == 1:
        result = [collections[0]["data"]]
    else:
        result = [collections]
    return result


def _meta_insert(store: "MilvusDocumentStore", coll_id: str, data: dict):
    logger.warning("Inserting metadata for collection %s: %s", coll_id, data)
    store.client.insert(
        collection_name=COLL_META,
        data=[{"id": coll_id, "embedding": [0, 0], "data": data}],
        using=store.client._using,
    )


def _meta_delete(store: "MilvusDocumentStore", coll_id: str):
    if coll_id == COLL_META:
        raise ValueError("Cannot delete the metadata registry collection itself.")
    store.client.delete(
        collection_name=COLL_META,
        filter=f"id == '{coll_id}'",
        using=store.client._using,
    )
    store.client.drop_collection(collection_name=coll_id)


# --------------------------------------------------------------------------- #
# Main store                                                                   #
# --------------------------------------------------------------------------- #
class MilvusDocumentStore(SerializerMixin):
    def __init__(
        self,
        uri: str,
        *,
        api_key: Optional[str] = None,
        database: Optional[str] = None,
        embedding_dim: int = 768,
        similarity: str = "dot_product",
    ):
        """
        Initialize a MilvusDocumentStore instance.

        Args:
            uri: The connection URI for the Milvus instance.
            api_key: Optional API key for authentication.
            database: Optional database name to connect to.
            embedding_dim: Dimension of the embedding vectors (default: 768).
            similarity: Similarity metric to use. Options: "dot_product", "cosine", "l2", "euclidean" (default: "dot_product").
        """
        self.uri = uri
        self.api_key = api_key
        self.database = database
        self.embedding_dim = embedding_dim

        self.similarity = similarity
        self.metric_type = (
            "COSINE"
            if similarity == "cosine"
            else "L2" if similarity in ("l2", "euclidean") else "IP"
        )
        self.dummy_data = [0] * embedding_dim

        conn: dict[str, Union[str, bool, None]] = {
            "uri": uri,
            "user": os.getenv("MILVUS_USERNAME"),
            "password": os.getenv("MILVUS_PASSWORD"),
            "db_name": database,
        }
        if os.getenv("MILVUS_CERTS_PATH"):
            p = os.getenv("MILVUS_CERTS_PATH")
            conn.update(
                secure=True,
                client_pem_path=f"{p}/client.pem",
                client_key_path=f"{p}/client.key",
                ca_pem_path=f"{p}/ca.pem",
                server_name=os.getenv("MILVUS_SERVER"),
            )
        if api_key:
            conn["token"] = api_key

        logger.warning(
            "Connecting to Milvus at %s with database %s",
            uri,
            database,
        )
        self.client = MilvusClient(**conn)  # type: ignore
        logger.warning(
            "Connected to Milvus at %s with database %s",
            uri,
            database,
        )
        logger.warning(
            "Calling _ensure_collection_meta_exists to initialize metadata registry"
        )
        _ensure_collection_meta_exists(self)
        logger.warning("Metadata registry initialized successfully")

    # ---------------------------------------------------------------
    # Standard doc‑level operations
    # ---------------------------------------------------------------
    def write_documents(
        self,
        collection_name: str,
        documents: List[Document],
        policy: Optional[DuplicatePolicy] = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents to a Milvus collection.

        Args:
            collection_name: Name of the collection to write documents to.
            documents: List of Document objects to write.
            policy: Policy for handling duplicate documents (currently not implemented).

        Returns:
            Number of documents successfully inserted.
        """
        logger.warning(
            "Writing %d documents to collection %s with policy %s",
            len(documents),
            collection_name,
            policy,
        )
        if not documents:
            return 0

        docs_modified = []
        for doc in documents:
            embedding = doc.embedding if doc.embedding is not None else self.dummy_data
            if embedding is not None and hasattr(embedding, "__len__"):
                embedding = np.array(embedding, dtype=np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.tolist()

            doc_dict = {
                ID_FIELD: doc.id,
                VECTOR_FIELD: embedding,
            }
            if doc.content is not None:
                doc_dict["content"] = doc.content
            for k, v in doc.meta.items():
                if v is not None:
                    doc_dict[k] = v
            docs_modified.append(doc_dict)

        collection = Collection(name=collection_name, using=self.client._using)
        # Note: We use insert here instead of upsert for performance reasons with large datasets.
        # The API already handles deduplication by deleting before inserting, so insert is sufficient.
        logger.debug(
            "Inserting %d docs into collection %s", len(docs_modified), collection_name
        )
        res = collection.insert(
            data=docs_modified,
        )
        logger.debug(
            "Inserted %d docs into collection %s", res.insert_count, collection_name
        )
        return res.insert_count

    def filter_documents(
        self,
        collection_name: str,
        filters: Optional[Union[Dict[str, Any], str]] = None,
        return_embedding: bool = False,
    ) -> List[Document]:
        """
        Filter and retrieve documents from a collection based on metadata filters.

        Args:
            collection_name: Name of the collection to search in.
            filters: Either a Milvus filter string or a dictionary of filters to apply.
            return_embedding: Whether to include embeddings in the returned documents.

        Returns:
            List of Document objects matching the filters.

        Raises:
            InputValidationError: If no filters are provided.
        """
        if not filters:
            raise InputValidationError(
                "Empty filter is not supported for performance reasons."
            )

        if isinstance(filters, str):
            milvus_filter = filters
        else:
            milvus_filter = LogicalFilterClause.parse(filters).convert_to_milvus()

        if return_embedding:
            ids = [
                x[ID_FIELD]
                for x in self.client.query(
                    collection_name=collection_name,
                    filter=milvus_filter,
                    output_fields=[ID_FIELD],
                    using=self.client._using,
                )
            ]
            res = self.client.get(
                collection_name=collection_name,
                ids=ids,
                using=self.client._using,
            )
            res_docs = []
            for hit in res:
                embed = hit.pop(VECTOR_FIELD, None)
                doc = Document.from_dict(hit)
                doc.embedding = embed
                res_docs.append(doc)
            return res_docs

        res = self.client.query(
            collection_name=collection_name,
            filter=milvus_filter,
            output_fields=["*"],
            using=self.client._using,
        )
        res_docs = []
        for hit in res:
            hit.pop(VECTOR_FIELD, None)
            res_docs.append(Document.from_dict(hit))
        return res_docs

    # ---------------------------------------------------------------
    # Search helpers
    # ---------------------------------------------------------------
    def get_embedding_count(self, collection_name: str) -> int:
        """
        Get the total number of embeddings in a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Number of entities (embeddings) in the collection.
        """
        collection = Collection(name=collection_name, using=self.client._using)
        return collection.num_entities

    def get_existing_ids(
        self, collection_name: str, ids: List, timeout: Optional[float] = None
    ) -> List:
        """
        Check which of the provided IDs exist in the collection.

        Args:
            collection_name: Name of the collection to check.
            ids: List of document IDs to check for existence.
            timeout: Optional timeout for the operation.

        Returns:
            List of IDs that exist in the collection.
        """
        res = self.client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=[ID_FIELD],
            timeout=timeout,
            using=self.client._using,
        )
        return [x[ID_FIELD] for x in res]

    def _embedding_retrieval(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        filters: Optional[Union[Dict[str, Any], str]] = None,
        top_k: int = 10,
        search_params: Dict = {},
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        if len(query_embedding) != self.embedding_dim:
            raise InputValidationError(
                f"Embedding dimension {len(query_embedding)} does not match expected {self.embedding_dim}."
            )

        load_state = self.client.get_load_state(collection_name=collection_name)
        if load_state.get("state") != LoadState.Loaded:
            raise InputValidationError(f"Collection {collection_name} is not loaded.")

        milvus_filter = None
        if filters:
            milvus_filter = (
                filters
                if isinstance(filters, str)
                else LogicalFilterClause.parse(filters).convert_to_milvus()
            )

        if "nprobe" not in search_params:
            search_params["nprobe"] = 32

        group_by_field = search_params.get("group_by_field")
        res = self.client.search(
            collection_name=collection_name,
            data=[query_embedding],
            filter=milvus_filter,
            output_fields=["*"],
            group_by_field=group_by_field,
            search_params={"metric_type": self.metric_type, "params": search_params},
            limit=top_k,
            using=self.client._using,
        )
        if not res:
            return []

        res = res[0]
        res_vectors = {}
        if return_embedding:
            ids = [x["entity"]["id"] for x in res]
            vecs = self.client.get(
                collection_name=collection_name,
                ids=ids,
                output_fields=[VECTOR_FIELD],
                using=self.client._using,
            )
            res_vectors = {val[ID_FIELD]: val[VECTOR_FIELD] for val in vecs}

        docs = []
        for hit in res:
            doc = Document.from_dict(hit["entity"])
            doc.score = hit["distance"]
            doc.embedding = res_vectors.get(hit[ID_FIELD]) if return_embedding else None
            docs.append(doc)
        return docs

    # ---------------------------------------------------------------
    # Deletion helpers
    # ---------------------------------------------------------------
    def delete_documents(self, collection_name: str, document_ids: List[str]) -> None:
        """
        Delete documents from a collection by their IDs.

        Args:
            collection_name: Name of the collection.
            document_ids: List of document IDs to delete.
        """
        self.client.delete(
            collection_name=collection_name,
            pks=document_ids,
            using=self.client._using,
        )

    def delete_documents_by_filter(
        self,
        collection_name: str,
        filters: Optional[Union[Dict[str, Any], str]] = None,
    ) -> int:
        """
        Delete documents from a collection based on metadata filters.

        Args:
            collection_name: Name of the collection.
            filters: Either a Milvus filter string or a dictionary of filters to apply.

        Returns:
            Number of documents deleted.

        Raises:
            InputValidationError: If no filters are provided.
        """
        if not filters:
            raise InputValidationError(
                "Empty filter is not supported for performance reasons."
            )

        milvus_filter = (
            filters
            if isinstance(filters, str)
            else LogicalFilterClause.parse(filters).convert_to_milvus()
        )
        res = self.client.delete(
            collection_name=collection_name,
            filter=milvus_filter,
            using=self.client._using,
        )
        return res.get("delete_count", 0)

    def delete_index(self, index: str) -> None:
        """
        Drop (delete) a Milvus collection.

        Args:
            index: Name of the collection to drop.
        """
        self.client.drop_collection(index, using=self.client._using)

    # ---------------------------------------------------------------
    # Collection creation
    # ---------------------------------------------------------------
    def create_index(
        self,
        index: str,
        collection_config: Dict = {},
        index_config: Dict = {},
        metadata_config: MetadataConfig = MetadataConfig(),
    ) -> None:
        """
        Create a new Milvus collection with specified configuration.

        Args:
            index: Name of the collection to create.
            collection_config: Milvus collection configuration parameters.
            index_config: Vector index configuration parameters.
            metadata_config: Configuration for metadata fields including schema definition.
        """
        logger.warning(
            "Creating collection %s with config: %s, index config: %s, metadata config: %s",
            index,
            collection_config,
            index_config,
            metadata_config,
        )
        if index in self.client.list_collections():
            return

        # build schema
        schema = MilvusClient.create_schema(
            auto_id=False, enable_dynamic_field=metadata_config.allow_dynamic_schema
        )
        schema.add_field(ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=255)
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self.embedding_dim)

        for field in metadata_config.fields:
            params = {}
            if field.max_length:
                params["max_length"] = field.max_length
            if field.max_capacity:
                params["max_capacity"] = field.max_capacity
            if field.element_dtype:
                params["element_type"] = DataType[field.element_dtype.value]

            schema.add_field(
                field_name=field.name,
                datatype=DataType[field.dtype.value],
                is_partition_key=field.is_partition_key,
                **params,
            )
        logger.warning("Creating a collection using the following configuration:")
        logger.warning("Name: %s", index)
        logger.warning("Schema: %s", schema)
        logger.warning("Collection config: %s", collection_config)
        self.client.create_collection(
            collection_name=index,
            schema=schema,
            using=self.client._using,
            **collection_config,
        )

        # index_config["metric_type"] = self.metric_type
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name=VECTOR_FIELD, **index_config)
        logger.warning("Creating index:")
        logger.warning("Parameters: %s", index_params)
        self.client.create_index(
            collection_name=index,
            index_params=index_params,
            using=self.client._using,
        )
        self.client.load_collection(collection_name=index, using=self.client._using)

    # ---------------------------------------------------------------
    # Metadata‑registry CRUD (universal across pipelines)
    # ---------------------------------------------------------------
    def register_collection(self, collection_id: str, data: dict) -> None:
        """
        Insert a new collection metadata entry. Raises ValueError if id exists.
        """
        _ensure_collection_meta_exists(self)
        if _get_collection_meta_by_id(self, [collection_id]):
            raise ValueError("Collection ID already registered")
        _meta_insert(self, collection_id, data)

    def update_collection_meta(self, collection_id: str, updates: dict) -> None:
        """
        Patch an existing collection metadata entry.
        """
        _ensure_collection_meta_exists(self)
        record = _get_collection_meta_by_id(self, [collection_id])
        if not record:
            raise KeyError("Collection ID not found in metadata store")
        raw = record[0]
        if isinstance(raw, str):
            raw = _json.loads(raw)
        raw.update(updates)
        
        # Use Milvus native upsert (available since 2.3+) instead of manual delete-then-insert
        # This is more efficient and cleaner than our previous approach
        logger.warning("Upserting metadata for collection %s: %s", collection_id, raw)
        self.client.upsert(
            collection_name=COLL_META,
            data=[{"id": collection_id, "embedding": [0, 0], "data": raw}],
            using=self.client._using,
        )

    def fetch_collection_meta(self, collection_ids: List[str]) -> List[dict]:
        """
        Return collection metadata list, or empty list if not present.
        """
        _ensure_collection_meta_exists(self)
        record = _get_collection_meta_by_id(self, collection_ids)
        if len(record) == 0:
            return []
        if isinstance(record, str):
            record = _json.loads(record)
        return record

    def delete_collection(self, collection_id: str) -> None:
        """
        Remove collection metadata entry (no error if missing).
        """
        _ensure_collection_meta_exists(self)
        try:
            _meta_delete(self, collection_id)
        except Exception:
            raise KeyError("Collection ID not found in metadata store")

    # ---------------------------------------------------------------
    # Bulk‑insert helpers (unchanged except for alias usage)
    # ---------------------------------------------------------------
    def bulk_insert_files(
        self,
        collection_name: str,
        file_paths: List[str],
        **storage_opts,
    ) -> List[int]:
        """
        Perform bulk insert of files into a Milvus collection.

        Args:
            collection_name: Name of the collection to insert into.
            file_paths: List of file paths to insert (typically Parquet files).
            **storage_opts: Additional storage options passed to Milvus bulk insert.

        Returns:
            List of job IDs for the bulk insert operations.
        """
        return utility.do_bulk_insert(
            collection_name=collection_name,
            files=file_paths,
            using=self.client._using,
            **storage_opts,
        )

    def get_bulk_insert_state(self, job_id: int):
        """
        Get the status of a bulk insert job.

        Args:
            job_id: The job ID returned from bulk_insert_files.

        Returns:
            Bulk insert job state object containing status information.
        """
        return utility.get_bulk_insert_state(job_id, using=self.client._using)

    def list_bulk_insert_tasks(
        self,
        limit: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """
        List bulk insert tasks using Milvus REST API v2.

        Milvus 2.4.x has a known bug (Issue #38172) where utility.list_bulk_insert_tasks()
        returns corrupted data (task_id=0). We use the REST API instead.

        Args:
            limit: Maximum number of tasks to return. None means unlimited.
            collection_name: Filter tasks by collection name.

        Returns:
            List of bulk insert task information.
        """
        # Extract host and port from URI
        uri = self.uri
        if uri.startswith("http://") or uri.startswith("https://"):
            base_url = uri.split("/")[0] + "//" + uri.split("/")[2]
        else:
            # Handle host:port format
            base_url = f"http://{uri.split('/')[0]}"

        # Call Milvus REST API v2
        url = f"{base_url}/v2/vectordb/jobs/import/list"
        payload = {}
        if collection_name:
            payload["collectionName"] = collection_name

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logger.warning("Milvus REST API returned code %s", data.get("code"))
                return []

            records = data.get("data", {}).get("records", [])

            # Map REST API response to BulkInsertState objects
            state_map = {
                "Pending": BulkInsertState.ImportPending,
                "InProgress": BulkInsertState.ImportStarted,
                "Completed": BulkInsertState.ImportCompleted,
                "Failed": BulkInsertState.ImportFailed,
            }

            class ImportJob:
                def __init__(self, record):
                    self.task_id = int(record.get("jobId", 0))
                    self.collection_name = record.get("collectionName", "")
                    self.progress = record.get("progress", 0)
                    self.state = state_map.get(
                        record.get("state"), BulkInsertState.ImportUnknownState
                    )
                    self.state_name = record.get("state", "Unknown")
                    self.files = []  # REST API doesn't return file list
                    self.failed_reason = record.get("reason", "")

            tasks = [ImportJob(r) for r in records]

            # Apply limit if specified
            if limit and limit > 0:
                tasks = tasks[:limit]

            return tasks

        except Exception as e:
            logger.warning("Error calling Milvus REST API: %s", e)
            return []

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------
    def _cast_field(self, field: Dict[str, Any]) -> MetadataField:
        params = field["params"]
        element_dtype = field.get("element_type")
        return MetadataField(
            name=field["name"],
            dtype=field["type"].name,
            is_partition_key=field.get("is_partition_key", False),
            element_dtype=element_dtype.name if element_dtype is not None else None,
            max_length=params.get("max_length"),
            max_capacity=params.get("max_capacity"),
        )

    def get_metadata_schema(self, collection_name: str) -> MetadataConfig:
        """
        Retrieve the metadata schema configuration for a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            MetadataConfig object containing the collection's metadata field definitions.
        """
        collection_info = self.client.describe_collection(
            collection_name, using=self.client._using
        )
        return MetadataConfig(
            allow_dynamic_schema=collection_info["enable_dynamic_field"],
            fields=[
                self._cast_field(f)
                for f in collection_info["fields"]
                if f["name"] not in (ID_FIELD, VECTOR_FIELD)
            ],
        )
