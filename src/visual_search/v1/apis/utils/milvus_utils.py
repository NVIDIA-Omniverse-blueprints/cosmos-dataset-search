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

# Copyright 2025, NVIDIA Corporation. All rights reserved.

import re
from typing import Any, Dict, Optional

# Third-Party Imports
import pyarrow as pa
import pyarrow.parquet as pq
from pymilvus import Collection, CollectionSchema, MilvusException, connections, utility

from src.visual_search.common.s3_import import open_s3_import

# Local Application Imports
from src.visual_search.logger import logger


# Define a custom exception for utility errors
class MilvusServiceError(Exception):
    pass


def create_safe_name(name: str) -> str:
    """
    Creates a Milvus-compatible collection or alias name.
    Replaces hyphens with underscores and prepends 'a' if it starts with a digit or is empty.
    Milvus names must start with a letter or underscore.
    """
    if not name:
        raise ValueError("Input name cannot be empty.")
    safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    if not re.match(r"^[a-zA-Z_]", safe_name):
        safe_name = f"a{safe_name}"
    if len(safe_name) > 255:
        safe_name = safe_name[:255]  # Truncate if too long
    return safe_name


def get_milvus_connection_details(milvus_uri: str) -> Dict[str, Any]:
    """Parses Milvus URI string into host and port."""
    try:
        if "://" in milvus_uri:
            uri_part = milvus_uri.split("://")[-1]
        else:
            uri_part = milvus_uri

        if ":" not in uri_part:
            raise ValueError(
                "Milvus URI must include host and port (e.g., 'localhost:19530')"
            )

        host, port_str = uri_part.split(":", 1)
        port = int(port_str)
        return {"host": host, "port": port}
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid Milvus URI format: '{milvus_uri}'. Expected 'host:port'. Error: {e}"
        )


def ensure_milvus_connection(milvus_uri: str, alias: str = "default") -> None:
    """Establishes or verifies connection to Milvus."""
    if not connections.has_connection(alias):
        try:
            conn_details = get_milvus_connection_details(milvus_uri)
            logger.info(
                f"Attempting to connect to Milvus ({alias}) at {conn_details['host']}:{conn_details['port']}..."
            )
            connections.connect(alias=alias, **conn_details)
            if not utility.has_collection(""):
                pass
            logger.info(f"Milvus connection ({alias}) established.")
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus ({alias}) at {milvus_uri}: {e}")
            raise MilvusServiceError(f"Milvus connection error: {e}")
        except ValueError as e:
            logger.error(f"Invalid Milvus URI '{milvus_uri}': {e}")
            raise MilvusServiceError(f"Invalid Milvus URI: {e}")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during Milvus connection ({alias}): {e}",
                exc_info=True,
            )
            raise MilvusServiceError(f"Unexpected Milvus connection error: {e}")
    else:
        logger.debug(f"Milvus connection ({alias}) already exists.")


def build_storage_options(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build credential options for the guarded S3 reader and Milvus."""
    storage_options: Dict[str, Any] = {}
    client_kwargs: Dict[str, Any] = {}

    if access_key and secret_key:
        storage_options["key"] = access_key
        storage_options["secret"] = secret_key
    if endpoint_url:
        client_kwargs["endpoint_url"] = str(endpoint_url)

    if client_kwargs:
        storage_options["client_kwargs"] = client_kwargs

    return storage_options


async def validate_parquet_schema(
    file_path: str,
    collection_name: str,
    storage_options: Optional[Dict[str, Any]] = None,
    alias: str = "default",
) -> None:
    """
    Validates the schema of a Parquet file against a Milvus collection schema.
    Uses guarded S3 range reads with provided storage options.
    """
    storage_options = storage_options or {}
    logger.debug(
        f"Validating schema for file '{file_path}' against collection '{collection_name}'"
    )

    try:
        # 1. Get Milvus Collection Schema
        ensure_milvus_connection("", alias=alias)
        collection: Collection = Collection(collection_name, using=alias)
        collection_schema: CollectionSchema = collection.schema
        collection_fields = {field.name for field in collection_schema.fields}
        logger.debug(
            f"Milvus collection '{collection_name}' fields: {collection_fields}"
        )

        # 2. Get Parquet File Schema through the guarded S3 transport.
        try:
            with open_s3_import(file_path, storage_options) as f:
                parquet_schema = pq.read_schema(f)
            parquet_fields = set(parquet_schema.names)
            logger.debug(f"Parquet file '{file_path}' fields: {parquet_fields}")
        except FileNotFoundError:
            logger.error(f"File not found via fsspec: {file_path}")
            raise
        except Exception as e:
            logger.error(
                f"Error reading Parquet file schema from '{file_path}': {e}",
                exc_info=True,
            )
            raise MilvusServiceError(
                f"Failed to read Parquet schema from '{file_path}': {e}"
            )

        # 3. Compare Schemas (Check if all collection fields are in Parquet)
        missing_fields = collection_fields - parquet_fields
        if missing_fields:
            logger.error(
                f"Schema mismatch: Parquet file '{file_path}' is missing required fields for collection '{collection_name}': {missing_fields}"
            )
            raise MilvusServiceError(
                f"Parquet file is missing required Milvus fields: {missing_fields}"
            )

        # ------------------------------------------------------------------
        # Validate embedding dimensionality with a lightweight read
        # ------------------------------------------------------------------
        try:
            # Identify the vector field in the Milvus schema; the dimension is
            # typically stored under `field.params["dim"]`.
            vector_field_dim: Optional[int] = None
            vector_field_name: Optional[str] = None
            for field in collection_schema.fields:
                params = getattr(field, "params", {})  # type: ignore[attr-defined]
                if "dim" in params:  # Milvus Float/Binary vector field
                    vector_field_dim = int(params["dim"])  # type: ignore[arg-type]
                    vector_field_name = field.name
                    break

            # If we cannot locate a dimension in the schema, skip this check.
            if (
                vector_field_dim
                and vector_field_name
                and vector_field_name in parquet_fields
            ):
                # Read only the first row (1 record) for the embedding column
                with open_s3_import(file_path, storage_options) as f:
                    pf = pq.ParquetFile(f)
                    first_batch = next(
                        pf.iter_batches(columns=[vector_field_name], batch_size=1)
                    )
                    table = pa.Table.from_batches([first_batch])
                # The resulting table should contain exactly one record
                first_embedding = table.column(0)[0]  # type: ignore[index]
                if hasattr(first_embedding, "to_pylist"):
                    first_embedding = first_embedding.to_pylist()

                parquet_emb_dim = len(first_embedding)

                if parquet_emb_dim != vector_field_dim:
                    logger.error(
                        "Embedding dimension mismatch between Parquet (%s) and Milvus collection (%s) for field '%s'",
                        parquet_emb_dim,
                        vector_field_dim,
                        vector_field_name,
                    )
                    raise MilvusServiceError(
                        f"Embedding dimension mismatch: Parquet has {parquet_emb_dim}, Milvus expects {vector_field_dim}."
                    )
        except FileNotFoundError:
            raise  # Propagate
        except Exception as exc:
            logger.error("Error validating embedding dimension: %s", exc, exc_info=True)
            raise

        logger.info(
            f"Schema validation passed for '{file_path}' against '{collection_name}'."
        )

    except MilvusException as e:
        logger.error(
            f"Milvus error retrieving schema for collection '{collection_name}': {e}"
        )
        # Translate Milvus error to custom or HTTP error
        if "does not exist" in str(e):
            raise MilvusServiceError(
                f"Milvus collection '{collection_name}' not found."
            )
        else:
            raise MilvusServiceError(f"Error retrieving collection schema: {e}")
    except Exception as e:
        # Catch unexpected errors during validation
        logger.error(
            f"Unexpected error during schema validation for '{file_path}': {e}",
            exc_info=True,
        )
        # Re-raise or wrap in custom error
        if isinstance(e, MilvusServiceError) or isinstance(e, FileNotFoundError):
            raise  # Propagate specific errors
        else:
            raise MilvusServiceError(
                f"An unexpected error occurred during schema validation: {e}"
            )
