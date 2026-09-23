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
Database helper functions for the Visual‑Search service.

Collections are **not** stored in Postgres.  They live in a single Milvus
collection called `_collection_registry`, shared by *all* Haystack pipelines.

Design
------
* At runtime there may be many Haystack pipelines, each with its own
  `MilvusDocumentStore` instance (one per embedding collection).
* Exactly **one** of those stores must also contain `_collection_registry`.
* We locate that store lazily:

    1.  Remember the store attached to the *first* pipeline in
        `enabled_pipelines` (fallback choice).
    2.  Iterate over **all** pipelines → stores.
    3.  If we find one whose `client` already lists `_collection_registry`,
        we adopt it and stop.
    4.  Otherwise we create the registry inside the first store
        (via `fetch_collection_meta()`, which is idempotent and internally
        guarantees the collection exists), then return that store.

* This method avoids race‑conditions across Gunicorn workers / K8s pods:
    Milvus handles concurrent `create_collection` calls for the same name
    safely, returning OK for the first and “already exists” for the rest.

The resulting store is then used by all CRUD helpers below.
"""

from __future__ import annotations

import logging
from datetime import datetime
from http import HTTPStatus
from typing import Dict, List
from uuid import uuid4 as uuid_factory

from fastapi import HTTPException

from src.haystack.components.milvus.document_store import (
    COLL_META,
    MilvusDocumentStore,
)
from src.visual_search.common.models import Collection
from src.visual_search.common.pipelines import enabled_pipelines, get_document_stores
from src.visual_search.v1.apis.utils.milvus_utils import create_safe_name

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Locate / lazily create the universal registry store                         #
# --------------------------------------------------------------------------- #
def _global_store() -> MilvusDocumentStore:
    """
    Return the `MilvusDocumentStore` that owns the `_collection_registry`
    metadata collection, creating it in the first pipeline’s store if needed.

    This function is safe to call from any worker process: Milvus itself
    guarantees idempotent collection creation, so simultaneous starts will
    converge without locks.
    """
    if not enabled_pipelines:
        raise RuntimeError("No pipelines have been initialised")

    # 1. Fallback store = first pipeline’s first store
    first_pipeline = next(iter(enabled_pipelines.values()))
    if not hasattr(first_pipeline, "index_pipeline"):
        raise RuntimeError("First pipeline has no index pipeline - cannot locate store")
    first_stores = get_document_stores(first_pipeline.index_pipeline)
    if not first_stores:
        raise RuntimeError("First pipeline has no document store attached")
    fallback_store: MilvusDocumentStore = first_stores[0]

    # 2. Search all stores for the registry collection
    for pipe in enabled_pipelines.values():
        try:
            for store in get_document_stores(pipe.index_pipeline):
                if COLL_META in store.client.list_collections():
                    return store  # found the registry owner
        except Exception as e:  # pragma: no cover – defensive, e.g. connection hiccup
            raise RuntimeError(
                "Fatal Error: No MilvusDocumentStore found with collection registry",
                e,
            )

    return fallback_store


# --------------------------------------------------------------------------- #
# CRUD helpers – all delegate to the located global store                     #
# --------------------------------------------------------------------------- #
def get_collections(collection_ids: Union[List[str], str]) -> List[Collection]:
    """
    Retrieve a collection by its ID from the metadata store.

    Args:
        collection_id: The unique identifier of the collection.

    Returns:
        Collection object containing the collection metadata.

    Raises:
        HTTPException: If the collection is not found (404).
    """
    store = _global_store()
    if isinstance(collection_ids, str):
        collection_ids = [collection_ids]
    collection = store.fetch_collection_meta(collection_ids)
    if len(collection) == 0:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Collection not found"
        )
    return Collection(**collection[0])


def create_collection(collection: Collection) -> Collection:
    """
    Create a new collection in the metadata store.

    Args:
        collection: Collection object to create. If no ID is provided, one will be generated.

    Returns:
        The created Collection object with assigned ID and creation timestamp.

    Raises:
        HTTPException: If a collection with the same ID already exists (409).
    """
    if not collection.id:
        collection.id = str(uuid_factory())
    collection.id = create_safe_name(collection.id)
    if not collection.created_at:
        collection.created_at = datetime.utcnow()

    store = _global_store()
    try:
        store.register_collection(collection.id, collection.model_dump(mode="json"))
    except ValueError:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT, detail="Collection ID already exists"
        )
    return collection


def update_collection(collection_id: str, updates: Dict) -> Collection:
    """
    Update an existing collection's metadata.

    Args:
        collection_id: The unique identifier of the collection to update.
        updates: Dictionary of field updates to apply to the collection.

    Returns:
        The updated Collection object.

    Raises:
        HTTPException: If the collection is not found (404).
    """
    store = _global_store()
    try:
        store.update_collection_meta(collection_id, updates)
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Collection not found"
        )
    collections = store.fetch_collection_meta([collection_id])
    if not collections:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Collection not found"
        )
    return Collection(**collections[0])


def list_collections() -> List[Collection]:
    """
    List all collections in the metadata store.

    Returns:
        List of Collection objects, excluding the internal metadata registry collection.
    """
    store = _global_store()
    collections = store.client.list_collections()
    collections.remove(COLL_META)
    if len(collections) == 0:
        return []
    collections_meta = store.fetch_collection_meta(collections)
    result = []
    for collection in collections_meta:
        logger.warning(f"Found collection {collection} in {collections_meta}")
        if collection is not None:
            result.append(Collection(**collection))
    return result


def delete_collection(collection: str) -> None:
    """
    Delete a collection from the metadata store and its associated Milvus collection.

    Args:
        collection: The collection ID to delete.

    Raises:
        HTTPException: If the collection is not found (404) or if deletion fails (500).
    """
    store = _global_store()
    try:
        store.delete_collection(collection)
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Collection not found"
        )
    except Exception as e:  # pragma: no cover – defensive, e.g. Milvus hiccup
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}",
        )
