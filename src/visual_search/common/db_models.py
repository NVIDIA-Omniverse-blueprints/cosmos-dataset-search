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

Collections are stored in VastDB ``_collection_registry`` (shared by all
Haystack pipelines). There is no Postgres or VastDB dependency.
"""

from __future__ import annotations

import logging
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Union
from uuid import uuid4 as uuid_factory

from fastapi import HTTPException

from src.haystack.components.vastdb.document_store import COLL_META, VastDocumentStore
from src.haystack.components.vector_store_utils import (
    is_vector_document_store,
    list_store_collections,
)
from src.visual_search.common.models import Collection
from src.visual_search.common.naming import create_safe_name
from src.visual_search.common.pipelines import enabled_pipelines, get_document_stores

logger = logging.getLogger(__name__)


def _global_store() -> VastDocumentStore:
    """Return the VastDocumentStore that owns ``_collection_registry``."""
    if not enabled_pipelines:
        raise RuntimeError("No pipelines have been initialised")

    first_pipeline = next(iter(enabled_pipelines.values()))
    if not hasattr(first_pipeline, "index_pipeline"):
        raise RuntimeError("First pipeline has no index pipeline - cannot locate store")
    first_stores = get_document_stores(first_pipeline.index_pipeline)
    if not first_stores:
        raise RuntimeError("First pipeline has no document store attached")
    fallback_store = first_stores[0]
    if not isinstance(fallback_store, VastDocumentStore):
        raise RuntimeError(
            f"Expected VastDocumentStore, got {type(fallback_store).__name__}"
        )

    for pipe in enabled_pipelines.values():
        try:
            for store in get_document_stores(pipe.index_pipeline):
                if not is_vector_document_store(store):
                    continue
                names = list_store_collections(store)
                if COLL_META in names:
                    return store  # type: ignore[return-value]
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Fatal Error: No VastDocumentStore found with collection registry",
                e,
            ) from e

    return fallback_store


def get_collections(collection_ids: Union[List[str], str]) -> Collection:
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
    store = _global_store()
    collections = list_store_collections(store)
    if COLL_META in collections:
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
    store = _global_store()
    try:
        store.delete_collection(collection)
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Collection not found"
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}",
        )
