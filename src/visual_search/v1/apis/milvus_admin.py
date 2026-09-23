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
"""Admin endpoints for Milvus-specific operations (e.g., manual flush).

These endpoints are intended for operational or testing use-cases where callers
need deterministic durability / query-visibility guarantees that exceed Milvus'
background flush scheduler.
"""
from __future__ import annotations

from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException
from haystack import Pipeline as HaystackPipeline  # type: ignore
from pymilvus import Collection
from pydantic import BaseModel

from src.visual_search.common.apis.collections import GetCollection, GetPipeline
from src.visual_search.common.pipelines import get_document_stores, EnabledPipeline
from src.visual_search.v1.apis.utils.milvus_utils import create_safe_name
from src.haystack.components.milvus.document_store import MilvusDocumentStore
from src.visual_search.common.models import Collection as CollectionModel

logger = logging.getLogger(__name__)

router = APIRouter()


class FlushResponse(BaseModel):
    """Response returned after a successful manual flush."""

    id: str
    flushed_at: datetime
    message: str = "Collection flushed successfully."


@router.post(
    "/admin/collections/{collection_id}/flush",
    response_model=FlushResponse,
    tags=["Admin", "Collections"],
    summary="Flush a Milvus collection to make recent inserts durable and searchable immediately.",
    operation_id="flush_collection",
)
def flush_collection(
    *,
    collection: CollectionModel = GetCollection,  # Collection metadata from database
    pipeline: EnabledPipeline = GetPipeline,  # EnabledPipeline for this collection
) -> FlushResponse:
    """Force-flush the Milvus collection backing *collection_id*.

    The call blocks until Milvus persists the sealed segment to its storage
    backend (local disk or object storage) and reloads the collection to make
    the new segment searchable.
    """

    index_pipeline: HaystackPipeline = pipeline.index_pipeline
    index_name = create_safe_name(collection.id)

    flushed_one = False
    for document_store in get_document_stores(index_pipeline):
        if not isinstance(document_store, MilvusDocumentStore):
            continue

        try:
            milvus_collection = Collection(name=index_name, using=document_store.client._using)
            milvus_collection.flush()
            logger.debug("Flush finished for collection %s; num_entities=%s", index_name, milvus_collection.num_entities)
            try:
                milvus_collection.load()
            except Exception as load_exc:
                logger.warning("Failed to load collection %s after flush: %s", index_name, load_exc)
            flushed_one = True
        except Exception as exc:
            logger.error("Failed to flush collection %s: %s", index_name, exc)
            raise HTTPException(status_code=500, detail=f"Flush failed: {exc}") from exc

    if not flushed_one:
        # This should only happen if the pipeline uses a non-Milvus document store.
        raise HTTPException(
            status_code=404,
            detail="No Milvus document store found for this collection.",
        )

    return FlushResponse(id=collection.id, flushed_at=datetime.utcnow())