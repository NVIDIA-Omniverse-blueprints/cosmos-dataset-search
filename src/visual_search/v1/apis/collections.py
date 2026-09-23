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

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from haystack import Pipeline as HaystackPipeline

from src.haystack.components.milvus.document_store import MilvusDocumentStore
from src.visual_search.common.apis.collections import (
    GetCollection,
    GetPipeline,
    MaybeGetPipeline,
)
from src.visual_search.common.db_models import (
    create_collection as db_create_collection,
)
from src.visual_search.common.db_models import (
    delete_collection as db_delete_collection,
)
from src.visual_search.common.db_models import (
    update_collection as db_update_collection,
)
from src.visual_search.common.db_models import list_collections
from src.visual_search.common.models import (
    Collection,
    CollectionCreate,
    CollectionInfoListResponse,
    CollectionInfoResponse,
    CollectionListResponse,
    CollectionPatch,
    CollectionResponse,
    DeleteResponse,
    make_path_with_id_restrictions,
)
from src.visual_search.v1.apis.utils.milvus_utils import create_safe_name

from ...common.pipelines import (
    EnabledPipeline,
    enabled_pipelines,
    get_all_pipelines,
    get_document_stores,
    get_pipeline_by_collection,
)

router = APIRouter()


def create_new_collection(
    collection: CollectionCreate,
    id: Optional[UUID] = Query(
        None, description="Optional custom ID for the new collection"
    ),
) -> Collection:
    """
    Create a new collection and initialize its Milvus index.

    This helper function handles the core logic of collection creation, including
    ID generation, validation, metadata storage, and Milvus index initialization.

    Args:
        collection: CollectionCreate object containing pipeline, name, tags, and configuration.
        id: Optional UUID to use as the collection ID. If not provided, one will be generated.

    Returns:
        Created Collection object with assigned ID and creation timestamp.

    Raises:
        HTTPException: If a collection with the specified ID already exists (409).
    """
    if id is not None:
        collections = list_collections()
        stored_collection = next((c for c in collections if c.id == str(id)), None)
        if stored_collection is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Specified collection ID {id} already exists",
            )
    stored_collection = Collection(
        pipeline=collection.pipeline,
        name=collection.name,
        tags=collection.tags,
        init_params=collection.init_params,
        cameras=collection.cameras,
        id=id if id is not None else str(uuid4()),
        created_at=datetime.utcnow(),
    )
    db_create_collection(stored_collection)

    # we must create the index for the collection in the document store. it is done
    # dynamically for miluvs but not for elasticsearch. this is an issue when trying
    # to search an empty index.
    #
    # todo: find a more efficient way to do this. writing and deleting a document has
    #  a nice side effect of validating the pipeline works, but it is an expensive way
    #  to force creation of an index. most document stores have a private method to
    #  create the index.
    #
    # todo: revalidate that collection.pipeline is still configured for this instance
    #  of the service. if it's not, the collection is useless.
    # index_pipeline: HaystackPipeline = pipeline.index_pipeline

    index_pipeline = get_pipeline_by_collection(collection).index_pipeline

    index_name = create_safe_name(stored_collection.id)
    for document_store in get_document_stores(index_pipeline):
        if isinstance(document_store, MilvusDocumentStore):
            document_store.create_index(
                index=index_name,
                collection_config=collection.collection_config,
                index_config=collection.index_config,
                metadata_config=collection.metadata_config,
            )

    return stored_collection


@router.post(
    "/collections",
    response_model=CollectionResponse,
    responses={
        200: {"description": "Collection successfully created."},
        400: {"description": "Pipeline does not exist."},
        409: {"description": "Collection ID specified already exists."},
        422: {
            "description": "Unable to create collection, invalid parameters provided."
        },
    },
    tags=["Collections"],
    summary="Create a collection",
    operation_id="create_collection",
)
def create_collection(
    *,
    collection: CollectionCreate,
    id: Optional[UUID] = None,
) -> CollectionResponse:
    """Create a new collection with the given pipeline."""

    stored_collection = create_new_collection(collection, id)

    return CollectionResponse(collection=stored_collection)  # type: ignore


@router.get(
    "/collections",
    response_model=CollectionListResponse,
    responses={
        200: {"description": "Successful collection listing."},
    },
    tags=["Collections"],
    summary="List all collections",
    operation_id="get_collections",
)
def get_collections() -> CollectionListResponse:
    """Retrieve all collections."""

    pipelines = get_all_pipelines()
    allowed_collections = []

    collections = list_collections()
    allowed_collections = [
        collection for collection in collections if collection.pipeline in pipelines
    ]

    return CollectionListResponse(collections=allowed_collections)


@router.get(
    "/pipelines/{pipeline_id}/collections",
    response_model=CollectionInfoListResponse,
    responses={
        200: {"description": "Successful collection info listing."},
    },
    tags=["Collections"],
    summary="List all collection info in a pipeline",
    operation_id="get_pipeline_collections",
)
def get_pipeline_collections(
    pipeline_id: str = make_path_with_id_restrictions("Identifier for the pipeline."),
) -> CollectionInfoListResponse:
    """Retrieve all collections with info"""

    pipeline_collections = []
    collections = list_collections()

    for collection in collections:
        if (
            collection.pipeline == pipeline_id
            and collection.pipeline in enabled_pipelines
        ):
            documents_count = 0
            index_pipeline: HaystackPipeline = get_pipeline_by_collection(
                collection
            ).index_pipeline
            index_name = create_safe_name(collection.id)
            for document_store in get_document_stores(index_pipeline):
                if isinstance(document_store, MilvusDocumentStore):
                    documents_count += document_store.get_embedding_count(
                        collection_name=index_name
                    )
            collection_info = CollectionInfoResponse(
                collection=collection, total_documents_count=documents_count
            )
            pipeline_collections.append(collection_info)

    return CollectionInfoListResponse(collections=pipeline_collections)


@router.get(
    "/collections/{collection_id}",
    response_model=CollectionInfoResponse,
    responses={
        200: {"description": "Successful collection retrieval."},
        404: {"description": "Collection does not exist."},
        422: {
            "description": (
                "Unable to retrieve collection due to validation "
                "error while processing request."
            )
        },
    },
    tags=["Collections"],
    summary="Return collection details",
    operation_id="get_collection",
)
def get_collection_api(
    pipeline: EnabledPipeline = GetPipeline,
    collection: Collection = GetCollection,
) -> CollectionInfoResponse:
    """Retrieve a collection by ID."""

    index_pipeline: HaystackPipeline = get_pipeline_by_collection(
        collection
    ).index_pipeline

    # TODO: what do we do if there are more than one BaseRetriever?
    documents_count = 0
    index_name = create_safe_name(collection.id)
    for document_store in get_document_stores(index_pipeline):
        if isinstance(document_store, MilvusDocumentStore):
            documents_count += document_store.get_embedding_count(
                collection_name=index_name
            )

    return CollectionInfoResponse(
        collection=collection, total_documents_count=documents_count
    )


@router.patch(
    "/collections/{collection_id}",
    responses={
        200: {"description": "Successful update of collection."},
        404: {"description": "Collection does not exist."},
        422: {
            "description": """
                Unable to update collection due to validation error while processing request.
            """
        },
    },
    response_model=CollectionResponse,
    summary="Update a collection",
    description="""
        Only the `name` and `tags` attributes are mutable. Once a collection is created,
        the `pipeline` cannot be changed.
    """,
    tags=["Collections"],
    operation_id="update_collection",
)
def update_collection(
    *,
    collection: Collection = GetCollection,
    patch: CollectionPatch,
) -> CollectionResponse:
    """
    Update a collection's mutable attributes.

    Only the 'name' and 'tags' attributes can be updated after collection creation.
    The pipeline association is immutable.

    Args:
        collection: The existing collection object (injected by GetCollection dependency).
        patch: CollectionPatch object containing the fields to update.

    Returns:
        CollectionResponse containing the updated collection.
    """
    updates = {}
    if patch.name:
        updates["name"] = patch.name
    if patch.tags:
        updates["tags"] = patch.tags
    
    # Persist changes to database
    updated_collection = db_update_collection(collection.id, updates)
    return CollectionResponse(collection=updated_collection)


@router.delete(
    "/collections/{collection_id}",
    responses={
        200: {"description": "Successful collection deletion."},
        404: {"description": "Collection does not exist."},
        422: {
            "description": (
                "Unable to delete collection due to validation "
                "error while processing request."
            )
        },
    },
    tags=["Collections"],
    operation_id="delete_collection",
)
def delete_collection(
    *,
    collection: Collection = GetCollection,
    pipeline: EnabledPipeline | None = MaybeGetPipeline,
) -> DeleteResponse:
    """Delete a collection by ID."""
    try:
        db_delete_collection(collection.id)
        return DeleteResponse(
            message=f"Collection {collection.id} deleted successfully.",
            id=str(collection.id),
            deleted_at=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection {collection.id}: {str(e)}",
        )
