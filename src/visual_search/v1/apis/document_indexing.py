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

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import pandas as pd
import pyarrow as pa
from fastapi import APIRouter, Body, HTTPException

from haystack import Document as HaystackDocument
from haystack import Pipeline as HaystackPipeline
from haystack.dataclasses import ByteStream
from src.haystack.components.milvus.document_store import MilvusDocumentStore
from src.haystack.components.milvus.schema_utils import MetadataConfig
from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedInputError
from src.visual_search.common.remote_fetch import (
    FetchError,
    FetchPolicyError,
    download_text,
)

from ...common.apis.collections import GetCollectionModel, GetPipeline
from ...common.models import (
    Collection,
    DeleteResponse,
    Document,
    DocumentResponse,
    DocumentUploadEmbedding,
    DocumentUploadJson,
    DocumentUploadUrl,
    ExistenceCheckMode,
    MimeType,
    make_path_with_id_restrictions,
    validate_search_filter,
)
from ...common.pipelines import (
    EnabledPipeline,
    get_document_stores,
    get_pipeline_by_collection,
)
from ...logger import logger
from .collections import create_safe_name

router = APIRouter()


@router.post(
    "/collections/{collection_id}/documents",
    responses={
        200: {"description": "Document successfully added to the collection."},
        404: {"description": "Collection does not exist."},
        422: {"description": "Unable to add document, invalid parameters provided."},
    },
    tags=["Document Indexing"],
    summary="Add multiple documents for indexing",
    description="""

        Use this endpoint to ingest files which are then processed by the specified pipeline.
        Three types of models are supported:
        - DocumentUploadJson: this uploads `content`, encoded as base64 string if not plain text (e.g. image)
        - DocumentUploadUrl: this provides a `url`, which the service will try download.
            This is the recommended method for files on s3 buckets, as one can provide a presigned URL.
        - DocumentUploadEmbedding: this provides `embedding`, which is pre-computed on the client side.

        For example, to upload an image:
        ```
        endpoint = f"{retriever_base}/collections/{collection_id}/documents"

        def upload_local(file: str, endpoint: str) -> Dict[str, Any]:
            with open(file, "rb") as file:
                payload = {
                    "content": base64.b64encode(file.read()).decode("utf-8"),
                    "mime-type": "image/jpeg",
                }
            response = requests.post(endpoint, json=[payload])
            return response.json()

        def upload_remote(file: str, endpoint: str) -> Dict[str, Any]:
            payload = {
                "url": file,
                "mime-type": "image/jpeg",
            }
            response = requests.post(endpoint, json=[payload])
            return response.json()

        def upload_embedding(emb: List[float], endpoint: str) -> Dict[str, Any]:
            payload = {
                "embedding": emb,
                "mime-type": "image/jpeg",
            }
            response = requests.post(endpoint, json=[payload])
            return response.json()
        ```
    """,
    response_model=DocumentResponse,
    operation_id="add_documents",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "maxItems": os.getenv("MAX_DOCS_PER_UPLOAD", 100),
                    }
                }
            },
            "required": True,
        },
    },
)
async def index_documents(
    documents: List[
        Union[DocumentUploadJson, DocumentUploadUrl, DocumentUploadEmbedding]
    ],
    collection: Collection = GetCollectionModel,
    pipeline: EnabledPipeline = GetPipeline,
    existence_check_mode: ExistenceCheckMode = ExistenceCheckMode.MUST_CHECK,
) -> DocumentResponse:
    max_docs = int(os.getenv("MAX_DOCS_PER_UPLOAD", 100))
    if len(documents) > max_docs:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Maximum number of documents per request {max_docs}",
        )

    try:
        return _index_documents(
            collection=collection,
            pipeline=pipeline,
            documents=documents,
            existence_check_mode=existence_check_mode,
        )
    except CosmosEmbedInputError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


def _download_url_data(url: str) -> bytes:
    """Fetch text through the same policy for every connection and redirect."""
    try:
        return download_text(url)
    except FetchPolicyError as error:
        raise HTTPException(400, str(error)) from error
    except FetchError as error:
        raise HTTPException(502, "Unable to download text import") from error


def _convert_to_haystack_document(
    document: Union[DocumentUploadJson, DocumentUploadUrl, DocumentUploadEmbedding],
) -> HaystackDocument:
    """Convert different document upload formats into Haystack document."""

    # augment document metadata
    id = str(uuid4()) if document.id is None else document.id
    metadata = document.metadata or {}
    metadata.update(
        source_id=id,
        indexed_at=datetime.utcnow().isoformat(),
        mime_type=document.mime_type.value,
    )

    # initialize additional haystack document attributes
    content: str = ""
    blob: Optional[ByteStream] = None
    embedding: Optional[List[float]] = None

    if isinstance(document, DocumentUploadJson):
        if document.mime_type == MimeType.TEXT:
            content = document.content
        elif document.content.startswith(("data:video/", "data:video_frames/")):
            content = document.content
        else:
            # keep base64 as data URI for downstream embedder
            ext = document.mime_type.value.split("/")[-1]
            content = f"data:video/{ext};base64,{document.content}"
    elif isinstance(document, DocumentUploadEmbedding):
        embedding = document.embedding
        content = None
    elif isinstance(document, DocumentUploadUrl):
        if document.mime_type == MimeType.TEXT:
            try:
                content = _download_url_data(document.url).decode("utf-8")
            except UnicodeDecodeError as error:
                raise HTTPException(
                    400, "Text import must contain UTF-8 text"
                ) from error
        else:
            content = document.url
            metadata["source_url"] = document.url
    else:
        raise TypeError(
            "Document is not `DocumentUploadJson`, `DocumentUploadUrl`, `DocumentUploadEmbedding`"
        )

    # generate haystack document
    return HaystackDocument(
        id=id,
        content=content,
        meta=metadata,
        embedding=embedding,
        blob=blob,
    )


def _index_documents(
    collection: Collection,
    pipeline: EnabledPipeline,
    documents: List[
        Union[DocumentUploadJson, DocumentUploadEmbedding, DocumentUploadUrl]
    ],
    existence_check_mode: ExistenceCheckMode = ExistenceCheckMode.CHECK_WITH_TIMEOUT,
) -> DocumentResponse:
    """Private method for indexing documents."""

    # Get indexing pipeline
    index_pipeline: HaystackPipeline = get_pipeline_by_collection(
        collection
    ).index_pipeline

    # Get index name from collection id
    index_name = create_safe_name(collection.id)

    # Finish URL validation/downloads before replacing existing documents. A
    # rejected URL (including a redirect) must not delete an existing document.
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(documents)))) as executor:
        haystack_docs = list(executor.map(_convert_to_haystack_document, documents))

    # Delete existing documents in Haystack document stores if IDs specified
    # Note: there's a DuplicatePolicy defined in haystack, which is also a param for `write_documents`
    # so ideally we want to use that enum, and move this logic into the document store,
    # one problem is there is no `CHECK_WITH_TIMEOUT` enum value in DuplicatePolicy
    existing_ids = [document.id for document in documents if document.id is not None]
    if existing_ids:
        if existence_check_mode == ExistenceCheckMode.SKIP:
            logger.info(f"Skipped checking existence for {len(existing_ids)} documents")
        else:
            timeout = None
            if existence_check_mode == ExistenceCheckMode.CHECK_WITH_TIMEOUT:
                timeout = 5.0
                logger.info(
                    f"Checking existence for {len(existing_ids)} documents with {timeout}s timeout"
                )
            for document_store in get_document_stores(index_pipeline):
                delete_existing_docs(
                    document_store, index_name, existing_ids, timeout=timeout
                )

    # Track video processing metrics
    video_docs_count = sum(
        1
        for doc in documents
        if isinstance(doc, DocumentUploadUrl)
        and doc.mime_type.value.startswith("video/")
    )

    if video_docs_count > 0:
        logger.info(
            f"Downloading {video_docs_count} video documents for secure base64 encoding"
        )

    logger.info(f"Processing {len(documents)} documents with cosmos-embed pipeline")

    # Index Haystack documents
    resp_docs = _index_haystack_documents(collection, pipeline, haystack_docs)
    resp = DocumentResponse(documents=resp_docs)

    return resp


@router.delete(
    "/collections/{collection_id}/documents/{document_id}",
    responses={
        200: {"description": "Document successfully deleted."},
        204: {"description": "Document successfully deleted, no content returned."},
        404: {"description": "Document or collection does not exist."},
        422: {"description": "Unable to delete document, invalid parameters provided."},
    },
    tags=["Document Indexing"],
    summary="Remove a document from the collection",
    response_model_by_alias=True,
    operation_id="delete_document",
)
def delete_document(
    collection: Collection = GetCollectionModel,
    pipeline: EnabledPipeline = GetPipeline,
    document_id: str = make_path_with_id_restrictions("Identifier for the document."),
) -> DeleteResponse:
    # todo: revalidate that collection.pipeline is still configured for this instance
    #  of the service. if it's not, the collection is useless.
    index_pipeline: HaystackPipeline = get_pipeline_by_collection(
        collection
    ).index_pipeline

    any_retrieved_docs = False
    document_store_found = False
    index_name = create_safe_name(collection.id)
    for document_store in get_document_stores(index_pipeline):
        document_store_found = True

        # First try to find document by source_id (original API-provided ID)
        filters = {"field": "meta.source_id", "operator": "==", "value": document_id}
        if isinstance(document_store, MilvusDocumentStore):
            retrieved_documents = document_store.filter_documents(
                collection_name=index_name,
                filters=filters,
            )
        else:
            retrieved_documents = document_store.filter_documents(filters=filters)

        # If not found by source_id, try to find by actual document ID (Milvus-generated UUID from UI)
        if len(retrieved_documents) == 0:
            if isinstance(document_store, MilvusDocumentStore):
                # For Milvus, try direct deletion by document ID
                try:
                    # Check if document exists by attempting to retrieve it directly
                    filters_by_id = {
                        "field": "id",
                        "operator": "==",
                        "value": document_id,
                    }
                    retrieved_documents = document_store.filter_documents(
                        collection_name=index_name,
                        filters=filters_by_id,
                    )
                except Exception as e:
                    logger.debug(f"Could not find document by ID {document_id}: {e}")
                    retrieved_documents = []
            else:
                # For other document stores, try filtering by document ID
                try:
                    filters_by_id = {
                        "field": "id",
                        "operator": "==",
                        "value": document_id,
                    }
                    retrieved_documents = document_store.filter_documents(
                        filters=filters_by_id
                    )
                except Exception as e:
                    logger.debug(f"Could not find document by ID {document_id}: {e}")
                    retrieved_documents = []

        if len(retrieved_documents) > 0:
            any_retrieved_docs = True
            if isinstance(document_store, MilvusDocumentStore):
                document_store.delete_documents(
                    collection_name=index_name,
                    document_ids=[document.id for document in retrieved_documents],
                )
            else:
                document_store.delete_documents(
                    document_ids=[document.id for document in retrieved_documents]
                )

    if not document_store_found:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail=(f"Document Store Not Found in Pipeline: '{collection.pipeline}'"),
        )

    if not any_retrieved_docs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document ID '{document_id}' Not Found "
                f"in any Document Stores in the Pipeline '{collection.pipeline}' "
                f"(searched both by source_id and document id)"
            ),
        )

    return DeleteResponse(id=document_id, deleted_at=datetime.utcnow())


@router.delete(
    "/collections/{collection_id}/documents",
    responses={
        200: {"description": "Documents successfully deleted."},
        204: {"description": "Documents successfully deleted, no content returned."},
        404: {"description": "Document or collection does not exist."},
        422: {"description": "Unable to delete document, invalid parameters provided."},
    },
    tags=["Document Indexing"],
    summary="Removes list of documents from the collection",
    response_model_by_alias=True,
    operation_id="delete_documents_by_filter",
)
def delete_documents_by_filter(
    collection: Collection = GetCollectionModel,
    pipeline: EnabledPipeline = GetPipeline,
    filters: Dict[str, Any] = Body(
        default={},
        examples=[{"field": "session_id", "operator": "in", "value": ["a", "b"]}],
        description='Extra pre-filters, for example {"field": "session_id", "operator": "in", "value": ["a", "b"]}',  # noqa
    ),
) -> DeleteResponse:
    validate_search_filter(filters)

    index_pipeline: HaystackPipeline = get_pipeline_by_collection(
        collection
    ).index_pipeline

    any_retrieved_docs = False
    document_store_found = False
    index_name = create_safe_name(collection.id)
    for document_store in get_document_stores(index_pipeline):
        document_store_found = True
        if isinstance(document_store, MilvusDocumentStore):
            deleted_documents = document_store.delete_documents_by_filter(
                collection_name=index_name, filters=filters
            )
        else:
            deleted_documents = document_store.delete_documents_by_filter(
                filters=filters
            )

        if deleted_documents > 0:
            any_retrieved_docs = True

    if not document_store_found:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail=(f"Document Store Not Found in Pipeline: '{collection.pipeline}'"),
        )

    if not any_retrieved_docs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No Documents Found with this filter "
                f"in any Document Stores in the Pipeline '{collection.pipeline}'"
            ),
        )

    return DeleteResponse(
        message=f"{deleted_documents} Documents deleted successfully.",
        deleted_at=datetime.utcnow(),
    )


def _index_haystack_documents(
    collection: Collection,
    pipeline: EnabledPipeline,
    haystack_documents: List[HaystackDocument],
) -> List[Document]:
    """Add a document for indexing according to chunking strategy defined for
    collection."""
    # todo: revalidate that collection.pipeline is still configured for this instance
    #  of the service. if it's not, the collection is useless.
    index_pipeline: HaystackPipeline = get_pipeline_by_collection(
        collection
    ).index_pipeline
    index_pipeline_inputs = get_pipeline_by_collection(collection).index_pipeline_inputs

    index_name = create_safe_name(collection.id)

    t1 = time.time()

    from src.visual_search.common import pipelines as _pipelines

    _pipelines.run_index_pipeline(
        index_pipeline=index_pipeline,
        index_pipeline_inputs=index_pipeline_inputs,
        documents=haystack_documents,
        index_name=index_name,
    )

    t2 = time.time()
    logger.debug(f"index_document    run_index_pipeline    {index_name}    {t2-t1}s")

    resp_docs = []
    for haystack_document in haystack_documents:
        mime_type = haystack_document.meta.pop("mime_type", "")
        indexed_at = haystack_document.meta.pop(
            "indexed_at", datetime.utcnow().isoformat()
        )
        resp_docs.append(
            Document(
                id=haystack_document.id,
                content=haystack_document.content or "",
                indexed_at=indexed_at,
                metadata=haystack_document.meta,
                mime_type=mime_type,
            )
        )

    return resp_docs


def flatten_dict(dictionary: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    items: List[Any] = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            if not isinstance(value, str):
                value = str(value)
            items.append((new_key, value))

    return dict(items)


def delete_existing_docs(
    document_store,
    index_name: str,
    candidate_ids: List,
    timeout: Optional[float] = None,
):
    """
    using a more performant method to query existing ids
    """
    if isinstance(document_store, MilvusDocumentStore):
        existing_ids = document_store.get_existing_ids(
            collection_name=index_name,
            ids=candidate_ids,
            timeout=timeout,
        )
        if len(existing_ids) > 0:
            logger.info(f"deleting existing docs {len(existing_ids)}")
            document_store.delete_documents(
                collection_name=index_name, document_ids=existing_ids
            )
        return

    retrieved_documents = document_store.filter_documents(
        {
            "field": "meta.source_id",
            "operator": "in",
            "value": candidate_ids,
        }
    )
    if len(retrieved_documents) > 0:
        document_store.delete_documents(
            document_ids=[document.id for document in retrieved_documents]
        )


def create_ids(df: pd.DataFrame, id_cols: Optional[List[str]]) -> pa.Array:
    """
    Creates a pyarrow array of hashed string IDs based on the specified columns in the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - id_cols (list): Optional list of column names to be used for generating the IDs.

    Returns:
    - pa.Array: A pyarrow array of string IDs.
    """

    if not id_cols:
        ids = (str(uuid4()) for _ in range(len(df)))
    else:

        def generate_id(row: pd.Series) -> str:
            assert id_cols is not None
            id_str = "_".join([str(row[col]) for col in id_cols])
            if len(id_cols) > 1:
                return sha256(id_str.encode("utf-8")).hexdigest()
            return id_str

        ids = df.apply(generate_id, axis=1)
    return pa.array(ids, type=pa.string())


def _generate_meta_json(row: pd.Series) -> str:
    return json.dumps(row.to_dict())


def create_meta(
    df: pd.DataFrame,
    metadata_cols: List[str],
    fillna: bool,
    metadata_config: MetadataConfig,
) -> Dict[str, pa.Array]:
    """
    Creates a pyarrow dict based on the specified metadata columns in the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - metadata_cols (list): A list of column names to be used for generating the metadata.
    - fillna: whether to fill metadata that is NaN.
    - metadata_config: collection metadata configuration.

    Returns:
    - Dict[str, pa.Array]: A dict of pyarrow arrays for metadata.
    """

    expected_fields = {field.name for field in metadata_config.fields}
    for field in expected_fields:
        if field not in metadata_cols:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=f"Collection requires {field} metadata field, have {metadata_cols}",
            )

    dynamic_fields = set(metadata_cols) - expected_fields
    if dynamic_fields and not metadata_config.allow_dynamic_schema:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Collection has dynamic schema disabled. Expected {expected_fields} fields, but got {dynamic_fields} extra.",
        )

    metadata_dict: Dict[str, pa.Array] = {}
    if dynamic_fields:
        dynamic_metadata = df[list(dynamic_fields)]
        if fillna:
            dynamic_metadata = dynamic_metadata.fillna(0.0)
        metadata_dict["$meta"] = pa.array(
            dynamic_metadata.apply(_generate_meta_json, axis=1), type=pa.string()
        )

    for field in metadata_config.fields:
        metadata_dict[field.name] = pa.array(
            df[field.name],
            type=field.to_pyarrow_type(),
        )
    return metadata_dict


def create_embeddings(df: pd.DataFrame, embeddings_col: str) -> pa.Array:
    """
    Creates a pyarrow array of embeddings based on the specified embeddings column in the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - embeddings_col (str): The name of the column containing the embeddings.

    Returns:
    - pa.Array: A pyarrow array of embeddings, where each embedding is a list of floats.
    """
    return pa.array(df[embeddings_col], pa.list_(pa.float32()))


def create_parquet_table(
    df: pd.DataFrame,
    id_cols: Optional[List[str]],
    embeddings_col: str,
    metadata_cols: List[str],
    fillna: bool,
    metadata_config: MetadataConfig,
) -> pa.Table:
    """
    Creates a Parquet table from a dataframe using specified columns for IDs, embeddings, and metadata.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - id_cols (list): Optional list of column names to be used for generating the IDs.
    - embeddings_col (str): The name of the column containing the embeddings.
    - metadata_cols (list): A list of column names to be used for generating the metadata JSON.
    - fillna (bool): whether to fill NaN in metadata (Milvus does not accept NaN).
    - metadata_config: collection metadata configuration.

    Returns:
    - pa.Table: An Apache Arrow table ready for Parquet conversion.
    """

    data = create_meta(df, metadata_cols, fillna, metadata_config)
    data["id"] = create_ids(df, id_cols)
    data["embedding"] = create_embeddings(df, embeddings_col)
    return pa.Table.from_pydict(data)
