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

import base64
import os
import re
from builtins import anext
from typing import Dict, Final, List, Optional

import boto3
import kubernetes
import requests
from botocore.client import BaseClient
from botocore.config import Config
from fastapi import APIRouter, Body, Depends, HTTPException
from haystack import Document as HaystackDocument
from haystack import Pipeline
from kubernetes.client.rest import ApiException
from pydantic import BaseModel

from src.visual_search.common.exceptions import SecretsNotFoundError
from src.visual_search.v1.apis.nvcf_file_based_secrets_manager import (
    NVCFFileBasedSecretsManager,
)

from ...common.apis.collections import GetCollectionModel, GetPipeline, pipeline_handler
from ...common.db_models import list_collections
from ...common.models import (
    Collection,
    MimeType,
    RadiusSearch,
    RetrievalQuery,
    RetrievedDocument,
    SearchRequest,
    SearchResponse,
)
from ...common.pipelines import EnabledPipeline, run_query_pipeline
from ...logger import logger
from .document_indexing import create_safe_name
from .k8s_secrets import get_k8s_secret

router = APIRouter()

PRESIGNED_TIMEOUT: Final[int] = 3600


def create_s3_client(
    config: Config, credentials: Optional[Dict[str, str]]
) -> BaseClient:
    """Get s3 client based on environment variables."""
    if credentials is not None:
        kwargs = {
            "aws_access_key_id": credentials.get("aws_access_key_id", ""),
            "aws_secret_access_key": credentials.get("aws_secret_access_key", ""),
            "region_name": credentials.get("aws_region"),
            "config": config,
        }

        endpoint = credentials.get("endpoint_url")
        if endpoint:  # add only when truthy
            kwargs["endpoint_url"] = endpoint

        return boto3.client("s3", **kwargs)
    return boto3.client("s3", config=config)


def _create_cdn_asset(collection, doc, media_cdn_route):
    presigned_url = None
    # storage-template: substitution and presigning required
    storage_template = collection.tags.get("storage-template", "none")
    match = re.search(r"\{\{(\w+)\}\}", storage_template)
    if match:
        key = match.group(1)
        url = storage_template.replace(f"{{{{{key}}}}}", doc.meta[key])
        # storage-secrets: cdn required

        storage_secrets = collection.tags.get("storage-secrets")
        if storage_secrets:
            offset = int(doc.meta.get("byte_offset", 0))
            size = int(doc.meta.get("byte_size", 0))
            request_json = {
                "s3_url": url,
                "storage_secrets": storage_secrets,
                "pipeline": collection.pipeline,
            }
            if size:
                request_json["byte_range"] = [offset, offset + size]
            cdn_route = f"{media_cdn_route}/create_asset"
            response = requests.post(cdn_route, json=request_json)
            presigned_url = response.json()
    return presigned_url


class PresignedUrl(BaseModel):
    presigned_url: str


def bucket_and_key(s3_url: str) -> Dict[str, str]:
    """Extract bucket and key from s3 url."""
    if not s3_url.startswith("s3://"):
        raise HTTPException(
            status_code=400, detail="The s3_url must start with 's3://'"
        )
    bucket, key = s3_url[5:].split("/", 1)
    return {"Bucket": bucket, "Key": key}


async def get_custom_aws_credentials(storage_secrets: str) -> Dict[str, str]:
    """Get decrypted secrets from either NVCF or Kubernetes based on the configuration and secret name."""

    use_nvcf_assets_creds = os.environ.get("NGC_SECRETS_FILE_PATH")

    if use_nvcf_assets_creds:
        # Use the NVCFFileBasedSecretsManager
        nvcf_manager = NVCFFileBasedSecretsManager()
        secret_data = nvcf_manager.get_secrets()
        if secret_data is None:
            raise SecretsNotFoundError("Secrets not found!")
        return secret_data
    else:
        # Use the new module to get the secret
        try:
            secret_data = get_k8s_secret(storage_secrets, namespace="default")
            return secret_data
        except Exception as e:
            # Raise a more informative exception with additional debugging data
            raise type(e)(
                f"Failed to retrieve Kubernetes secret '{storage_secrets}' in namespace 'default'. "
                f"Error: {str(e)}"
            )


async def get_presigned_url(
    s3_url: str, credentials: Dict[str, str] = None
) -> PresignedUrl:
    """Get presigned URL for s3 URL."""

    params = bucket_and_key(s3_url)
    if credentials is None:
        aws_region_name = os.getenv("AWS_REGION", None)
    else:
        aws_region_name = credentials["aws_region"]

    config_to_fix_url_403 = Config(
        region_name=aws_region_name,
        signature_version="s3v4",
        retries={
            "max_attempts": 10,
            # 'mode': 'standard'
        },
        s3={"addressing_style": "path"},
    )
    s3_client = create_s3_client(config_to_fix_url_403, credentials)
    try:
        presigned_url = s3_client.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=PRESIGNED_TIMEOUT
        )
        return PresignedUrl(presigned_url=presigned_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/collections/{collection_id}/search",
    responses={
        200: {
            "description": "Search results across specified collection.",
        },
        400: {
            "description": (
                "Invalid input "
                "(e.g., missing query, collections, or invalid top_k value)."
            )
        },
        422: {"description": "Invalid search parameters."},
        404: {"description": "Collection does not exist."},
    },
    tags=["Retrieval"],
    summary="Search within a collection",
    operation_id="search_collection",
)
async def search(
    collection: Collection = GetCollectionModel,
    pipeline: EnabledPipeline = GetPipeline,
    search_post_request: SearchRequest = Body(None, description=""),
) -> SearchResponse:
    """Search for documents in the specified collection."""
    # todo: revalidate that collection.pipeline is still configured for this instance
    #  of the service. if it's not, the collection is useless.
    query_pipeline: Pipeline = pipeline.query_pipeline
    query_pipeline_inputs = pipeline.query_pipeline_inputs

    index_name = create_safe_name(collection.id)
    retrieved_documents: List[HaystackDocument] = run_query_pipeline(
        query_pipeline,
        query_pipeline_inputs,
        query=search_post_request.query,
        index_name=index_name,
        search_params=search_post_request.search_params,
        filters=search_post_request.filters,
        top_k=search_post_request.top_k,
        reconstruct=search_post_request.reconstruct,
        clf=search_post_request.clf,
    )

    media_cdn_route = os.getenv("VIUS_CDN_ENDPOINT", None)
    retrievals = []
    storage_secrets = (
        collection.tags.get("storage-secrets", None) if collection.tags else None
    )
    storage_template = (
        collection.tags.get("storage-template", None) if collection.tags else None
    )
    credentials = None
    if storage_secrets is not None:
        credentials = (
            await get_custom_aws_credentials(
                storage_secrets,
            )
            if storage_secrets
            else None
        )
    for doc in retrieved_documents[: search_post_request.top_k]:
        presigned_url = None
        if search_post_request.generate_asset_url and collection.tags:
            if media_cdn_route:
                try:
                    presigned_url = _create_cdn_asset(collection, doc, media_cdn_route)
                except Exception as e:
                    logger.error("No asset available for this document.")
                    logger.error(e)
            else:
                try:
                    # Waabi force presigned_url retrieval
                    # Subsitute and presign
                    storage_template = collection.tags.get("storage-template", "none")
                    match = re.search(r"\{\{(\w+)\}\}", storage_template)
                    if match:
                        key = match.group(1)
                        url = storage_template.replace(f"{{{{{key}}}}}", doc.meta[key])
                        presigned_url_obj = await get_presigned_url(url, credentials)
                        presigned_url = presigned_url_obj.presigned_url
                except Exception as e:
                    logger.error("No asset available for this document.")
                    logger.error(e)
                    presigned_url = None
        mime_type = doc.meta.pop("mime_type", MimeType.TEXT.value)
        retrievals.append(
            RetrievedDocument(
                id=doc.id,
                collection_id=collection.id,
                asset_url=presigned_url,
                content=doc.content or "",
                score=doc.score,
                metadata=doc.meta,
                mime_type=mime_type,
                embedding=doc.embedding,
            )
        )
    logger.info(f"Returning {len(retrievals)} documents.")
    return SearchResponse(retrievals=retrievals)


@router.post(
    "/retrieval",
    responses={
        200: {
            "description": "Search results across multiple collections.",
        },
        400: {
            "description": (
                "Invalid input "
                "(e.g., missing query, collections, invalid params, or payload_keys value)."
            )
        },
        422: {"description": "Invalid search parameters."},
        404: {"description": "Invalid collection specified"},
    },
    tags=["Retrieval"],
    summary="Retrieve results from multiple collections",
    operation_id="retrieval",
)
async def retrieval(
    retrieval_query: RetrievalQuery = Body(...),
):
    # Radius search not supported
    if isinstance(retrieval_query.params, RadiusSearch):
        raise HTTPException(
            status_code=422,
            detail="Radius search is currently not implemented.",
        )
    # Gather the collections that will be used by the pipelines
    # In the search function
    all_retrievals = []
    lookup = list_collections()
    collections = [c for c in lookup if c.id in retrieval_query.collections]
    search_request = SearchRequest(
        query=retrieval_query.query,
        top_k=retrieval_query.params.nb_neighbors,
        filters=retrieval_query.params.filters,
        search_params=retrieval_query.params.search_params,
        generate_asset_url=retrieval_query.generate_asset_url,
        reconstruct=retrieval_query.params.reconstruct,
    )
    # Gather the pipelines
    pipelines = [pipeline_handler(collection) for collection in collections]
    # Query the search endpoint
    for collection, pipeline in zip(collections, pipelines):
        search_response = await search(collection, pipeline, search_request)
        all_retrievals.extend(search_response.retrievals)

    # Rerank if pipelines all the same type or error
    if retrieval_query.rerank and len(all_retrievals) > 1:
        if len(set(pipeline.id for pipeline in pipelines)) == 1:
            all_retrievals.sort(reverse=True)
        else:
            raise HTTPException(
                status_code=400,
                detail="Re-ranking is not supported for mixed pipelines.",
            )

    # Handle payload keys
    if retrieval_query.payload_keys is None:
        # Return all metadata keys
        pass
    elif retrieval_query.payload_keys == []:
        # Return no metadata keys
        for retrieval in all_retrievals:
            retrieval.metadata = {}
    else:
        # Validate that all keys are valid, and delete all others

        # Delete all the metadata keys except those specified by payload_keys
        # This approach really doesn't scale - if we're returning 10,000 rows
        # then there'll be substantial time just checking on payload keys,
        # possibly more time than is spent transmitting them.
        # This is the kind of logic that should be pushed down to the database
        # layer, where it can be done in a single operation, or be done using
        # a cudf-backed database for much more performance. Right now the
        # results are a dictionary returned by the pipeline, so improving
        # performance is a more serious refactor.

        for retrieval in all_retrievals:
            new_meta = {}
            for key in (
                retrieval_query.payload_keys if retrieval_query.payload_keys else []
            ):
                if key not in retrieval.metadata:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Key {key} not found in metadata.",
                    )
                new_meta[key] = retrieval.metadata[key]
            retrieval.metadata = new_meta

    return SearchResponse(retrievals=all_retrievals)
