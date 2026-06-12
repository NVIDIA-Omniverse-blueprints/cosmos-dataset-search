# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Endpoint to train a collection of search refinement models."""


from typing import List, Tuple, Union

from fastapi import APIRouter, Body, Depends, HTTPException

from src.haystack.components.milvus.document_store import MilvusDocumentStore
from src.models.linear_classifier.model import run_linear_classifier_training

from ...common.apis.collections import pipeline_handler
from ...common.db_models import get_collections, list_collections
from ...common.models import (
    EmbeddingQuery,
    LinearClassifierResponse,
    LinearProbeResponse,
    SearchRefinementMode,
    SearchRefinementRequest,
)
from ...common.pipelines import get_document_stores, run_linear_probe_pipeline
from ...logger import logger
from .document_indexing import create_safe_name
router = APIRouter()


@router.post(
    "/search_refinement/train",
    responses={
        200: {"description": "Train a model."},
        400: {"description": "Invalid input (e.g., missing values)."},
        422: {"description": "Invalid parameters."},
        404: {"description": "Collection does not exist."},
        501: {"description": "Model_type not implemented."},
    },
    tags=["SearchRefinement"],
    summary="""Train a model.""",
    operation_id="train",
)
async def train(
    search_refinement_request: SearchRefinementRequest = Body(
        None,
        description="",
        example={
            "model_type": "linear_probe",
            "grounding_queries": [
                {
                    "text": "picture of a cat",
                }
            ],
            "labels": [
                {
                    "collection_name": "d51c9157-e6c5-46cf-9b29-1dd7a9a1febe",
                    "labelled_documents": {
                        "f0bdff82-4b90-4776-82dd-54f130861dfc": True,
                        "0f2334a1-2a08-4e77-9c14-2e906dce6e4c": False,
                    },
                }
            ],
            "regularization_strength": 0.05,
        },
    ),
) -> Union[LinearProbeResponse, LinearClassifierResponse]:
    """
    The search refinement endpoint learns an optimal query.

    The inputs are:
    1. grounding queries, which will be embedded by the appropriate pipeline (e.g. by some text or video embedders).
    2. lists of document IDs in various collections with binary labels, defining good (`True`) or bad (`False`) retrievals examples.

    if model_type == SearchRefinementMode.LINEAR_PROBE:
        A linear regression model with weights of same dimension as the embeddings will be initialized.
        The embedded queries will be used as regularization for the model weights, while the labels
        will be used to calculate the binary classification loss.
        The optimized linear regression model's weights are equivalent to an optimized retrieval query.

    elif model_type == SearchRefinementMode.LINEAR_CLASSIFIER:
        A linear regression model with weights of same dimension, along with a bias/intercept term will be initialized.
        The labelled embeddings will be used to train a model with the regular L2 regularization.
        This model can be used within the `filtering_models` haystack component.
    """

    # Gather the collections that will be used by the pipeline
    collection_ids = list(
        set(
            labelled_docs.collection_name
            for labelled_docs in search_refinement_request.labels
        )
    )
    collections = get_collections(collection_ids)
    if not isinstance(collections, list):
        collections = [collections]
    logger.warning(
        f"Requested collections {collection_ids} for search refinement: {collections}"
    )

    if len(collections) != len(collection_ids):
        missing = collection_ids - set(collection.id for collection in collections)
        raise HTTPException(
            status_code=404,
            detail=(f"Requested collections {missing} not found!"),
        )

    # Check that collections have the same pipeline type
    pipeline_types = set(collection.pipeline for collection in collections)
    if len(pipeline_types) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Requested collections {collection_ids} come from mixed pipelines {pipeline_types}. "
                "This is not supported."
            ),
        )

    # Fetch pipeline (can use any collection as they are all the same)
    pipeline = pipeline_handler(collections[0])
    document_stores = get_document_stores(pipeline.query_pipeline)
    if len(document_stores) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected a single document store in requested pipeline {pipeline_types}. "
                f"Got {document_stores}. "
            ),
        )
    document_store = document_stores[0]

    # Extract embeddings from labelled documents
    labelled_embeddings: List[Tuple[List[float], bool]] = []
    index_name = ""
    for docs in search_refinement_request.labels:
        index_name = create_safe_name(docs.collection_name)

        filters = {
            "field": "meta.id",
            "operator": "in",
            "value": list(docs.labelled_documents.keys()),
        }
        if isinstance(document_store, MilvusDocumentStore):
            retrieved_documents = document_store.filter_documents(
                collection_name=index_name,
                filters=filters,
                return_embedding=True,
            )
        else:
            retrieved_documents = document_store.filter_documents(
                filters=filters, return_embedding=True
            )

        if len(retrieved_documents) != len(docs.labelled_documents):
            missing_docs = docs.labelled_documents.keys() - set(
                doc.id for doc in retrieved_documents
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Requested documents {missing_docs} do not exists in collection {docs.collection_name}"
                ),
            )

        for doc in retrieved_documents:
            labelled_embeddings.append((doc.embedding, docs.labelled_documents[doc.id]))

    logger.debug(f"Extracted {len(labelled_embeddings)} embeddings")

    # Pipe data through linear probe pipeline
    if search_refinement_request.model_type == SearchRefinementMode.LINEAR_PROBE:
        learnt_queries = run_linear_probe_pipeline(
            query_pipeline=pipeline.query_pipeline,
            query_pipeline_inputs=pipeline.query_pipeline_inputs,
            query=search_refinement_request.grounding_queries,
            subgraph_output_names=["query_learner"],
            labelled_embeddings=labelled_embeddings,
            regularization_strength=search_refinement_request.regularization_strength,
        )

        response = LinearProbeResponse(
            queries=[EmbeddingQuery(embedding=tuple(emb)) for emb in learnt_queries]
        )

    elif search_refinement_request.model_type == SearchRefinementMode.LINEAR_CLASSIFIER:
        _, weights = run_linear_classifier_training(
            labelled_embeddings=labelled_embeddings
        )

        response = LinearClassifierResponse(weights=weights)

    else:
        raise HTTPException(
            status_code=501,
            detail=(
                f'Requested model_type "{search_refinement_request.model_type}" is not implemented, choose from: {[model_type.value for model_type in SearchRefinementMode]}'
            ),
        )

    return response
