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

# ============================================================================ #
#  src/visual_search/v1/apis/bulk_indexing.py                                  #
#  – FastAPI endpoints for Milvus bulk-insert + job monitoring                 #
# ============================================================================ #

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pymilvus import BulkInsertState, MilvusException

from src.visual_search.common.db_models import get_collections
from src.visual_search.common.models import (
    InsertDataRequest,
    InsertDataResponse,
    JobDetail,
    JobStatusResponse,
)
from src.visual_search.common.pipelines import (
    EnabledPipeline,
    enabled_pipelines,
    get_document_stores,
)
from src.visual_search.common.remote_fetch import (
    FetchPolicyError,
    validate_s3_endpoint,
    validate_s3_paths,
)
from src.visual_search.logger import logger
from src.visual_search.v1.apis.utils.milvus_utils import (
    MilvusServiceError,
    build_storage_options,
    create_safe_name,
    validate_parquet_schema,
)

router = APIRouter()

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


def _resolve_pipeline(collection_id: str) -> EnabledPipeline:
    stored = get_collections(collection_id)
    if not stored:
        raise HTTPException(404, f"Collection {collection_id} does not exist")
    pipeline_key = stored.pipeline

    try:
        return enabled_pipelines[pipeline_key]
    except KeyError:
        raise HTTPException(
            400,
            f"Disabled/Unknown pipeline {pipeline_key}\n"
            f"Enabled pipelines: {list(enabled_pipelines)}",
        )


def _safe_is_milvus(ds: Any) -> bool:
    """
    `isinstance(ds, MilvusDocumentStore)` but tolerant of the
    test-suite patch that turns `MilvusDocumentStore` into a MagicMock.
    """
    logger.warning("_safe_is_milvus")
    try:
        from src.haystack.components.milvus.document_store import MilvusDocumentStore

        if isinstance(MilvusDocumentStore, type) and isinstance(
            ds, MilvusDocumentStore
        ):
            return True
    except Exception:
        pass

    # Duck-type check for test doubles
    return all(
        hasattr(ds, attr)
        for attr in (
            "bulk_insert_files",
            "get_bulk_insert_state",
            "list_bulk_insert_tasks",
        )
    )


def _get_milvus_store(pipeline: EnabledPipeline):
    for ds in get_document_stores(pipeline.index_pipeline):
        if _safe_is_milvus(ds):
            return ds
    stores = get_document_stores(pipeline.index_pipeline)
    return stores[0] if stores else None


# --------------------------------------------------------------------------- #
#  POST /insert-data                                                          #
# --------------------------------------------------------------------------- #


@router.post("/insert-data", response_model=InsertDataResponse, status_code=202)
async def insert_data(request: InsertDataRequest):
    """
    Initiate bulk insert of Parquet files into a Milvus collection.

    This endpoint validates the Parquet schema against the collection schema and
    starts asynchronous bulk insert jobs for the provided files.

    Args:
        request: InsertDataRequest containing collection name, parquet file paths,
                and optional authentication credentials.

    Returns:
        InsertDataResponse with status "success" and the job ID for tracking.

    Raises:
        HTTPException:
            - 404 if collection doesn't exist or files not found
            - 400 if schema validation fails or pipeline is disabled
            - 500 if Milvus fails to return a job ID
    """
    # Validate every path and the endpoint before schema reads or job creation.
    # fsspec otherwise accepts local files, HTTP and chained protocols here.
    try:
        keys = validate_s3_paths(request.parquet_paths)
        validate_s3_endpoint(request.endpoint_url)
    except FetchPolicyError as error:
        raise HTTPException(400, str(error)) from error

    pipeline = _resolve_pipeline(request.collection_name)
    store = _get_milvus_store(pipeline)

    safe_coll = create_safe_name(request.collection_name)
    opts = build_storage_options(
        request.access_key, request.secret_key, request.endpoint_url
    )

    # Schema validation
    for path in request.parquet_paths:
        try:
            await validate_parquet_schema(path, safe_coll, opts, store.client._using)
        except MilvusServiceError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError:
            raise HTTPException(404, f"File not found: {path}")

    job_ids = []
    for key in keys:
        job_id = store.bulk_insert_files(
            collection_name=safe_coll,
            file_paths=[key],
            **opts,
        )
        if not job_id:
            raise HTTPException(500, "Milvus did not return a job ID")
        job_ids.append(job_id)
    if not job_ids:
        raise HTTPException(500, "Milvus did not return a job ID")

    return InsertDataResponse(
        status="success", message="Data insertion started", job_id=str(job_ids[0])
    )


# --------------------------------------------------------------------------- #
#  GET /job-status/{job_id}                                                   #
# --------------------------------------------------------------------------- #


# shared mapping (move to a constants.py if you like)
BULK_STATE_MAP = {
    BulkInsertState.ImportPending: "pending",
    BulkInsertState.ImportStarted: "in_progress",
    BulkInsertState.ImportPersisted: "persisted",
    BulkInsertState.ImportCompleted: "completed",
    BulkInsertState.ImportFailed: "failed",
    BulkInsertState.ImportFailedAndCleaned: "failed_cleaned",
    BulkInsertState.ImportUnknownState: "unknown",
}


@router.get(
    "/job-status/{job_id}",
    response_model=JobStatusResponse,
    tags=["Bulk Indexing"],
    summary="Get status of a bulk-insert job (searches all pipelines)",
)
async def job_status(job_id: str):
    """Return the milvus `job_id`."""
    for pipe in enabled_pipelines.values():
        store = _get_milvus_store(pipe)
        if not store:
            continue  # pipeline without Milvus

        try:
            st = store.get_bulk_insert_state(int(job_id))
        except MilvusException as e:
            # not found in this store → try next; any other Milvus error → 500
            if "can't find task" in str(e).lower():
                continue
            raise HTTPException(500, f"Milvus error in pipeline '{pipe.id}': {e}")

        # found → translate state and return
        return JobStatusResponse(
            job_id=job_id,
            status=BULK_STATE_MAP.get(st.state, "unknown"),
            details=(
                getattr(st, "failed_reason", None)
                or getattr(st, "state_name", str(st.state))
            ),
        )

    # exhausted all pipelines
    raise HTTPException(404, f"Bulk insert job {job_id} not found")


# --------------------------------------------------------------------------- #
#  GET /jobs                                                                  #
# --------------------------------------------------------------------------- #


@router.get(
    "/jobs",
    response_model=list[JobDetail],
    tags=["Bulk Indexing"],
    summary="List recent bulk-insert jobs across *all* pipelines",
)
async def list_jobs(
    limit: int | None = None,
    collection_name: str | None = None,
):
    """
    List bulk insert jobs across all enabled pipelines.

    This endpoint queries all Milvus stores across enabled pipelines to gather
    a comprehensive list of bulk insert jobs.

    Args:
        limit: Optional maximum number of jobs to return.
        collection_name: Optional filter to show jobs for a specific collection only.

    Returns:
        List of JobDetail objects containing job information including status,
        progress, collection name, and file paths.
    """
    safe_name = create_safe_name(collection_name) if collection_name else None
    mapping = {
        BulkInsertState.ImportPending: "pending",
        BulkInsertState.ImportStarted: "in_progress",
        BulkInsertState.ImportPersisted: "persisted",
        BulkInsertState.ImportCompleted: "completed",
        BulkInsertState.ImportFailed: "failed",
        BulkInsertState.ImportFailedAndCleaned: "failed_cleaned",
        BulkInsertState.ImportUnknownState: "unknown",
    }

    all_tasks: list[JobDetail] = []
    for pipe in enabled_pipelines.values():
        store = _get_milvus_store(pipe)
        if not store:
            continue  # pipeline without a Milvus store

        try:
            tasks = store.list_bulk_insert_tasks(limit=limit, collection_name=safe_name)
        except MilvusException as e:
            logger.warning("Milvus error listing tasks: %s", e)
            continue

        all_tasks.extend(
            JobDetail(
                job_id=str(t.task_id),
                status=mapping.get(t.state, "unknown"),
                details=getattr(t, "failed_reason", t.state_name),
                progress=getattr(t, "progress", None),
                collection_name=t.collection_name,
                parquet_paths=t.files if isinstance(t.files, list) else [],
            )
            for t in tasks
            if t.task_id != 0  # Filter out invalid/stale tasks
        )

    # Optionally trim the global limit
    if limit is not None:
        all_tasks = all_tasks[:limit]

    return all_tasks
