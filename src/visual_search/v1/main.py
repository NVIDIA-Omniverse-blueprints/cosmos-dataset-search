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

import os

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.visual_search.config import settings
from src.visual_search.v1.apis.bulk_indexing import router as BulkIndexingApiRouter
from src.visual_search.v1.apis.collections import router as CollectionsApiRouter
from src.visual_search.v1.apis.document_indexing import (
    router as DocumentIndexingApiRouter,
)
from src.visual_search.v1.apis.linear_probe import router as LinearProbeApiRouter
from src.visual_search.v1.apis.pipelines import router as PipelinesApiRouter
from src.visual_search.v1.apis.search import router as SearchApiRouter
from src.visual_search.v1.apis.search_refinement import (
    router as SearchRefinementApiRouter,
)
from src.visual_search.v1.apis.milvus_admin import router as MilvusAdminApiRouter

app = FastAPI(
    title="Visual Indexing and Search µService",
    description=(
        "Service for indexing and querying of collections of video data for RAG applications"
    ),
    version="1.1.0",
    contact={
        "name": "#sw-cvcore-visual-search",
        "url": "https://nvidia.com",
    },
    openapi_tags=settings.openapi_tags,
    debug=settings.debug,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Any time any a path operation encounters errors validating a request (e.g.,
    deserializing a JSON payload into a ``pydantic`` dataclass), this function is run.

    ``fastapi``'s default behavior includes the body of the request in exceptions, which
    risks leaking sensitive information into logs.

    This approach attempts to avoid that, while still preserving enough information that
    error messages are helpful for debugging and for differentiating between root causes.

    Notes
    -----

    * description of this approach: https://fastapi.tiangolo.com/tutorial/handling-errors/
    """
    # TODO: could be simplified if https://github.com/tiangolo/fastapi/discussions/10934 happens
    errors = []
    for err in exc.errors():
        if isinstance(err, dict):
            err.pop("input", None)
        errors.append(err)
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": err}),
    )


app.include_router(CollectionsApiRouter)
app.include_router(DocumentIndexingApiRouter)
app.include_router(LinearProbeApiRouter)
app.include_router(SearchApiRouter)
app.include_router(SearchRefinementApiRouter)
app.include_router(PipelinesApiRouter)
app.include_router(MilvusAdminApiRouter)
app.include_router(BulkIndexingApiRouter)

if os.getenv("EXPOSE_BACKFILL_ENDPOINT"):
    app.include_router(BackfillApiRouter)

# /metrics endpoint scraped by Prometheus
Instrumentator().instrument(app).expose(app, tags=["Metrics"])
