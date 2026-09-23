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
import traceback
from http import HTTPStatus
from typing import ByteString, Callable, Dict, Union

import exceptiongroup
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as SRequest
from starlette.responses import Response as SResponse
from starlette.responses import StreamingResponse

from src.visual_search.exceptions import InputValidationError

from .common.dependencies import app_dependencies
from .common.models import ErrorResponse
from .config import settings
from .logger import logger
from .v1.main import app as app_v1

app = FastAPI(
    title="Visual Indexing and Search µService",
    description="Service for indexing and querying of "
    "collections of image and video data for RAG applications",
    version="Latest version: 1.0.0-rc",
    servers=[
        {
            "name": "#sw-cvcore-visual-search",
            "url": "https://nvidia.com",
        },
    ],
    lifespan=app_dependencies,
    debug=settings.debug,
    root_path=os.getenv("FASTAPI_ROOT_PATH", None),
)

tracer = trace.get_tracer(__name__)

# region ============ Middleware configuration section ============


class PrettyJSONMiddleware(BaseHTTPMiddleware):
    async def read_body(self, response: StreamingResponse) -> ByteString:
        body = b""
        async for chunk in response.body_iterator:
            assert isinstance(chunk, bytes)  # add this line to ensure chunks are bytes
            body += chunk
        return body

    async def dispatch(
        self, request: SRequest, call_next: RequestResponseEndpoint
    ) -> SResponse:
        response = await call_next(request)
        pretty_query = (
            "pretty" in request.query_params
            and request.query_params["pretty"] == "true"
        )

        pretty_env = os.environ.get("DEBUG_PRETTY") or False
        pretty_mode = pretty_query or pretty_env
        if pretty_mode and response.headers.get("content-type") == "application/json":
            # this is for large response bodies
            if isinstance(response, StreamingResponse):
                response_body = await self.read_body(response)
                if isinstance(response_body, memoryview):
                    response_body = response_body.tobytes()
                data = json.loads(response_body.decode())  # type: ignore

            # this is for normal sized bodies
            else:
                response_body = response.body
                data = json.loads(response_body.decode())  # type: ignore

            pretty_body = json.dumps(data, indent=2).encode("utf-8")
            headers = dict(response.headers)
            headers = {key.lower(): value for key, value in response.headers.items()}
            headers.pop("content-length", None)
            headers["content-length"] = str(len(pretty_body))
            return SResponse(
                pretty_body,
                media_type="application/json",
                headers=headers,
                status_code=response.status_code,
            )

        return response


class CaptureTraceId(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = trace.get_current_span().get_span_context().trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = format(trace_id, "x")

        return response


def create_error_response(
    status_code: HTTPStatus,
    message: str,
    error_type: str,
    detail: Union[str, Dict] = {},
) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type=error_type, detail=detail).dict(),
        status_code=status_code.value,
    )


def handle_unhandled_exception(request: Request, exc: Exception) -> Response:
    traceback.print_exception(exc)
    logger.error(traceback.format_exception(exc))

    if isinstance(exc, exceptiongroup.ExceptionGroup):
        exc_detail = ",".join(
            [f"{type(e).__name__}: {e}" for _, e in enumerate(exc.exceptions)]
        )
        # Check if any exception in the group is InputValidationError
        if any(isinstance(e, InputValidationError) for e in exc.exceptions):
            return create_error_response(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                message="",
                detail=exc_detail,
                error_type="input_validation_error",
            )
    elif isinstance(exc, InputValidationError):
        exc_detail = f"{type(exc).__name__}: {exc}"
        return create_error_response(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            message="",
            detail=exc_detail,
            error_type="input_validation_error",
        )
    else:
        exc_detail = f"{type(exc).__name__}: {exc}"

    return create_error_response(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        message="",
        detail=f"Something went wrong with the request: <{exc_detail}>",
        error_type="internal_server_error",
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_domains,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(PrettyJSONMiddleware)

# Unhandled exception (500 - Internal Server Error) handler
app_v1.add_exception_handler(Exception, handle_unhandled_exception)

# Due to the nature of trace handling a context of a thread of a mounted sub-app (FastAPI),
# CaptureTraceId can only propely work and capture the current span if configured for the
# mounted sub-app.
app_v1.add_middleware(CaptureTraceId)

# endregion

# To be able to capture trace id for a mounted sub-app,
# make sure to add CaptureTraceId middleware for it! (see above)
app.mount("/v1", app_v1)


@app.get(
    "/health",
    tags=["Health Checks"],
    summary="Perform a Health Check",
    description="""
        Immediately returns 200 when service is up.
        This does not check the health of downstream
        services.
    """,
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_health() -> str:
    # Perform a health check
    return "OK"


@app.get(
    "/callback",
    tags=["OIDC callback"],
    summary="OIDC Callback",
    description="Callback URL for OIDC",
    response_description="Redirect to the OIDC callback URL",
    status_code=status.HTTP_200_OK,
)
def get_oidc_callback() -> str:
    return "OK"
