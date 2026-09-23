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

"""CE1-compatible HTTP service surface for the CVDS-owned OSS embedder."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import cosmos_embed_oss
from src.cosmos_embed_oss import backends

BASE64_TOKEN = ";base64,"
PRESIGNED_URL_TOKEN = ";presigned_url,"
SUPPORTED_ENCODING_FORMATS = {"float", "base64"}
SUPPORTED_REQUEST_TYPES = {"query", "bulk_text", "bulk_video"}
VIDEO_DATA_PREFIXES = ("data:video/", "data:video_frames/")
DEFAULT_MAX_REQUEST_BYTES = 128 * 1024 * 1024
DEFAULT_SERVICE_VERSION = "1.2.0"
REQUEST_FIELDS = {"encoding_format", "input", "model", "request_type"}


class RequestValidationError(ValueError):
    """Raised when a request would violate the CE1 service contract."""


class RequestBodyError(ValueError):
    """Raised when an HTTP request body cannot be read as JSON."""


class RequestTooLargeError(RequestBodyError):
    """Raised when the declared request body exceeds the configured limit."""


class _ServiceMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._requests = 0
        self._successes = 0
        self._errors = 0
        self._embeddings = 0
        self._videos = 0
        self._total_seconds = 0.0

    def record(
        self,
        *,
        elapsed_seconds: float,
        embedding_count: int = 0,
        video_count: int = 0,
        success: bool,
    ) -> None:
        with self._lock:
            self._requests += 1
            self._successes += int(success)
            self._errors += int(not success)
            self._embeddings += embedding_count
            self._videos += video_count
            self._total_seconds += elapsed_seconds

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average_seconds = (
                self._total_seconds / self._requests if self._requests else 0.0
            )
            return {
                "uptime_seconds": max(0.0, time.monotonic() - self._started_at),
                "requests": {
                    "total": self._requests,
                    "success": self._successes,
                    "error": self._errors,
                },
                "latency": {
                    "total_seconds": self._total_seconds,
                    "average_seconds": average_seconds,
                },
                "business_metrics": {
                    "total_embeddings": self._embeddings,
                    "total_videos": self._videos,
                },
            }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        requests = snapshot["requests"]
        business = snapshot["business_metrics"]
        latency = snapshot["latency"]
        return "\n".join(
            (
                "# TYPE cosmos_embed_requests_total counter",
                f"cosmos_embed_requests_total {requests['total']}",
                "# TYPE cosmos_embed_request_errors_total counter",
                f"cosmos_embed_request_errors_total {requests['error']}",
                "# TYPE cosmos_embed_embeddings_total counter",
                f"cosmos_embed_embeddings_total {business['total_embeddings']}",
                "# TYPE cosmos_embed_videos_total counter",
                f"cosmos_embed_videos_total {business['total_videos']}",
                "# TYPE cosmos_embed_request_duration_seconds_sum counter",
                f"cosmos_embed_request_duration_seconds_sum {latency['total_seconds']}",
                "",
            )
        )


_SERVICE_METRICS = _ServiceMetrics()


@dataclass(frozen=True)
class EmbeddingsRequest:
    """Validated embeddings request used by the service implementation."""

    encoding_format: str
    inputs: list[str]
    model: str
    request_type: str


def configured_embedding_dimension() -> int:
    """Return the configured embedding dimension."""

    return _positive_int_from_env(
        "COSMOS_EMBED_DIM",
        cosmos_embed_oss.DEFAULT_EMBEDDING_DIMENSION,
    )


def configured_max_batch_size() -> int:
    """Return the configured maximum batch size."""

    return _positive_int_from_env(
        "COSMOS_EMBED_MAX_BATCH_SIZE",
        cosmos_embed_oss.DEFAULT_MAX_BATCH_SIZE,
    )


def configured_max_request_bytes() -> int:
    """Return the maximum accepted JSON request size."""

    return _positive_int_from_env(
        "COSMOS_EMBED_MAX_REQUEST_BYTES",
        DEFAULT_MAX_REQUEST_BYTES,
    )


def configured_model_name() -> str:
    """Return the canonical model name exposed by this service."""

    return os.environ.get("COSMOS_EMBED_MODEL", cosmos_embed_oss.DEFAULT_MODEL_NAME)


def configured_service_version() -> str:
    """Return the version advertised by compatibility metadata endpoints."""

    return os.environ.get("COSMOS_EMBED_SERVICE_VERSION", DEFAULT_SERVICE_VERSION)


def parse_embeddings_request(
    payload: Mapping[str, Any],
    *,
    max_batch_size: int | None = None,
    model_name: str | None = None,
) -> EmbeddingsRequest:
    """Validate and normalize a Cosmos-Embed compatible embeddings request."""

    if not isinstance(payload, Mapping):
        raise RequestValidationError("request body must be a JSON object")

    unexpected_fields = sorted(set(payload) - REQUEST_FIELDS)
    if unexpected_fields:
        raise RequestValidationError(
            f"unexpected request field(s): {', '.join(unexpected_fields)}"
        )

    expected_model = model_name or configured_model_name()
    request_model = payload.get("model")
    if not isinstance(request_model, str) or request_model != expected_model:
        raise RequestValidationError(f"model must be {expected_model!r}")

    request_type = payload.get("request_type")
    if not isinstance(request_type, str):
        raise RequestValidationError("request_type is required")
    request_type = request_type.lower()
    if request_type not in SUPPORTED_REQUEST_TYPES:
        supported = ", ".join(sorted(SUPPORTED_REQUEST_TYPES))
        raise RequestValidationError(f"request_type must be one of: {supported}")

    encoding_format = payload.get("encoding_format", "float")
    if not isinstance(encoding_format, str):
        raise RequestValidationError("encoding_format must be a string")
    encoding_format = encoding_format.lower()
    if encoding_format not in SUPPORTED_ENCODING_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_ENCODING_FORMATS))
        raise RequestValidationError(f"encoding_format must be one of: {supported}")

    inputs = _normalize_inputs(payload.get("input"))
    limit = max_batch_size or configured_max_batch_size()
    if len(inputs) > limit:
        raise RequestValidationError(
            f"cosmos-embed supports maximum {limit} inputs per request"
        )

    _validate_request_type_inputs(request_type, inputs)

    return EmbeddingsRequest(
        encoding_format=encoding_format,
        inputs=inputs,
        model=expected_model,
        request_type=request_type,
    )


def build_embeddings_response(
    request: EmbeddingsRequest,
    *,
    backend: backends.EmbeddingBackend | None = None,
) -> dict[str, Any]:
    """Build a CE1-compatible response using the configured embedding backend."""

    resolved_backend = backend or backends.get_embedding_backend()
    input_kinds = [_input_kind_for_value(value) for value in request.inputs]
    embeddings = resolved_backend.embed(
        request.inputs,
        request_type=request.request_type,
        input_kinds=input_kinds,
    )
    if len(embeddings) != len(request.inputs):
        raise backends.EmbeddingBackendError(
            "backend returned the wrong embedding count"
        )

    data = []
    for index, embedding in enumerate(embeddings):
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": _encode_embedding(embedding, request.encoding_format),
            }
        )

    prompt_tokens = sum(
        _token_count(value) for value in request.inputs if not _is_video_data(value)
    )
    return {
        "object": "list",
        "model": request.model,
        "data": data,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            "num_videos": sum(1 for value in request.inputs if _is_video_data(value)),
        },
    }


def run_server(host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    """Create the HTTP server without blocking."""

    resolved_host = host or os.environ.get("NIM_HTTP_HOST", "0.0.0.0")
    resolved_port = (
        port if port is not None else _positive_int_from_env("NIM_HTTP_API_PORT", 8000)
    )
    return ThreadingHTTPServer((resolved_host, resolved_port), CosmosEmbedOSSHandler)


def main() -> None:
    """Run the CE1-compatible OSS service."""

    server = run_server()
    host, port = server.server_address
    print(f"Serving CVDS Cosmos-Embed OSS service on {host}:{port}", flush=True)
    server.serve_forever()


class CosmosEmbedOSSHandler(BaseHTTPRequestHandler):
    """HTTP handler for the CE1-compatible OSS service skeleton."""

    server_version = "CVDSCosmosEmbedOSS/1.2"

    def do_GET(self) -> None:
        """Handle health, model, and metadata endpoints."""

        if self.path == "/v1/health/live":
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "health.response",
                    "message": "Service is live and running",
                    "status": "ready",
                },
            )
            return

        if self.path == "/v1/health/ready":
            try:
                backend = backends.get_embedding_backend()
                backend.ready()
            except (backends.EmbeddingBackendError, OSError, RuntimeError) as error:
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    _error_payload(
                        "service_unavailable",
                        str(error),
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    ),
                )
                return

            self._write_json(
                HTTPStatus.OK,
                _service_metadata_payload(
                    status="ready",
                    load_backend=False,
                    runtime=backend.name,
                ),
            )
            return

        if self.path == "/v1/metrics":
            self._write_text(
                HTTPStatus.OK,
                _SERVICE_METRICS.prometheus(),
                content_type="text/plain; version=0.0.4; charset=utf-8",
            )
            return

        if self.path in {"/health/metrics", "/v1/health/metrics"}:
            self._write_json(HTTPStatus.OK, _SERVICE_METRICS.snapshot())
            return

        if self.path == "/v1/models":
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": configured_model_name(),
                            "created": 1686935002,
                            "object": "model",
                            "owned_by": "nvidia",
                        }
                    ],
                },
            )
            return

        if self.path == "/v1/metadata":
            try:
                backend = backends.get_embedding_backend()
                backend.ready()
                payload = _nim_metadata_payload(runtime=backend.name)
            except (backends.EmbeddingBackendError, OSError, RuntimeError) as error:
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    _error_payload(
                        "service_unavailable",
                        str(error),
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    ),
                )
                return
            self._write_json(HTTPStatus.OK, payload)
            return

        if self.path == "/v1/license":
            try:
                payload = _license_info_payload()
            except OSError:
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    _error_payload(
                        "service_unavailable",
                        "configured service license is unavailable",
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    ),
                )
                return
            self._write_json(HTTPStatus.OK, payload)
            return

        if self.path == "/v1/manifest":
            try:
                manifest_file = _configured_manifest_path().read_text(encoding="utf-8")
            except OSError:
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    _error_payload(
                        "service_unavailable",
                        "configured service manifest is unavailable",
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    ),
                )
                return
            self._write_json(
                HTTPStatus.OK,
                {"manifest_file": manifest_file},
            )
            return

        self._write_json(
            HTTPStatus.NOT_FOUND,
            _error_payload(
                "not_found", f"unknown endpoint: {self.path}", HTTPStatus.NOT_FOUND
            ),
        )

    def do_POST(self) -> None:
        """Handle the embeddings endpoint."""

        if self.path != "/v1/embeddings":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                _error_payload(
                    "not_found", f"unknown endpoint: {self.path}", HTTPStatus.NOT_FOUND
                ),
            )
            return

        started_at = time.monotonic()
        request: EmbeddingsRequest | None = None
        try:
            payload = self._read_json()
            request = parse_embeddings_request(payload)
            response = build_embeddings_response(request)
        except RequestValidationError as error:
            _SERVICE_METRICS.record(
                elapsed_seconds=time.monotonic() - started_at,
                success=False,
            )
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                _error_payload(
                    "invalid_request_error",
                    str(error),
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                ),
            )
            return
        except RequestTooLargeError as error:
            _SERVICE_METRICS.record(
                elapsed_seconds=time.monotonic() - started_at,
                success=False,
            )
            self._write_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                _error_payload(
                    "request_too_large",
                    str(error),
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                ),
            )
            return
        except (RequestBodyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            _SERVICE_METRICS.record(
                elapsed_seconds=time.monotonic() - started_at,
                success=False,
            )
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                _error_payload("invalid_json", str(error), HTTPStatus.BAD_REQUEST),
            )
            return
        except backends.EmbeddingInputError as error:
            _SERVICE_METRICS.record(
                elapsed_seconds=time.monotonic() - started_at,
                success=False,
            )
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                _error_payload(
                    "invalid_request_error",
                    str(error),
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                ),
            )
            return
        except backends.MediaDownloadError as error:
            _SERVICE_METRICS.record(
                elapsed_seconds=time.monotonic() - started_at,
                success=False,
            )
            self._write_json(
                HTTPStatus.BAD_GATEWAY,
                _error_payload(
                    "media_download_error",
                    str(error),
                    HTTPStatus.BAD_GATEWAY,
                ),
            )
            return
        except (backends.EmbeddingBackendError, RuntimeError) as error:
            _SERVICE_METRICS.record(
                elapsed_seconds=time.monotonic() - started_at,
                success=False,
            )
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                _error_payload(
                    "service_unavailable",
                    str(error),
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ),
            )
            return

        _SERVICE_METRICS.record(
            elapsed_seconds=time.monotonic() - started_at,
            embedding_count=len(response["data"]),
            video_count=response["usage"]["num_videos"],
            success=True,
        )
        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep unit test output quiet unless the embedding service fails."""

    def _read_json(self) -> Mapping[str, Any]:
        raw_content_length = self.headers.get("content-length")
        if raw_content_length is None:
            raise RequestBodyError("Content-Length header is required")
        try:
            content_length = int(raw_content_length)
        except ValueError as error:
            raise RequestBodyError(
                "Content-Length header must be an integer"
            ) from error
        if content_length <= 0:
            raise RequestBodyError("Content-Length header must be positive")
        max_request_bytes = configured_max_request_bytes()
        if content_length > max_request_bytes:
            raise RequestTooLargeError(
                f"request body exceeds maximum {max_request_bytes} bytes"
            )
        payload = self.rfile.read(content_length)
        if len(payload) != content_length:
            raise RequestBodyError("request body ended before Content-Length bytes")
        return json.loads(payload.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(
        self,
        status: HTTPStatus,
        payload: str,
        *,
        content_type: str,
    ) -> None:
        body = payload.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _normalize_inputs(raw_input: Any) -> list[str]:
    if isinstance(raw_input, str):
        values = [raw_input]
    elif isinstance(raw_input, list) and all(
        isinstance(value, str) for value in raw_input
    ):
        values = raw_input
    else:
        raise RequestValidationError("input must be a string or list of strings")

    if not values:
        raise RequestValidationError("input must not be empty")
    return values


def _validate_request_type_inputs(request_type: str, inputs: Sequence[str]) -> None:
    if request_type == "bulk_text":
        if any(_is_video_data(value) for value in inputs):
            raise RequestValidationError("bulk_text input must contain text only")
        return

    if request_type == "bulk_video":
        if any(not _is_video_data(value) for value in inputs):
            raise RequestValidationError(
                "bulk_video input must contain video data URIs only"
            )
        media_tokens = {_video_media_token(value) for value in inputs}
        if None in media_tokens:
            raise RequestValidationError(
                "bulk_video inputs must use ;base64, or ;presigned_url, video payloads"
            )
        if len(media_tokens) > 1:
            raise RequestValidationError(
                "Cannot mix base64 and presigned URL inputs in same request"
            )
        for value in inputs:
            if value.startswith("data:video_frames/"):
                try:
                    backends.parse_video_frames_data_uri(value)
                except backends.EmbeddingInputError as error:
                    raise RequestValidationError(str(error)) from error
            elif PRESIGNED_URL_TOKEN not in value:
                raise RequestValidationError(
                    "bulk_video full-video inputs must use ;presigned_url, payloads"
                )
        return

    if request_type == "query":
        video_inputs = [value for value in inputs if _is_video_data(value)]
        if not video_inputs:
            # Current CVDS text clients use query with a list. Preserve that
            # compatibility extension while matching NIM video semantics.
            return
        if len(inputs) != 1:
            raise RequestValidationError(
                "query video input must contain exactly one video"
            )
        if _video_media_token(video_inputs[0]) is None:
            raise RequestValidationError(
                "query video input must use ;base64, or ;presigned_url,"
            )
        if video_inputs[0].startswith("data:video_frames/"):
            try:
                backends.parse_video_frames_data_uri(video_inputs[0])
            except backends.EmbeddingInputError as error:
                raise RequestValidationError(str(error)) from error


def _encode_embedding(
    embedding: Sequence[float], encoding_format: str
) -> list[float] | str:
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(embedding)}f", *embedding)).decode(
            "ascii"
        )
    return list(embedding)


def _error_payload(error_type: str, detail: str, status: HTTPStatus) -> dict[str, Any]:
    return {
        "error": {
            "type": error_type,
            "detail": detail,
            "status_code": status.value,
        }
    }


def _input_kind_for_value(value: str) -> str:
    return "video" if _is_video_data(value) else "text"


def _is_video_data(value: str) -> bool:
    return value.startswith(VIDEO_DATA_PREFIXES)


def _positive_int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if parsed < 1:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _token_count(value: str) -> int:
    return max(1, len(value.split()))


def _service_metadata_payload(
    *,
    status: str,
    load_backend: bool,
    runtime: str | None = None,
) -> dict[str, Any]:
    config = backends.backend_config_from_env()
    resolved_runtime = runtime
    if resolved_runtime is None:
        resolved_runtime = (
            backends.get_embedding_backend().name
            if load_backend
            else backends.backend_name_for_config(config)
        )
    return {
        "object": "health.response",
        "message": "Model is ready",
        "status": status,
        "model": configured_model_name(),
        "runtime": resolved_runtime,
        "embedding_dim": config.dimension,
        "max_batch_size": configured_max_batch_size(),
        "request_types": sorted(SUPPORTED_REQUEST_TYPES),
        "encoding_formats": sorted(SUPPORTED_ENCODING_FORMATS),
    }


def _nim_metadata_payload(*, runtime: str) -> dict[str, Any]:
    config = backends.backend_config_from_env()
    return {
        "version": configured_service_version(),
        "modelInfo": [
            {
                "shortName": configured_model_name(),
                "modelUrl": (
                    config.model_path
                    or "https://huggingface.co/nvidia/Cosmos-Embed1-224p"
                ),
            }
        ],
        "assetInfo": [],
        "licenseInfo": _license_info_payload(),
        "runtime": runtime,
    }


def _license_info_payload() -> dict[str, Any]:
    license_path = _configured_license_path()
    content = license_path.read_text(encoding="utf-8")
    encoded_content = content.encode("utf-8")
    return {
        "name": "NVIDIA Software and Model Evaluation License",
        "path": str(license_path),
        "sha": hashlib.sha256(encoded_content).hexdigest(),
        "size": len(encoded_content),
        "url": (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-software-and-model-evaluation-license/"
        ),
        "type": "text/plain",
        "content": content,
    }


def _configured_license_path() -> Path:
    configured = os.environ.get("COSMOS_EMBED_LICENSE_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "LICENSE"


def _configured_manifest_path() -> Path:
    configured = os.environ.get("COSMOS_EMBED_MANIFEST_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).with_name("manifest.yaml")


def _video_media_token(value: str) -> str | None:
    if BASE64_TOKEN in value:
        return BASE64_TOKEN
    if PRESIGNED_URL_TOKEN in value:
        return PRESIGNED_URL_TOKEN
    return None


if __name__ == "__main__":
    main()
