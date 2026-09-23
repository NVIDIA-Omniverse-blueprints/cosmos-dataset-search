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

"""Embedding backends for the CVDS-owned Cosmos-Embed OSS service."""

from __future__ import annotations

import base64
import binascii
import http.client
import ipaddress
import math
import os
import socket
import tempfile
import threading
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from src import cosmos_embed_oss
from src.cosmos_embed_oss import model_integrity, runtime_check
from src.cosmos_embed_oss.media_probe import GpuDecodeUnsupported

BASE64_VIDEO_TOKEN = ";base64,"
CE1_VIDEO_FRAME_COUNT = 8
DEFAULT_BACKEND_MODE = "pytorch"
DEFAULT_MAX_MEDIA_BYTES = 96 * 1024 * 1024
DEFAULT_MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 300.0
DEFAULT_MEDIA_TMP_DIR = "/tmp/ram"
DEFAULT_PYTORCH_VARIANT = "224p"
TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
VIDEO_PIPELINE_DEPTH = 4
PRESIGNED_URL_ALLOWED_HOSTS_ENV = "COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HOSTS"
PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV = (
    "COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HTTP_ORIGINS"
)
PRESIGNED_URL_ENDPOINT_ENV = "COSMOS_EMBED_PRESIGNED_URL_ENDPOINT_URL"
PRESIGNED_URL_VIDEO_TOKEN = ";presigned_url,"
PYTORCH_VARIANT_TO_MODEL = {
    "224p": "nvidia/Cosmos-Embed1-224p",
    "336p": "nvidia/Cosmos-Embed1-336p",
    "448p": "nvidia/Cosmos-Embed1-448p",
}
DEFAULT_HF_MODEL_ID = PYTORCH_VARIANT_TO_MODEL[DEFAULT_PYTORCH_VARIANT]
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
PYTORCH_BACKEND_ALIASES = {"ce1-pytorch", "pytorch", "pytorch-ce1"}
VIDEO_EXTENSION_BY_MEDIA_TYPE = {
    "video/avi": ".avi",
    "video/mp4": ".mp4",
    "video/mov": ".mov",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

MediaDownloader = Callable[[str, Path, float, int], None]


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot serve the requested payload."""


class EmbeddingInputError(EmbeddingBackendError):
    """Raised when request media is malformed or violates configured limits."""


class MediaDownloadError(EmbeddingBackendError):
    """Raised when a validated remote media download fails."""


class EmbeddingBackend(Protocol):
    """Common service-facing embedding backend surface."""

    name: str

    def ready(self) -> None:
        """Raise if the backend cannot currently serve embeddings."""

    def close(self) -> None:
        """Release backend resources."""

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_type: str,
        input_kinds: Sequence[str],
    ) -> list[list[float]]:
        """Return one embedding per input."""


@dataclass(frozen=True)
class BackendConfig:
    """Runtime configuration for the embedding backend."""

    mode: str = DEFAULT_BACKEND_MODE
    model_path: str | None = None
    variant: str = DEFAULT_PYTORCH_VARIANT
    dimension: int = cosmos_embed_oss.DEFAULT_EMBEDDING_DIMENSION
    device: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = True
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES
    allow_hf_download: bool = False
    hf_model_id: str = DEFAULT_HF_MODEL_ID
    hf_revision: str = model_integrity.MODEL_REVISION
    media_tmp_dir: str = DEFAULT_MEDIA_TMP_DIR
    media_download_timeout_seconds: float = DEFAULT_MEDIA_DOWNLOAD_TIMEOUT_SECONDS
    sdpa: bool = False
    torch_compile: bool = False
    gpu_frames: bool = False
    gpu_video: bool = True
    preparation_workers: int = 1


@dataclass(frozen=True)
class _TorchRuntime:
    model: Any
    processor: Any
    torch: Any
    device: str
    dtype: Any
    gpu_media: Any | None = None


@dataclass(frozen=True)
class _ParsedVideoDataUri:
    media_type: str
    payload: str
    payload_kind: str


@dataclass(frozen=True)
class _ParsedVideoFramesDataUri:
    frame_format: str
    payload_kind: str
    payloads: list[str]


@dataclass(frozen=True)
class _MaterializedVideo:
    media_type: str
    path: Path
    source_kind: str


@dataclass(frozen=True)
class _ResolvedPytorchModel:
    reference: str
    source: str
    local_files_only: bool
    use_auth_token: bool = False


_BACKEND_CACHE: tuple[tuple[str | None, ...], EmbeddingBackend] | None = None
_BACKEND_LOCK = threading.RLock()


def backend_config_from_env() -> BackendConfig:
    """Build backend configuration from environment variables."""

    return BackendConfig(
        mode=os.environ.get("COSMOS_EMBED_BACKEND", DEFAULT_BACKEND_MODE),
        model_path=os.environ.get("COSMOS_EMBED_MODEL_PATH")
        or os.environ.get("COSMOS_EMBED_WEIGHTS_DIR"),
        variant=os.environ.get("COSMOS_EMBED_MODEL_VARIANT", DEFAULT_PYTORCH_VARIANT),
        dimension=_positive_int_from_env(
            "COSMOS_EMBED_DIM",
            cosmos_embed_oss.DEFAULT_EMBEDDING_DIMENSION,
        ),
        device=os.environ.get("COSMOS_EMBED_DEVICE"),
        trust_remote_code=_bool_from_env("COSMOS_EMBED_TRUST_REMOTE_CODE", True),
        local_files_only=_bool_from_env("COSMOS_EMBED_LOCAL_FILES_ONLY", True),
        max_media_bytes=_positive_int_from_env(
            "COSMOS_EMBED_MAX_MEDIA_BYTES",
            DEFAULT_MAX_MEDIA_BYTES,
        ),
        allow_hf_download=_bool_from_env("COSMOS_EMBED_ALLOW_HF_DOWNLOAD", False),
        hf_model_id=os.environ.get("COSMOS_EMBED_HF_MODEL_ID", DEFAULT_HF_MODEL_ID),
        hf_revision=os.environ.get(
            "COSMOS_EMBED_HF_REVISION", model_integrity.MODEL_REVISION,
        ),
        media_tmp_dir=os.environ.get("COSMOS_EMBED_TMP_DIR", DEFAULT_MEDIA_TMP_DIR),
        media_download_timeout_seconds=_positive_float_from_env(
            "COSMOS_EMBED_MEDIA_DOWNLOAD_TIMEOUT_SECONDS",
            DEFAULT_MEDIA_DOWNLOAD_TIMEOUT_SECONDS,
        ),
        sdpa=_bool_from_env("COSMOS_EMBED_SDPA", False),
        torch_compile=_bool_from_env("COSMOS_EMBED_TORCH_COMPILE", False),
        gpu_frames=_bool_from_env("COSMOS_EMBED_GPU_FRAMES", False),
        gpu_video=_bool_from_env("COSMOS_EMBED_GPU_VIDEO", True),
        preparation_workers=_positive_int_from_env(
            "COSMOS_EMBED_PREPARATION_WORKERS",
            1,
        ),
    )


def backend_name_for_config(config: BackendConfig) -> str:
    """Return the public backend name for a config without loading model deps."""

    _normalize_backend_mode(config.mode)
    return f"pytorch-ce1-{config.variant.lower()}"


def create_embedding_backend(config: BackendConfig) -> EmbeddingBackend:
    """Create an embedding backend from explicit configuration."""

    mode = _normalize_backend_mode(config.mode)
    if mode == "pytorch":
        return PyTorchCosmosEmbedBackend(config)
    raise AssertionError(f"unexpected backend mode {mode!r}")


def get_embedding_backend() -> EmbeddingBackend:
    """Return the configured process-wide embedding backend."""

    global _BACKEND_CACHE

    fingerprint = _backend_env_fingerprint()
    with _BACKEND_LOCK:
        if _BACKEND_CACHE is None or _BACKEND_CACHE[0] != fingerprint:
            _BACKEND_CACHE = (
                fingerprint,
                create_embedding_backend(backend_config_from_env()),
            )
        return _BACKEND_CACHE[1]


def reset_embedding_backend_for_tests() -> None:
    """Clear the process-wide backend cache."""

    global _BACKEND_CACHE
    with _BACKEND_LOCK:
        if _BACKEND_CACHE is not None:
            _BACKEND_CACHE[1].close()
        _BACKEND_CACHE = None


def normalize_embedding(
    embedding: Any,
    *,
    dimension: int = cosmos_embed_oss.DEFAULT_EMBEDDING_DIMENSION,
) -> list[float]:
    """Coerce and L2-normalize a backend embedding to the CVDS CE1 dimension."""

    vector = _as_flat_float_vector(embedding)
    if len(vector) != dimension:
        raise EmbeddingBackendError(
            f"expected {dimension}-d CE1 embedding, got {len(vector)}"
        )

    norm = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise EmbeddingBackendError("CE1 backend returned a zero or invalid embedding")
    return [component / norm for component in vector]


@contextmanager
def materialize_video_data_uri(
    data_uri: str,
    *,
    tmp_dir: Path,
    download_timeout_seconds: float,
    max_media_bytes: int,
    media_downloader: MediaDownloader | None = None,
) -> Iterator[_MaterializedVideo]:
    """Materialize a CVDS CE1 video data URI as a temporary local file."""

    parsed = parse_video_data_uri(data_uri)
    resolved_downloader = media_downloader or download_presigned_video
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        suffix = _video_suffix_for_media_type(parsed.media_type)

        with tempfile.TemporaryDirectory(
            prefix="cvds-ce1-", dir=str(tmp_dir)
        ) as work_dir:
            media_path = Path(work_dir) / f"input{suffix}"
            if parsed.payload_kind == "base64":
                _write_base64_payload(
                    parsed.payload,
                    media_path,
                    max_bytes=max_media_bytes,
                    payload_name="video",
                )
            elif parsed.payload_kind == "presigned_url":
                resolved_downloader(
                    parsed.payload,
                    media_path,
                    download_timeout_seconds,
                    max_media_bytes,
                )
            else:
                raise EmbeddingBackendError(
                    f"unsupported video payload kind: {parsed.payload_kind}"
                )

            if not media_path.exists() or media_path.stat().st_size == 0:
                raise EmbeddingBackendError(
                    "video payload materialized to an empty file"
                )

            yield _MaterializedVideo(
                media_type=parsed.media_type,
                path=media_path,
                source_kind=parsed.payload_kind,
            )
    except EmbeddingBackendError:
        raise
    except OSError as error:
        raise EmbeddingBackendError("failed to materialize video payload") from error


def parse_video_data_uri(data_uri: str) -> _ParsedVideoDataUri:
    """Parse the CE1-compatible video data URI shapes used by CVDS clients."""

    for token, payload_kind in (
        (BASE64_VIDEO_TOKEN, "base64"),
        (PRESIGNED_URL_VIDEO_TOKEN, "presigned_url"),
    ):
        if token not in data_uri:
            continue
        header, payload = data_uri.split(token, 1)
        media_type = _media_type_from_data_uri_header(header)
        if not payload:
            raise EmbeddingInputError(f"{payload_kind} video payload must not be empty")
        return _ParsedVideoDataUri(
            media_type=media_type,
            payload=payload,
            payload_kind=payload_kind,
        )

    raise EmbeddingInputError(
        "video input must use ;base64, or ;presigned_url, data URI payloads"
    )


def parse_video_frames_data_uri(data_uri: str) -> _ParsedVideoFramesDataUri:
    """Parse a CE1 video_frames URI containing exactly eight image payloads."""

    for token, payload_kind in (
        (BASE64_VIDEO_TOKEN, "base64"),
        (PRESIGNED_URL_VIDEO_TOKEN, "presigned_url"),
    ):
        if token not in data_uri:
            continue
        header, raw_payloads = data_uri.split(token, 1)
        prefix = "data:video_frames/"
        if not header.startswith(prefix):
            raise EmbeddingInputError(
                "video_frames input must start with data:video_frames/"
            )
        frame_format = header[len(prefix) :].strip().lower()
        if not frame_format or not frame_format.replace("-", "").isalnum():
            raise EmbeddingInputError("video_frames format is invalid")
        raw_payloads = raw_payloads.strip()
        if raw_payloads.startswith("{") and raw_payloads.endswith("}"):
            raw_payloads = raw_payloads[1:-1]
        payloads = [payload.strip() for payload in raw_payloads.split(",")]
        if len(payloads) != CE1_VIDEO_FRAME_COUNT or any(
            not payload for payload in payloads
        ):
            raise EmbeddingInputError(
                f"video_frames input must contain exactly {CE1_VIDEO_FRAME_COUNT} frames"
            )
        return _ParsedVideoFramesDataUri(
            frame_format=frame_format,
            payload_kind=payload_kind,
            payloads=payloads,
        )

    raise EmbeddingInputError(
        "video_frames input must use ;base64, or ;presigned_url, payloads"
    )


def decode_video_frames_data_uri(
    data_uri: str,
    *,
    tmp_dir: Path,
    download_timeout_seconds: float,
    max_media_bytes: int,
    media_downloader: MediaDownloader | None = None,
) -> list[Any]:
    """Decode exactly eight image payloads into RGB arrays."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise EmbeddingBackendError(
            "PyTorch CE1 video_frames backend requires Pillow and numpy"
        ) from error

    frame_payloads = read_video_frames_data_uri(
        data_uri,
        tmp_dir=tmp_dir,
        download_timeout_seconds=download_timeout_seconds,
        max_media_bytes=max_media_bytes,
        media_downloader=media_downloader,
    )
    frames: list[Any] = []
    for index, frame_bytes in enumerate(frame_payloads):
        try:
            with Image.open(BytesIO(frame_bytes)) as image:
                frames.append(np.asarray(image.convert("RGB")))
        except Exception as error:
            raise EmbeddingInputError(
                f"video frame {index} is not a supported image"
            ) from error
    return frames


def read_video_frames_data_uri(
    data_uri: str,
    *,
    tmp_dir: Path,
    download_timeout_seconds: float,
    max_media_bytes: int,
    media_downloader: MediaDownloader | None = None,
) -> list[bytes]:
    """Read and validate exactly eight encoded image payloads."""

    parsed = parse_video_frames_data_uri(data_uri)
    resolved_downloader = media_downloader or download_presigned_video
    total_bytes = 0
    frame_payloads: list[bytes] = []
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="cvds-ce1-frames-", dir=str(tmp_dir)
        ) as work_dir:
            for index, payload in enumerate(parsed.payloads):
                remaining_bytes = max_media_bytes - total_bytes
                if remaining_bytes <= 0:
                    raise EmbeddingInputError(
                        f"video_frames media exceeds maximum {max_media_bytes} bytes"
                    )

                if parsed.payload_kind == "base64":
                    frame_bytes = _decode_base64_payload(
                        payload,
                        max_bytes=remaining_bytes,
                        payload_name=f"video frame {index}",
                    )
                else:
                    frame_path = Path(work_dir) / (
                        f"frame-{index}{_image_suffix_for_format(parsed.frame_format)}"
                    )
                    resolved_downloader(
                        payload,
                        frame_path,
                        download_timeout_seconds,
                        remaining_bytes,
                    )
                    frame_bytes = frame_path.read_bytes()

                total_bytes += len(frame_bytes)
                if total_bytes > max_media_bytes:
                    raise EmbeddingInputError(
                        f"video_frames media exceeds maximum {max_media_bytes} bytes"
                    )
                frame_payloads.append(frame_bytes)
    except EmbeddingBackendError:
        raise
    except OSError as error:
        raise EmbeddingBackendError("failed to read video_frames payload") from error
    return frame_payloads


def download_presigned_video(
    url: str,
    destination_path: Path,
    timeout_seconds: float,
    max_bytes: int,
) -> None:
    """Download a presigned video URL to a local temporary file."""

    download_url = _presigned_video_download_url(url)
    _validate_presigned_download_url(download_url)

    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "cvds-cosmos-embed-oss/0.1"},
    )
    opener = urllib.request.build_opener(
        _ValidatingRedirectHandler(), _PinnedHTTPSHandler()
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise MediaDownloadError(
                    f"presigned video download failed with HTTP {status}"
                )
            declared_size: int | None = None
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise MediaDownloadError(
                        "presigned video returned an invalid Content-Length"
                    ) from error
                if declared_size < 0:
                    raise MediaDownloadError(
                        "presigned video returned an invalid Content-Length"
                    )
                if declared_size > max_bytes:
                    raise EmbeddingInputError(
                        f"remote media exceeds maximum {max_bytes} bytes"
                    )
            with destination_path.open("wb") as output:
                downloaded_size = _copy_response_with_limit(
                    response,
                    output,
                    max_bytes=max_bytes,
                )
            if declared_size is not None and downloaded_size != declared_size:
                raise MediaDownloadError(
                    "presigned video download ended before Content-Length bytes"
                )
    except EmbeddingBackendError:
        raise
    except urllib.error.HTTPError as error:
        raise MediaDownloadError(
            f"presigned video download failed with HTTP {error.code}"
        ) from error
    except (
        OSError,
        urllib.error.URLError,
        TimeoutError,
        http.client.HTTPException,
    ) as error:
        raise MediaDownloadError(
            "failed to download presigned video payload"
        ) from error


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request: Any) -> Any:
        target = urllib.parse.urlsplit(request.full_url)
        connection = partial(
            _PinnedHTTPSConnection,
            media_host=target.hostname,
            media_port=_parsed_url_port(target),
            allow_private=_is_configured_presigned_endpoint(target),
        )
        return self.do_open(connection, request, context=self._context)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect only to validated addresses, retaining the original TLS name."""

    def __init__(
        self,
        *args: Any,
        media_host: str,
        media_port: int,
        allow_private: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._media_host = media_host
        self._media_port = media_port
        self._allow_private = allow_private
        self._media_addresses: list[Any] = []
        self._tunnel_address: str | None = None

    def request(self, *args: Any, **kwargs: Any) -> None:
        # A failed proxy CONNECT closes the connection and resets its HTTP
        # state. Finish connection retries before formatting the GET request.
        if self.sock is None:
            self.connect()
        super().request(*args, **kwargs)

    def connect(self) -> None:
        # Revalidate the actual dial addresses. An earlier URL check can race
        # a DNS change; never pass the hostname back to socket.create_connection.
        self._media_addresses = _resolve_presigned_host(
            self._media_host, self._media_port, allow_private=self._allow_private
        )
        if not self._tunnel_host:
            self._create_connection = self._connect_media
            super().connect()
            return

        # Keep the operator's proxy and its authentication. CONNECT uses the
        # validated numeric target so the proxy cannot resolve that name again.
        last_error: OSError | None = None
        for address in self._media_addresses:
            self._tunnel_address = address[4][0]
            try:
                super().connect()
                return
            except OSError as error:
                self.close()
                last_error = error
        assert last_error is not None
        raise last_error

    def _connect_media(
        self, address: Any, timeout: Any, source_address: Any
    ) -> socket.socket:
        del address  # The validated address list is authoritative, not a hostname.
        return _connect_resolved_addresses(
            self._media_addresses, timeout, source_address
        )

    def _tunnel(self) -> None:
        original_host = self._tunnel_host
        assert self._tunnel_address is not None
        self._tunnel_host = self._tunnel_address
        if ":" in self._tunnel_host:
            self._tunnel_host = f"[{self._tunnel_host}]"
        try:
            super()._tunnel()
        finally:
            # HTTPSConnection.connect uses this name for SNI and certificate
            # verification after CONNECT. Do not verify against the numeric IP.
            self._tunnel_host = original_host


def _connect_resolved_addresses(
    addresses: Sequence[Any], timeout: Any, source_address: Any
) -> socket.socket:
    last_error: OSError | None = None
    for family, socktype, protocol, _canonical_name, sockaddr in addresses:
        connection = None
        try:
            connection = socket.socket(family, socktype, protocol)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                connection.settimeout(timeout)
            if source_address:
                connection.bind(source_address)
            connection.connect(sockaddr)
            return connection
        except OSError as error:
            if connection is not None:
                connection.close()
            last_error = error
    if last_error is None:
        raise MediaDownloadError("presigned video host did not resolve")
    raise last_error


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        _validate_presigned_download_url(resolved_url)
        return super().redirect_request(req, fp, code, msg, headers, resolved_url)


def _presigned_video_download_url(url: str) -> str:
    original = urllib.parse.urlsplit(url)
    if original.scheme not in {"http", "https"} or not original.hostname:
        raise EmbeddingInputError("presigned video URL must be an absolute http(s) URL")
    if original.username or original.password:
        raise EmbeddingInputError("presigned video URL must not contain credentials")
    _parsed_url_port(original)

    endpoint_override = os.environ.get(PRESIGNED_URL_ENDPOINT_ENV)
    if not endpoint_override:
        return url

    endpoint = urllib.parse.urlsplit(endpoint_override)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise EmbeddingBackendError(
            f"{PRESIGNED_URL_ENDPOINT_ENV} must be an absolute http(s) URL"
        )
    if endpoint.username or endpoint.password:
        raise EmbeddingBackendError(
            f"{PRESIGNED_URL_ENDPOINT_ENV} must not contain credentials"
        )
    _parsed_url_port(endpoint, config_name=PRESIGNED_URL_ENDPOINT_ENV)
    if not _presigned_endpoint_override_applies(original, endpoint):
        return url
    return urllib.parse.urlunsplit(
        (
            endpoint.scheme,
            endpoint.netloc,
            original.path,
            original.query,
            "",
        )
    )


def _presigned_endpoint_override_applies(
    original: urllib.parse.SplitResult,
    endpoint: urllib.parse.SplitResult,
) -> bool:
    """Return whether an operator endpoint represents the input URL's origin."""

    original_host = original.hostname.lower().rstrip(".")
    endpoint_host = endpoint.hostname.lower().rstrip(".")
    if _parsed_url_port(original) != _parsed_url_port(
        endpoint,
        config_name=PRESIGNED_URL_ENDPOINT_ENV,
    ):
        return False
    if original_host == endpoint_host:
        return True
    if original_host in {"localhost", "host.docker.internal"}:
        return True
    try:
        return ipaddress.ip_address(original_host).is_loopback
    except ValueError:
        return False


def _validate_presigned_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise EmbeddingInputError("presigned video URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise EmbeddingInputError("presigned video URL must not contain credentials")
    parsed_port = _parsed_url_port(parsed)

    is_configured_endpoint = _is_configured_presigned_endpoint(parsed)
    is_allowed_http_origin = (
        parsed.scheme == "http"
        and (
            parsed.scheme.lower(),
            parsed.hostname.lower().rstrip("."),
            parsed_port,
        )
        in _configured_allowed_presigned_http_origins()
    )
    if (
        parsed.scheme != "https"
        and not is_configured_endpoint
        and not is_allowed_http_origin
    ):
        raise EmbeddingInputError(
            "presigned video URL must use https unless its exact HTTP origin or an operator endpoint override is configured"
        )

    if not is_configured_endpoint and not is_allowed_http_origin:
        allowed_hosts = _configured_allowed_presigned_hosts()
        if allowed_hosts and not any(
            _host_matches_pattern(parsed.hostname, pattern) for pattern in allowed_hosts
        ):
            raise EmbeddingInputError(
                "presigned video host is not in the configured allowlist"
            )
        _reject_non_public_host(parsed.hostname, parsed_port)


def _is_configured_presigned_endpoint(parsed: urllib.parse.SplitResult) -> bool:
    endpoint = _configured_endpoint_origin()
    return (
        endpoint is not None
        and (
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            _parsed_url_port(parsed),
        )
        == endpoint
    )


def _configured_endpoint_origin() -> tuple[str, str, int] | None:
    raw_endpoint = os.environ.get(PRESIGNED_URL_ENDPOINT_ENV)
    if not raw_endpoint:
        return None
    endpoint = urllib.parse.urlsplit(raw_endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise EmbeddingBackendError(
            f"{PRESIGNED_URL_ENDPOINT_ENV} must be an absolute http(s) URL"
        )
    return (
        endpoint.scheme.lower(),
        endpoint.hostname.lower(),
        _parsed_url_port(endpoint, config_name=PRESIGNED_URL_ENDPOINT_ENV),
    )


def _configured_allowed_presigned_hosts() -> list[str]:
    raw_hosts = os.environ.get(PRESIGNED_URL_ALLOWED_HOSTS_ENV, "")
    return [host.strip().lower() for host in raw_hosts.split(",") if host.strip()]


def _configured_allowed_presigned_http_origins() -> set[tuple[str, str, int]]:
    raw_origins = os.environ.get(PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV, "")
    origins: set[tuple[str, str, int]] = set()
    for raw_origin in raw_origins.split(","):
        value = raw_origin.strip()
        if not value:
            continue
        parsed = urllib.parse.urlsplit(value)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise EmbeddingBackendError(
                f"{PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV} entries must be exact http://host[:port] origins"
            )
        origins.add(
            (
                "http",
                parsed.hostname.lower().rstrip("."),
                _parsed_url_port(
                    parsed,
                    config_name=PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV,
                ),
            )
        )
    return origins


def _host_matches_pattern(host: str, pattern: str) -> bool:
    host = host.lower().rstrip(".")
    pattern = pattern.lower().rstrip(".")
    if pattern.startswith("*."):
        suffix = pattern[2:]
        return host.endswith(f".{suffix}") and host != suffix
    return host == pattern


def _reject_non_public_host(host: str, port: int | None) -> None:
    _resolve_presigned_host(host, port)


def _resolve_presigned_host(
    host: str, port: int | None, *, allow_private: bool = False
) -> list[Any]:
    try:
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise MediaDownloadError(
            "presigned video host could not be resolved"
        ) from error
    if not addresses:
        raise MediaDownloadError("presigned video host did not resolve")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not allow_private and not ip.is_global:
            raise EmbeddingInputError(
                "presigned video URL must not resolve to a private or reserved address"
            )
    return addresses


def _default_url_port(scheme: str) -> int:
    return 443 if scheme.lower() == "https" else 80


def _parsed_url_port(
    parsed: urllib.parse.SplitResult,
    *,
    config_name: str | None = None,
) -> int:
    try:
        port = parsed.port
    except ValueError as error:
        if config_name:
            raise EmbeddingBackendError(f"{config_name} has an invalid port") from error
        raise EmbeddingInputError("presigned video URL has an invalid port") from error
    return port or _default_url_port(parsed.scheme)


def _copy_response_with_limit(response: Any, output: Any, *, max_bytes: int) -> int:
    total_bytes = 0
    while True:
        chunk = response.read(min(1024 * 1024, max_bytes - total_bytes + 1))
        if not chunk:
            return total_bytes
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise EmbeddingInputError(f"remote media exceeds maximum {max_bytes} bytes")
        output.write(chunk)


def formulate_ce1_video_input(
    processor: Any,
    frames: Sequence[Any],
    *,
    target_num_frames: int,
) -> Any:
    """Prepare sampled frames for Cosmos-Embed1 video embedding."""

    if len(frames) < target_num_frames:
        raise EmbeddingBackendError(
            f"video has {len(frames)} frames; CE1 requires at least {target_num_frames}"
        )

    try:
        import numpy as np
    except ImportError as error:
        raise EmbeddingBackendError(
            "PyTorch CE1 video backend requires numpy for frame preparation"
        ) from error

    step = len(frames) // target_num_frames
    sampled_frames = list(frames[::step][:target_num_frames])
    try:
        video_batch = np.expand_dims(np.stack(sampled_frames), 0)
        video_batch = np.transpose(video_batch, (0, 1, 4, 2, 3))
        processed = processor(videos=video_batch, return_tensors="pt")
    except Exception as error:
        raise EmbeddingBackendError("failed to prepare CE1 video frames") from error

    try:
        return processed["videos"]
    except (KeyError, TypeError) as error:
        raise EmbeddingBackendError(
            "CE1 processor output missing videos tensor"
        ) from error


class PyTorchCosmosEmbedBackend:
    """Lazy PyTorch CE1 backend using explicit local or opted-in HF weights."""

    def __init__(
        self,
        config: BackendConfig,
        *,
        media_downloader: MediaDownloader | None = None,
    ) -> None:
        if not config.gpu_video:
            raise EmbeddingBackendError(
                "CE1 requires GPU video decoding; "
                "COSMOS_EMBED_GPU_VIDEO=false is not supported"
            )
        self._config = config
        self._variant = config.variant.lower()
        self._model_reference = _resolve_pytorch_model_reference(config, self._variant)
        self._media_downloader = media_downloader or download_presigned_video
        self._runtime: _TorchRuntime | None = None
        self._runtime_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._video_request_slots = threading.BoundedSemaphore(VIDEO_PIPELINE_DEPTH)
        self._preparation_pool = ThreadPoolExecutor(
            max_workers=config.preparation_workers,
            thread_name_prefix="ce1-prepare",
        )
        self.name = f"pytorch-ce1-{self._variant}"

    def close(self) -> None:
        """Release background preparation workers."""

        if self._preparation_pool is not None:
            self._preparation_pool.shutdown(wait=True)
            self._preparation_pool = None

    def ready(self) -> None:
        self._ensure_runtime()

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_type: str,
        input_kinds: Sequence[str],
    ) -> list[list[float]]:
        _validate_backend_batch(inputs, input_kinds)
        runtime = self._ensure_runtime()
        kinds = set(input_kinds)
        if kinds == {"text"}:
            return self._embed_text_batch(runtime, inputs)
        if kinds == {"video"}:
            if self._preparation_pool is None:
                return self._embed_video_batch(runtime, inputs)
            with self._video_request_slots:
                return self._embed_video_batch(runtime, inputs)
        unsupported_kinds = sorted(kinds - {"text", "video"})
        if unsupported_kinds:
            raise EmbeddingBackendError(
                f"unsupported input kind(s): {', '.join(unsupported_kinds)}"
            )
        raise EmbeddingBackendError(
            "mixed text and video backend batches are unsupported"
        )

    def _ensure_runtime(self) -> _TorchRuntime:
        if self._runtime is not None:
            return self._runtime

        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            try:
                runtime_modules = runtime_check.import_pytorch_runtime_modules(
                    module_names=(
                        "PyNvVideoCodec",
                        *runtime_check.REQUIRED_PYTORCH_MODULES,
                    ),
                )
            except runtime_check.RuntimeDependencyError as error:
                raise EmbeddingBackendError(str(error)) from error

            torch = runtime_modules["torch"]
            transformers = runtime_modules["transformers"]
            AutoModel = transformers.AutoModel
            AutoProcessor = transformers.AutoProcessor

            device = self._config.device or _default_torch_device(torch)
            if not str(device).startswith("cuda") or not torch.cuda.is_available():
                raise EmbeddingBackendError(
                    "CE1 requires a CUDA GPU with NVDEC video decoding support"
                )
            dtype = _default_torch_dtype(torch, device)
            # Resolve the pinned Hub snapshot (if explicitly enabled), then
            # verify local bytes BEFORE Transformers executes any custom code.
            verified_reference = _verified_model_reference(
                self._config, self._model_reference,
            )
            pretrained_kwargs = _from_pretrained_kwargs(
                self._config,
                verified_reference,
            )
            try:
                model = AutoModel.from_pretrained(
                    verified_reference.reference,
                    **pretrained_kwargs,
                )
                model = _move_to_device(model, device=device, dtype=dtype)
                if hasattr(model, "eval"):
                    model.eval()
                if self._config.sdpa:
                    _enable_sdpa_attention(model, torch)
                if self._config.torch_compile:
                    _compile_visual_encoder(model, torch)
                processor = AutoProcessor.from_pretrained(
                    verified_reference.reference,
                    **pretrained_kwargs,
                )
                gpu_media = self._create_gpu_media_processor(
                    torch=torch,
                    device=device,
                    dtype=dtype,
                )
                if self._config.torch_compile:
                    _warmup_visual_encoder(
                        model,
                        torch,
                        device=device,
                        dtype=dtype,
                        resolution=int(self._variant.removesuffix("p")),
                    )
            except EmbeddingBackendError:
                raise
            except Exception as error:
                raise EmbeddingBackendError("failed to load CE1 model") from error

            self._runtime = _TorchRuntime(
                model=model,
                processor=processor,
                torch=torch,
                device=device,
                dtype=dtype,
                gpu_media=gpu_media,
            )
            return self._runtime

    def _create_gpu_media_processor(
        self,
        *,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any | None:
        if not str(device).startswith("cuda"):
            raise EmbeddingBackendError("GPU media decoding requires a CUDA device")

        try:
            from src.cosmos_embed_oss.gpu_pipeline import GpuMediaProcessor

            return GpuMediaProcessor(
                torch=torch,
                device=device,
                dtype=dtype,
                resolution=int(self._variant.removesuffix("p")),
                frame_count=CE1_VIDEO_FRAME_COUNT,
            )
        except EmbeddingBackendError:
            raise
        except Exception as error:
            raise EmbeddingBackendError(
                "failed to initialize GPU media processing"
            ) from error

    def _embed_text_batch(
        self,
        runtime: _TorchRuntime,
        texts: Sequence[str],
    ) -> list[list[float]]:
        with self._inference_lock, _torch_no_grad(runtime.torch):
            batch = runtime.processor(text=list(texts), return_tensors="pt")
            batch = _move_batch_to_device(
                batch,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            output = runtime.model.get_text_embeddings(**batch)

        projection = getattr(output, "text_proj", None)
        if projection is None:
            raise EmbeddingBackendError("CE1 text embedding output missing text_proj")
        rows = _as_embedding_rows(projection)
        if len(rows) != len(texts):
            raise EmbeddingBackendError(
                f"CE1 text backend returned {len(rows)} embeddings for {len(texts)} inputs"
            )
        return [
            normalize_embedding(row, dimension=self._config.dimension) for row in rows
        ]

    def _embed_video_batch(
        self,
        runtime: _TorchRuntime,
        data_uris: Sequence[str],
    ) -> list[list[float]]:
        try:
            video_tensors = self._prepare_video_batch(runtime, data_uris)
            videos = _concat_video_tensors(runtime.torch, video_tensors)
            with self._inference_lock, _torch_no_grad(runtime.torch):
                videos = _move_to_device(
                    videos,
                    device=runtime.device,
                    dtype=runtime.dtype,
                )
                output = runtime.model.get_video_embeddings(videos=videos)
        except EmbeddingBackendError:
            raise
        except Exception as error:
            raise EmbeddingBackendError("failed to embed video payload") from error

        projection = getattr(output, "visual_proj", None)
        if projection is None:
            raise EmbeddingBackendError(
                "CE1 video embedding output missing visual_proj"
            )
        rows = _as_embedding_rows(projection)
        if len(rows) != len(data_uris):
            raise EmbeddingBackendError(
                f"CE1 video backend returned {len(rows)} embeddings for "
                f"{len(data_uris)} inputs"
            )
        return [
            normalize_embedding(row, dimension=self._config.dimension) for row in rows
        ]

    def _prepare_video_batch(
        self,
        runtime: _TorchRuntime,
        data_uris: Sequence[str],
    ) -> list[Any]:
        preparation_pool = self._preparation_pool
        if preparation_pool is None:
            return [self._prepare_video_input(runtime, value) for value in data_uris]

        futures: list[Future[Any]] = [
            preparation_pool.submit(self._prepare_video_input, runtime, value)
            for value in data_uris
        ]
        try:
            return [future.result() for future in futures]
        except Exception:
            for future in futures:
                future.cancel()
            wait(futures)
            raise

    def _prepare_video_input(self, runtime: _TorchRuntime, data_uri: str) -> Any:
        target_num_frames = _target_num_video_frames(runtime.processor)
        if data_uri.startswith("data:video_frames/"):
            if self._config.gpu_frames:
                frame_payloads = read_video_frames_data_uri(
                    data_uri,
                    tmp_dir=Path(self._config.media_tmp_dir),
                    download_timeout_seconds=(
                        self._config.media_download_timeout_seconds
                    ),
                    max_media_bytes=self._config.max_media_bytes,
                    media_downloader=self._media_downloader,
                )
                try:
                    return runtime.gpu_media.decode_frames(frame_payloads)
                except Exception as error:
                    raise EmbeddingInputError(
                        "video_frames payload contains an image the GPU decoder "
                        "does not support; use JPEG or PNG frames"
                    ) from error
            frames = decode_video_frames_data_uri(
                data_uri,
                tmp_dir=Path(self._config.media_tmp_dir),
                download_timeout_seconds=self._config.media_download_timeout_seconds,
                max_media_bytes=self._config.max_media_bytes,
                media_downloader=self._media_downloader,
            )
        else:
            with materialize_video_data_uri(
                data_uri,
                tmp_dir=Path(self._config.media_tmp_dir),
                download_timeout_seconds=self._config.media_download_timeout_seconds,
                max_media_bytes=self._config.max_media_bytes,
                media_downloader=self._media_downloader,
            ) as video:
                try:
                    return runtime.gpu_media.decode_video(video.path)
                except GpuDecodeUnsupported as error:
                    raise EmbeddingInputError(str(error)) from error
                except Exception as error:
                    raise EmbeddingInputError(
                        "video payload could not be decoded by the GPU video "
                        "decoder; re-encode to an NVDEC-supported codec such "
                        "as H.264, HEVC, VP9, or AV1"
                    ) from error

        videos = formulate_ce1_video_input(
            runtime.processor,
            frames,
            target_num_frames=target_num_frames,
        )
        return _as_torch_tensor(runtime.torch, videos)


def _as_flat_float_vector(embedding: Any) -> list[float]:
    if hasattr(embedding, "detach"):
        embedding = embedding.detach()
    if hasattr(embedding, "to"):
        try:
            embedding = embedding.to("cpu")
        except TypeError:
            embedding = embedding.to(device="cpu")
    if hasattr(embedding, "float"):
        embedding = embedding.float()
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if _looks_like_nested_vector(embedding):
        embedding = embedding[0]

    if isinstance(embedding, (bytes, str)) or not isinstance(embedding, Sequence):
        raise EmbeddingBackendError("CE1 backend returned a non-vector embedding")

    try:
        return [float(component) for component in embedding]
    except (TypeError, ValueError) as error:
        raise EmbeddingBackendError(
            "CE1 backend returned a non-numeric embedding"
        ) from error


def _as_embedding_rows(embeddings: Any) -> list[Any]:
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach()
    if hasattr(embeddings, "to"):
        try:
            embeddings = embeddings.to("cpu")
        except TypeError:
            embeddings = embeddings.to(device="cpu")
    if hasattr(embeddings, "float"):
        embeddings = embeddings.float()
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if isinstance(embeddings, (bytes, str)) or not isinstance(embeddings, Sequence):
        raise EmbeddingBackendError("CE1 backend returned non-matrix embeddings")
    if not embeddings:
        return []
    if _looks_like_nested_vector(embeddings):
        return list(embeddings)
    return [embeddings]


def _backend_env_fingerprint() -> tuple[str | None, ...]:
    return (
        os.environ.get("COSMOS_EMBED_BACKEND"),
        os.environ.get("COSMOS_EMBED_MODEL_PATH"),
        os.environ.get("COSMOS_EMBED_WEIGHTS_DIR"),
        os.environ.get("COSMOS_EMBED_MODEL_VARIANT"),
        os.environ.get("COSMOS_EMBED_DIM"),
        os.environ.get("COSMOS_EMBED_DEVICE"),
        os.environ.get("COSMOS_EMBED_TRUST_REMOTE_CODE"),
        os.environ.get("COSMOS_EMBED_LOCAL_FILES_ONLY"),
        os.environ.get("COSMOS_EMBED_MAX_MEDIA_BYTES"),
        os.environ.get("COSMOS_EMBED_ALLOW_HF_DOWNLOAD"),
        os.environ.get("COSMOS_EMBED_HF_MODEL_ID"),
        os.environ.get("COSMOS_EMBED_HF_REVISION"),
        os.environ.get("HF_HOME"),
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        _secret_presence_fingerprint(HF_TOKEN_ENV_NAMES),
        os.environ.get("COSMOS_EMBED_TMP_DIR"),
        os.environ.get("COSMOS_EMBED_MEDIA_DOWNLOAD_TIMEOUT_SECONDS"),
        os.environ.get("COSMOS_EMBED_SDPA"),
        os.environ.get("COSMOS_EMBED_TORCH_COMPILE"),
        os.environ.get("COSMOS_EMBED_GPU_FRAMES"),
        os.environ.get("COSMOS_EMBED_GPU_VIDEO"),
        os.environ.get("COSMOS_EMBED_PREPARATION_WORKERS"),
        os.environ.get(PRESIGNED_URL_ALLOWED_HOSTS_ENV),
        os.environ.get(PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV),
        os.environ.get(PRESIGNED_URL_ENDPOINT_ENV),
    )


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise EmbeddingBackendError(f"{name} must be a boolean")


def _default_torch_device(torch: Any) -> str:
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available():
            return "cuda"
    return "cpu"


def _default_torch_dtype(torch: Any, device: str) -> Any:
    if device.split(":", 1)[0].lower() == "cuda":
        cuda = getattr(torch, "cuda", None)
        is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
        if hasattr(torch, "bfloat16") and callable(is_bf16_supported):
            try:
                if is_bf16_supported():
                    return torch.bfloat16
            except RuntimeError:
                pass
        if hasattr(torch, "float16"):
            return torch.float16
    return getattr(torch, "float32", None)


def _looks_like_nested_vector(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)) or not value:
        return False
    first = value[0]
    return isinstance(first, Sequence) and not isinstance(first, (bytes, str))


def _move_batch_to_device(batch: Any, *, device: str, dtype: Any) -> Any:
    if isinstance(batch, Mapping):
        return {
            key: _move_to_device(value, device=device, dtype=dtype)
            for key, value in batch.items()
        }
    moved = _move_to_device(batch, device=device, dtype=dtype)
    if moved is not batch:
        return moved
    return batch


def _can_cast_tensor_dtype(value: Any) -> bool:
    is_floating_point = getattr(value, "is_floating_point", None)
    if callable(is_floating_point):
        return bool(is_floating_point())
    return True


def _move_to_device(value: Any, *, device: str, dtype: Any) -> Any:
    to_method = getattr(value, "to", None)
    if to_method is None:
        return value
    try:
        if dtype is not None and _can_cast_tensor_dtype(value):
            return to_method(device, dtype=dtype)
        return to_method(device)
    except TypeError:
        try:
            return to_method(device)
        except TypeError:
            return to_method(device=device)


def _as_torch_tensor(torch: Any, value: Any) -> Any:
    if hasattr(value, "to"):
        return value
    from_numpy = getattr(torch, "from_numpy", None)
    if from_numpy is None:
        raise EmbeddingBackendError("CE1 video input is not a torch tensor")
    try:
        return from_numpy(value)
    except Exception as error:
        raise EmbeddingBackendError(
            "failed to convert CE1 video input to tensor"
        ) from error


def _concat_video_tensors(torch: Any, tensors: Sequence[Any]) -> Any:
    if not tensors:
        raise EmbeddingBackendError("video batch must not be empty")
    cat = getattr(torch, "cat", None)
    if cat is None:
        if len(tensors) == 1:
            return tensors[0]
        raise EmbeddingBackendError("PyTorch runtime does not provide torch.cat")
    try:
        return cat(list(tensors), dim=0)
    except Exception as error:
        raise EmbeddingBackendError("failed to batch CE1 video inputs") from error


def _linspace_frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        raise EmbeddingBackendError("video payload did not contain frames")
    if count <= 1:
        return [0]
    scale = (total_frames - 1) / (count - 1)
    return [min(total_frames - 1, int(index * scale)) for index in range(count)]


def _media_type_from_data_uri_header(header: str) -> str:
    if not header.startswith("data:"):
        raise EmbeddingInputError("video input must be a data URI")
    media_type = header[len("data:") :].lower()
    if not media_type.startswith("video/"):
        raise EmbeddingInputError("video input must use a video media type")
    return media_type


def _normalize_backend_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in PYTORCH_BACKEND_ALIASES:
        return "pytorch"
    raise EmbeddingBackendError("COSMOS_EMBED_BACKEND must select the PyTorch backend")


def _positive_int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise EmbeddingBackendError(f"{name} must be an integer") from error
    if parsed < 1:
        raise EmbeddingBackendError(f"{name} must be positive")
    return parsed


def _positive_float_from_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise EmbeddingBackendError(f"{name} must be a number") from error
    if parsed <= 0.0:
        raise EmbeddingBackendError(f"{name} must be positive")
    return parsed


def _torch_no_grad(torch: Any) -> Any:
    no_grad = getattr(torch, "no_grad", None)
    if no_grad is None:
        return nullcontext()
    return no_grad()


def _enable_sdpa_attention(model: Any, torch: Any) -> None:
    """Use PyTorch SDPA for the CE1 visual encoder without changing semantics."""

    visual_encoder = getattr(model, "visual_encoder", None)
    blocks = getattr(visual_encoder, "blocks", None)
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    if (
        not blocks
        or functional is None
        or not hasattr(functional, "scaled_dot_product_attention")
    ):
        raise EmbeddingBackendError(
            "COSMOS_EMBED_SDPA=true requires the CE1 visual encoder and PyTorch SDPA"
        )

    for block in blocks:
        attention = getattr(block, "attn", None)
        required = (
            "qkv",
            "num_heads",
            "scale",
            "relative_position_bias_table",
            "relative_position_index",
            "window_size",
            "attn_drop",
            "proj",
            "proj_drop",
        )
        if (
            attention is None
            or any(not hasattr(attention, attribute) for attribute in required)
            or hasattr(attention, "rope")
            or (getattr(attention, "q_bias", None) is None)
            != (getattr(attention, "v_bias", None) is None)
        ):
            raise EmbeddingBackendError(
                "COSMOS_EMBED_SDPA=true found an unsupported CE1 attention module"
            )
        attention.forward = types.MethodType(_sdpa_attention_forward, attention)


def _sdpa_attention_forward(self: Any, x: Any, rel_pos_bias: Any | None = None) -> Any:
    """Equivalent CE1 EVA-ViT attention implemented with PyTorch SDPA."""

    import torch
    import torch.nn.functional as functional

    batch_size, token_count, _ = x.shape
    qkv_bias = getattr(self.qkv, "bias", None)
    if getattr(self, "q_bias", None) is not None:
        qkv_bias = torch.cat(
            (
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            )
        )
    query, key, value = (
        functional.linear(x, self.qkv.weight, qkv_bias)
        .reshape(batch_size, token_count, 3, self.num_heads, -1)
        .permute(2, 0, 3, 1, 4)
    )

    attention_bias = rel_pos_bias
    if self.relative_position_bias_table is not None:
        window_height, window_width = self.window_size
        window_tokens = window_height * window_width + 1
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(window_tokens, window_tokens, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attention_bias = (
            relative_bias if attention_bias is None else attention_bias + relative_bias
        )

    output = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_bias,
        dropout_p=self.attn_drop.p if self.training else 0.0,
        scale=self.scale,
    )
    output = output.transpose(1, 2).reshape(batch_size, token_count, -1)
    return self.proj_drop(self.proj(output))


def _compile_visual_encoder(model: Any, torch: Any) -> None:
    """Compile the dominant CE1 visual encoder while retaining the model API."""

    visual_encoder = getattr(model, "visual_encoder", None)
    compile_function = getattr(torch, "compile", None)
    if visual_encoder is None or not callable(compile_function):
        raise EmbeddingBackendError(
            "COSMOS_EMBED_TORCH_COMPILE=true requires torch.compile and the CE1 visual encoder"
        )
    model.visual_encoder = compile_function(
        visual_encoder,
        mode=TORCH_COMPILE_MODE,
    )


def _warmup_visual_encoder(
    model: Any,
    torch: Any,
    *,
    device: Any,
    dtype: Any,
    resolution: int,
) -> None:
    """Force torch.compile to build the visual encoder before readiness.

    ``torch.compile`` defers compilation to the first invocation, so without
    this the first customer request absorbs the full compile cost and can
    exceed the client timeout.
    """

    videos = torch.zeros(
        (1, CE1_VIDEO_FRAME_COUNT, 3, resolution, resolution),
        dtype=dtype,
        device=device,
    )
    try:
        with _torch_no_grad(torch):
            model.get_video_embeddings(videos=videos)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as error:
        raise EmbeddingBackendError(
            "failed to warm up the compiled CE1 visual encoder"
        ) from error


def _validate_backend_batch(inputs: Sequence[str], input_kinds: Sequence[str]) -> None:
    if len(inputs) != len(input_kinds):
        raise EmbeddingBackendError("backend input count does not match input kinds")


def _cached_hf_model_dir(model_id: str) -> Path | None:
    cache_name = f"models--{model_id.replace('/', '--')}"
    for cache_root in _huggingface_cache_roots():
        model_dir = cache_root / cache_name
        if (model_dir / "snapshots").is_dir():
            return model_dir
    return None


def _verified_model_reference(
    config: BackendConfig,
    model_reference: _ResolvedPytorchModel,
) -> _ResolvedPytorchModel:
    try:
        reference = model_reference.reference
        if model_reference.source == "huggingface":
            from huggingface_hub import snapshot_download

            model_integrity.validate_model_source(config.hf_model_id, config.hf_revision)
            reference = snapshot_download(
                repo_id=config.hf_model_id,
                revision=config.hf_revision,
                local_files_only=model_reference.local_files_only,
                token=model_reference.use_auth_token,
            )
        verified = model_integrity.verify_model_snapshot(reference)
    except (model_integrity.ModelIntegrityError, OSError) as error:
        raise EmbeddingBackendError(
            f"CE1 model integrity verification failed: {error}"
        ) from error
    except Exception as error:
        raise EmbeddingBackendError("failed to obtain the approved CE1 snapshot") from error
    # Both loaders use the SAME verified local snapshot. No remote model/code
    # resolution is allowed after verification, even in the opt-in Hub mode.
    return _ResolvedPytorchModel(str(verified), source="local", local_files_only=True)


def _from_pretrained_kwargs(
    config: BackendConfig,
    model_reference: _ResolvedPytorchModel,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": model_reference.local_files_only,
    }
    if model_reference.use_auth_token:
        kwargs["token"] = True
    return kwargs


def _has_hf_auth_token() -> bool:
    return any(os.environ.get(name) for name in HF_TOKEN_ENV_NAMES)


def _huggingface_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    deduped_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        expanded = root.expanduser()
        if expanded not in seen:
            deduped_roots.append(expanded)
            seen.add(expanded)
    return deduped_roots


def _resolve_pytorch_model_reference(
    config: BackendConfig,
    variant: str,
) -> _ResolvedPytorchModel:
    if config.dimension != cosmos_embed_oss.DEFAULT_EMBEDDING_DIMENSION:
        raise EmbeddingBackendError(
            "PyTorch CE1 backend currently supports only 256-d embeddings"
        )
    if variant not in PYTORCH_VARIANT_TO_MODEL:
        supported = ", ".join(sorted(PYTORCH_VARIANT_TO_MODEL))
        raise EmbeddingBackendError(
            f"COSMOS_EMBED_MODEL_VARIANT must be one of: {supported}"
        )
    if variant != DEFAULT_PYTORCH_VARIANT:
        raise EmbeddingBackendError(
            "CVDS CE1 compatibility requires Cosmos-Embed1-224p 256-d weights"
        )

    if config.model_path:
        model_path = Path(config.model_path)
        if not model_path.is_dir():
            raise EmbeddingBackendError(
                "configured CE1 model path must be an existing directory"
            )
        return _ResolvedPytorchModel(
            reference=str(model_path),
            source="local",
            local_files_only=True,
        )

    if not config.allow_hf_download:
        raise EmbeddingBackendError(
            "COSMOS_EMBED_MODEL_PATH or COSMOS_EMBED_WEIGHTS_DIR is required "
            "when COSMOS_EMBED_BACKEND=pytorch. Set "
            "COSMOS_EMBED_ALLOW_HF_DOWNLOAD=true only for explicit Hugging Face "
            "model loading."
        )

    hf_model_id = config.hf_model_id.strip()
    if not hf_model_id:
        raise EmbeddingBackendError(
            "COSMOS_EMBED_HF_MODEL_ID must not be empty when "
            "COSMOS_EMBED_ALLOW_HF_DOWNLOAD=true"
        )
    try:
        model_integrity.validate_model_source(hf_model_id, config.hf_revision)
    except model_integrity.ModelIntegrityError as error:
        raise EmbeddingBackendError(str(error)) from error
    has_hf_auth_token = _has_hf_auth_token()
    cached_model_dir = _cached_hf_model_dir(hf_model_id)
    if not has_hf_auth_token and cached_model_dir is None:
        raise EmbeddingBackendError(
            "COSMOS_EMBED_ALLOW_HF_DOWNLOAD=true requires HF_TOKEN or "
            "HUGGING_FACE_HUB_TOKEN, or cached Hugging Face model files for "
            f"{hf_model_id}."
        )

    return _ResolvedPytorchModel(
        reference=hf_model_id,
        source="huggingface",
        local_files_only=not has_hf_auth_token,
        use_auth_token=has_hf_auth_token,
    )


def _secret_presence_fingerprint(env_names: Sequence[str]) -> str:
    return ",".join(name for name in env_names if os.environ.get(name))


def _target_num_video_frames(processor: Any) -> int:
    raw_value = getattr(processor, "num_video_frames", None)
    if raw_value is None:
        return CE1_VIDEO_FRAME_COUNT
    try:
        target_num_frames = int(raw_value)
    except (TypeError, ValueError) as error:
        raise EmbeddingBackendError(
            "CE1 processor must expose num_video_frames for video embedding"
        ) from error
    if target_num_frames != CE1_VIDEO_FRAME_COUNT:
        raise EmbeddingBackendError(
            f"CE1 processor must require exactly {CE1_VIDEO_FRAME_COUNT} video frames"
        )
    return target_num_frames


def _video_suffix_for_media_type(media_type: str) -> str:
    return VIDEO_EXTENSION_BY_MEDIA_TYPE.get(media_type.lower(), ".mp4")


def _image_suffix_for_format(frame_format: str) -> str:
    normalized = frame_format.lower()
    if normalized in {"jpeg", "jpg"}:
        return ".jpg"
    if normalized in {"png", "webp", "bmp"}:
        return f".{normalized}"
    return ".img"


def _decode_base64_payload(
    payload: str,
    *,
    max_bytes: int,
    payload_name: str,
) -> bytes:
    max_encoded_bytes = 4 * ((max_bytes + 2) // 3)
    if len(payload) > max_encoded_bytes:
        raise EmbeddingInputError(f"{payload_name} exceeds maximum {max_bytes} bytes")
    try:
        media_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise EmbeddingInputError(f"invalid base64 {payload_name} payload") from error
    if not media_bytes:
        raise EmbeddingInputError(f"base64 {payload_name} payload must not be empty")
    if len(media_bytes) > max_bytes:
        raise EmbeddingInputError(f"{payload_name} exceeds maximum {max_bytes} bytes")
    return media_bytes


def _write_base64_payload(
    payload: str,
    destination_path: Path,
    *,
    max_bytes: int,
    payload_name: str,
) -> None:
    media_bytes = _decode_base64_payload(
        payload,
        max_bytes=max_bytes,
        payload_name=payload_name,
    )
    destination_path.write_bytes(media_bytes)
