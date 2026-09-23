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

"""Contract tests for the CVDS-owned Cosmos-Embed OSS service."""

from __future__ import annotations

import base64
import http.client
import math
import struct
import threading
from collections.abc import Iterator, Sequence
from unittest import mock

import pytest
import requests

from src import cosmos_embed_oss as ce1_defaults
from src.cosmos_embed_oss import backends
from src.cosmos_embed_oss.server import run_server
from src.haystack.components.video import cosmos_video_embedder

EMBEDDING_DIMENSION = ce1_defaults.DEFAULT_EMBEDDING_DIMENSION
MODEL_NAME = ce1_defaults.DEFAULT_MODEL_NAME


class _ContractTestBackend:
    name = "contract-test"

    def ready(self) -> None:
        return None

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_type: str,
        input_kinds: Sequence[str],
    ) -> list[list[float]]:
        del request_type, input_kinds
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in inputs]


@pytest.fixture(scope="module")
def ce1_service_url() -> Iterator[str]:
    """Run the service with a test-local backend on an ephemeral port."""

    with mock.patch.object(
        backends,
        "get_embedding_backend",
        return_value=_ContractTestBackend(),
    ):
        server = run_server(host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        host, port = server.server_address
        try:
            yield f"http://{host}:{port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_health_and_models_endpoints(ce1_service_url: str) -> None:
    response = requests.get(f"{ce1_service_url}/v1/health/live", timeout=5)
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

    response = requests.get(f"{ce1_service_url}/v1/health/ready", timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL_NAME
    assert body["embedding_dim"] == EMBEDDING_DIMENSION
    assert body["status"] == "ready"

    response = requests.get(f"{ce1_service_url}/v1/models", timeout=5)
    assert response.status_code == 200
    models = response.json()["data"]
    assert models == [
        {
            "id": MODEL_NAME,
            "created": 1686935002,
            "object": "model",
            "owned_by": "nvidia",
        }
    ]


def test_metadata_license_manifest_and_metrics_compatibility_endpoints(
    ce1_service_url: str,
) -> None:
    metadata = requests.get(f"{ce1_service_url}/v1/metadata", timeout=5)
    assert metadata.status_code == 200
    metadata_body = metadata.json()
    assert metadata_body["version"] == "1.2.0"
    assert metadata_body["modelInfo"][0]["shortName"] == MODEL_NAME
    assert metadata_body["runtime"] == "contract-test"
    assert metadata_body["licenseInfo"]["size"] > 0

    license_response = requests.get(f"{ce1_service_url}/v1/license", timeout=5)
    assert license_response.status_code == 200
    license_body = license_response.json()
    assert license_body["type"] == "text/plain"
    assert len(license_body["sha"]) == 64
    assert license_body["size"] == len(license_body["content"].encode("utf-8"))

    manifest = requests.get(f"{ce1_service_url}/v1/manifest", timeout=5)
    assert manifest.status_code == 200
    assert "schema_version:" in manifest.json()["manifest_file"]
    assert 'release: "1.2.0"' in manifest.json()["manifest_file"]

    metrics_alias = requests.get(f"{ce1_service_url}/health/metrics", timeout=5)
    assert metrics_alias.status_code == 200
    assert "business_metrics" in metrics_alias.json()


def test_legacy_text_list_query_returns_ordered_256d_embeddings(
    ce1_service_url: str,
) -> None:
    payload = {
        "input": ["left turn", "pedestrian crossing"],
        "request_type": "query",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }

    response = requests.post(
        f"{ce1_service_url}/v1/embeddings", json=payload, timeout=5
    )
    assert response.status_code == 200

    body = response.json()
    assert body["model"] == MODEL_NAME
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert [len(item["embedding"]) for item in body["data"]] == [
        EMBEDDING_DIMENSION,
        EMBEDDING_DIMENSION,
    ]
    assert body["usage"]["num_videos"] == 0

    second_response = requests.post(
        f"{ce1_service_url}/v1/embeddings", json=payload, timeout=5
    )
    assert second_response.json()["data"] == body["data"]


def test_base64_output_encodes_256_little_endian_float32_values(
    ce1_service_url: str,
) -> None:
    response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json={
            "input": "left turn",
            "request_type": "query",
            "encoding_format": "base64",
            "model": MODEL_NAME,
        },
        timeout=5,
    )

    assert response.status_code == 200
    encoded_embedding = response.json()["data"][0]["embedding"]
    raw_embedding = base64.b64decode(encoded_embedding, validate=True)
    values = struct.unpack(f"<{EMBEDDING_DIMENSION}f", raw_embedding)
    assert len(raw_embedding) == EMBEDDING_DIMENSION * 4
    assert math.isclose(sum(value * value for value in values), 1.0, abs_tol=1e-6)


def test_base64_video_query_shape(ce1_service_url: str) -> None:
    payload = {
        "input": "data:video/mp4;base64,ZmFrZV92aWRlbw==",
        "request_type": "query",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }

    response = requests.post(
        f"{ce1_service_url}/v1/embeddings", json=payload, timeout=5
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == EMBEDDING_DIMENSION
    assert body["usage"]["num_videos"] == 1


def test_bulk_video_presigned_url_shape_is_all_or_non_2xx(ce1_service_url: str) -> None:
    payload = {
        "input": [
            "data:video/mp4;presigned_url,https://example.com/one.mp4",
            "data:video/mp4;presigned_url,https://example.com/two.mp4",
        ],
        "request_type": "bulk_video",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }

    response = requests.post(
        f"{ce1_service_url}/v1/embeddings", json=payload, timeout=5
    )
    assert response.status_code == 200
    body = response.json()
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert all(len(item["embedding"]) == EMBEDDING_DIMENSION for item in body["data"])
    assert body["usage"]["num_videos"] == 2

    invalid_payload = dict(payload)
    invalid_payload["input"] = payload["input"] + ["plain text"]
    error_response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json=invalid_payload,
        timeout=5,
    )
    assert error_response.status_code == 422
    assert "data" not in error_response.json()


def test_invalid_mixed_payload_and_batch_errors_return_non_2xx(
    ce1_service_url: str,
) -> None:
    mixed_payload = {
        "input": [
            "data:video/mp4;base64,ZmFrZQ==",
            "data:video/mp4;presigned_url,https://example.com/two.mp4",
        ],
        "request_type": "bulk_video",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }
    mixed_response = requests.post(
        f"{ce1_service_url}/v1/embeddings", json=mixed_payload, timeout=5
    )
    assert mixed_response.status_code == 422
    assert (
        "Cannot mix base64 and presigned URL"
        in mixed_response.json()["error"]["detail"]
    )

    too_many_payload = {
        "input": [f"text {index}" for index in range(65)],
        "request_type": "query",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }
    too_many_response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json=too_many_payload,
        timeout=5,
    )
    assert too_many_response.status_code == 422
    assert "maximum 64 inputs" in too_many_response.json()["error"]["detail"]


def test_bulk_video_and_video_frames_validation_matches_nim_contract(
    ce1_service_url: str,
) -> None:
    full_video_base64 = {
        "input": ["data:video/mp4;base64,ZmFrZQ=="],
        "request_type": "bulk_video",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }
    response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json=full_video_base64,
        timeout=5,
    )
    assert response.status_code == 422
    assert (
        "full-video inputs must use ;presigned_url,"
        in response.json()["error"]["detail"]
    )

    malformed_frames = {
        "input": "data:video_frames/png;base64,{ZmFrZQ==,ZmFrZQ==}",
        "request_type": "query",
        "encoding_format": "float",
        "model": MODEL_NAME,
    }
    response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json=malformed_frames,
        timeout=5,
    )
    assert response.status_code == 422
    assert "exactly 8 frames" in response.json()["error"]["detail"]

    valid_frames = dict(malformed_frames)
    valid_frames["input"] = (
        "data:video_frames/png;base64,{" + ",".join(["ZmFrZQ=="] * 8) + "}"
    )
    response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json=valid_frames,
        timeout=5,
    )
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == EMBEDDING_DIMENSION


def test_existing_cvds_client_can_call_service_without_payload_changes(
    ce1_service_url: str,
) -> None:
    client = cosmos_video_embedder.CosmosEmbedClient(ce1_service_url)

    text_embeddings = client.embed_texts(["left turn", "pedestrian crossing"])
    assert len(text_embeddings) == 2
    assert all(len(embedding) == EMBEDDING_DIMENSION for embedding in text_embeddings)

    base64_embeddings = client.embed_videos(["data:video/mp4;base64,ZmFrZV92aWRlbw=="])
    assert len(base64_embeddings) == 1
    assert len(base64_embeddings[0]) == EMBEDDING_DIMENSION

    presigned_embeddings = client.embed_videos(
        [
            "data:video/mp4;presigned_url,https://example.com/one.mp4",
            "data:video/mp4;presigned_url,https://example.com/two.mp4",
        ]
    )
    assert len(presigned_embeddings) == 2
    assert all(
        len(embedding) == EMBEDDING_DIMENSION for embedding in presigned_embeddings
    )


def test_metrics_endpoints_report_completed_embedding_requests(
    ce1_service_url: str,
) -> None:
    embedding_response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json={
            "input": "left turn",
            "request_type": "query",
            "encoding_format": "float",
            "model": MODEL_NAME,
        },
        timeout=5,
    )
    assert embedding_response.status_code == 200

    health_metrics = requests.get(
        f"{ce1_service_url}/v1/health/metrics",
        timeout=5,
    )
    assert health_metrics.status_code == 200
    snapshot = health_metrics.json()
    assert snapshot["requests"]["total"] >= 1
    assert snapshot["business_metrics"]["total_embeddings"] >= 1

    prometheus = requests.get(f"{ce1_service_url}/v1/metrics", timeout=5)
    assert prometheus.status_code == 200
    assert "cosmos_embed_requests_total" in prometheus.text
    assert "cosmos_embed_embeddings_total" in prometheus.text


def test_invalid_content_length_returns_bad_request(ce1_service_url: str) -> None:
    host_port = ce1_service_url.removeprefix("http://")
    host, raw_port = host_port.rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(raw_port), timeout=5)
    connection.putrequest("POST", "/v1/embeddings")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", "not-an-integer")
    connection.endheaders()

    response = connection.getresponse()
    body = response.read()
    connection.close()

    assert response.status == 400
    assert b"Content-Length header must be an integer" in body


def test_request_size_limit_is_enforced(
    ce1_service_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSMOS_EMBED_MAX_REQUEST_BYTES", "16")
    response = requests.post(
        f"{ce1_service_url}/v1/embeddings",
        json={
            "input": ["too large"],
            "request_type": "query",
            "model": MODEL_NAME,
        },
        timeout=5,
    )

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"
    assert "request body exceeds maximum 16 bytes" in response.json()["error"]["detail"]
