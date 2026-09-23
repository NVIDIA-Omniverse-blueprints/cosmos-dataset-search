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

# Copyright (C) 2024, NVIDIA CORPORATION.

import json
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from kubernetes.client.rest import ApiException

from src.visual_search.common.exceptions import SecretsNotFoundError
from src.visual_search.common.models import (
    Collection,
    RetrievedDocument,
    SearchResponse,
)
from src.visual_search.tests.conftest import (
    HaystackResult,
    RadiusSearchJson,
    TopKSearchJson,
    client,
    good_retrieval_request_body,
)
from src.visual_search.v1.apis.search import get_custom_aws_credentials, retrieval


# Test TopKSearch
def test_top_k_search():
    response = client.post(
        "/retrieval", json=good_retrieval_request_body(TopKSearchJson())
    )
    assert response.status_code == 200


# Test RadiusSearch
def test_radius_search():
    response = client.post(
        "/retrieval", json=good_retrieval_request_body(RadiusSearchJson())
    )
    assert response.status_code == 422


# Test mis-configured query params:
def test_read_retrieval_empty():
    response = client.post("/retrieval")
    assert response.status_code == 422


def test_read_retrieval_partial():
    response = client.post("/retrieval", json={"pipeline": "test"})
    assert response.status_code == 422


# Test successful
def test_read_retrieval_full():
    with patch(
        "src.visual_search.v1.apis.search.search",
        return_value=SearchResponse(retrievals=[]),
    ):
        response = client.post(
            "/retrieval",
            json={
                "collections": ["a", "b"],
                "query": [
                    {
                        "text": "a car",
                    },
                    {"text": "a house"},
                ],
                "params": {
                    "nb_neighbors": 1,
                    "filters": {
                        "field": "session_id",
                        "operator": "in",
                        "value": ["a", "b"],
                    },
                    "search_params": {"nprobe": 32},
                    "reconstruct": True,
                },
                "payload_keys": ["score"],
                "generate_asset_url": True,
                "rerank": True,
                "num_results": 2,
            },
        )
        assert response.status_code == 200
        assert response.json() == {"retrievals": []}


# Test retrieval function
@pytest.mark.asyncio
async def test_retrieval(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result = RetrievedDocument(
        id="test_id",
        collection_id="test_coll",
        content="test_content",
        mime_type="text/plain",
        metadata={},
        asset_url=None,
    )
    mock_search_response = SearchResponse(retrievals=[mock_result, mock_result])

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        # Call the retrieval function
        response = await retrieval(retrieval_query=mock_retrieval_query)

        assert len(response.retrievals) > 0


@pytest.mark.asyncio
async def test_rerank_default(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result1 = RetrievedDocument(
        id="test_id1",
        collection_id="test_coll",
        content="test_content1",
        mime_type="text/plain",
        metadata={},
        score=0.5,
        asset_url=None,
    )
    mock_result2 = RetrievedDocument(
        id="test_id2",
        collection_id="test_coll",
        content="test_content2",
        mime_type="text/plain",
        metadata={},
        score=0.9,
        asset_url=None,
    )
    mock_search_response = SearchResponse(retrievals=[mock_result1, mock_result2])

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        scores = [doc["score"] for doc in result["retrievals"]]
        assert scores == sorted(scores, reverse=True)


# For the next two sort comparisons
def sort_key(item):
    return (item is not None, item)


@pytest.mark.asyncio
async def test_rerank_self_missing_score(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result1 = RetrievedDocument(
        id="test_id1",
        collection_id="test_coll",
        content="test_content1",
        mime_type="text/plain",
        metadata={},
        score=None,
        asset_url=None,
    )
    mock_result2 = RetrievedDocument(
        id="test_id2",
        collection_id="test_coll",
        content="test_content2",
        mime_type="text/plain",
        metadata={},
        score=0.9,
        asset_url=None,
    )
    mock_search_response = SearchResponse(retrievals=[mock_result1, mock_result2])

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        scores = [doc["score"] for doc in result["retrievals"]]
        assert scores == sorted(scores, key=sort_key, reverse=True)


@pytest.mark.asyncio
async def test_rerank_other_missing_score(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result1 = RetrievedDocument(
        id="test_id1",
        collection_id="test_coll",
        content="test_content1",
        mime_type="text/plain",
        metadata={},
        score=0.9,
        asset_url=None,
    )
    mock_result2 = RetrievedDocument(
        id="test_id2",
        collection_id="test_coll",
        content="test_content2",
        mime_type="text/plain",
        metadata={},
        score=None,
        asset_url=None,
    )
    mock_search_response = SearchResponse(retrievals=[mock_result1, mock_result2])

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        scores = [doc["score"] for doc in result["retrievals"]]
        assert scores == sorted(scores, key=sort_key, reverse=True)


@pytest.mark.asyncio
async def test_payload_keys_must_be_valid(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result = HaystackResult()
    mock_result.meta = {"valid_key": "value"}  # No 'invalid_key'
    mock_run_query_pipeline.return_value = [mock_result]
    mock_retrieval_query.payload_keys = ["invalid_key"]
    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=SearchResponse(
                retrievals=[
                    RetrievedDocument(
                        id=mock_result.id,
                        collection_id="test_coll",
                        content=mock_result.content,
                        mime_type=mock_result.mime_type,
                        metadata=mock_result.meta,
                        score=mock_result.score,
                        asset_url=None,
                    )
                ]
            ),
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        with pytest.raises(HTTPException):
            await retrieval(retrieval_query=mock_retrieval_query)


@pytest.mark.asyncio
async def test_payload_key_only_is_returned(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result = HaystackResult()
    mock_run_query_pipeline.return_value = [mock_result]
    mock_retrieval_query.payload_keys = ["d"]
    mock_search_response = SearchResponse(
        retrievals=[
            RetrievedDocument(
                id=mock_result.id,
                collection_id="test_coll",
                content=mock_result.content,
                mime_type=mock_result.mime_type,
                metadata=mock_result.meta,
                score=mock_result.score,
                asset_url=None,
            )
        ]
    )

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        assert "d" in result["retrievals"][0]["metadata"]
        assert "e" not in result["retrievals"][0]["metadata"]


@pytest.mark.asyncio
async def test_two_payload_keys_are_returned(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result = HaystackResult()
    mock_run_query_pipeline.return_value = [mock_result]
    mock_retrieval_query.payload_keys = ["d", "e"]
    mock_search_response = SearchResponse(
        retrievals=[
            RetrievedDocument(
                id=mock_result.id,
                collection_id="test_coll",
                content=mock_result.content,
                mime_type=mock_result.mime_type,
                metadata=mock_result.meta,
                asset_url=None,
                score=mock_result.score,
            )
        ]
    )

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        assert "d" in result["retrievals"][0]["metadata"]
        assert "e" in result["retrievals"][0]["metadata"]
        assert "f" not in result["retrievals"][0]["metadata"]


@pytest.mark.asyncio
async def test_all_payload_keys_are_returned(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result = HaystackResult()
    mock_run_query_pipeline.return_value = [mock_result]
    mock_retrieval_query.payload_keys = None
    mock_search_response = SearchResponse(
        retrievals=[
            RetrievedDocument(
                id=mock_result.id,
                collection_id="test_coll",
                content=mock_result.content,
                mime_type=mock_result.mime_type,
                metadata=mock_result.meta,
                asset_url=None,
                score=mock_result.score,
            )
        ]
    )

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        assert "d" in result["retrievals"][0]["metadata"]
        assert "e" in result["retrievals"][0]["metadata"]
        assert "f" in result["retrievals"][0]["metadata"]


@pytest.mark.asyncio
async def test_no_payload_keys_are_returned(
    mock_pipeline_handler, mock_run_query_pipeline, mock_retrieval_query
):
    mock_result = HaystackResult()
    mock_run_query_pipeline.return_value = [mock_result]
    mock_retrieval_query.payload_keys = []
    mock_search_response = SearchResponse(
        retrievals=[
            RetrievedDocument(
                id=mock_result.id,
                collection_id="test_coll",
                content=mock_result.content,
                mime_type=mock_result.mime_type,
                metadata=mock_result.meta,
                asset_url=None,
                score=mock_result.score,
            )
        ]
    )

    with (
        patch(
            "src.visual_search.v1.apis.search.search",
            return_value=mock_search_response,
        ),
        patch(
            "src.visual_search.v1.apis.search.list_collections",
            return_value=[
                Collection(
                    id="1", pipeline="test", name="test1", created_at=datetime.utcnow()
                ),
                Collection(
                    id="2", pipeline="test", name="test2", created_at=datetime.utcnow()
                ),
            ],
        ),
    ):
        response = await retrieval(retrieval_query=mock_retrieval_query)
        result = json.loads(response.json())
        assert "d" not in result["retrievals"][0]["metadata"]
        assert "e" not in result["retrievals"][0]["metadata"]
        assert "f" not in result["retrievals"][0]["metadata"]


@pytest.mark.asyncio
async def test_get_custom_aws_credentials_success(monkeypatch):
    # Mock the config loader to prevent filesystem access in the CI environment
    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)

    # Mock the Kubernetes client
    class MockV1Api:
        def read_namespaced_secret(self, name, namespace):
            class MockSecret:
                data = {
                    "aws_access_key_id": "dGVzdF9rZXk=",  # base64 for 'test_key'
                    "aws_secret_access_key": "dGVzdF9zZWNyZXQ=",  # base64 for 'test_secret'
                    "aws_region": "dGVzdF9yZWdpb24=",  # base64 for 'test_region'
                    "endpoint_url": "aHR0cDovL2xvY2FsaG9zdDo4ODg4",  # base64 for 'http://localhost:8888'
                }

            return MockSecret()

    monkeypatch.setattr("kubernetes.client.CoreV1Api", MockV1Api)

    # Call the function
    secrets = await get_custom_aws_credentials("test-secret")

    # Assert the secrets are correctly retrieved
    assert secrets["aws_access_key_id"] == "test_key"
    assert secrets["aws_secret_access_key"] == "test_secret"
    assert secrets["aws_region"] == "test_region"
    assert secrets["endpoint_url"] == "http://localhost:8888"


@pytest.mark.asyncio
async def test_get_custom_aws_credentials_not_found(monkeypatch):
    # Mock the config loader to prevent filesystem access in the CI environment
    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)

    # Mock the Kubernetes client to raise an exception
    class MockV1Api:
        def read_namespaced_secret(self, name, namespace):
            raise ApiException(status=404)

    monkeypatch.setattr("kubernetes.client.CoreV1Api", MockV1Api)

    # Call the function and assert it raises SecretsNotFoundError
    with pytest.raises(SecretsNotFoundError):
        await get_custom_aws_credentials("non-existent-secret")


@pytest.mark.asyncio
async def test_get_custom_aws_credentials_nvcf_success(monkeypatch):
    # Mock the environment variable
    monkeypatch.setenv("NGC_SECRETS_FILE_PATH", "/var/secrets/secrets.json")

    # Mock the NVCFFileBasedSecretsManager
    class MockNVCFManager:
        def __init__(self, nvcf_secrets_path: str = None):
            pass

        def get_secrets(self):
            return {
                "aws_access_key_id": "nvcf_key",
                "aws_secret_access_key": "nvcf_secret",
                "aws_region": "nvcf_region",
                "endpoint_url": "http://nvcf.example.com",
            }

    monkeypatch.setattr(
        "src.visual_search.v1.apis.search.NVCFFileBasedSecretsManager", MockNVCFManager
    )
    # Call the function
    secrets = await get_custom_aws_credentials("test-secret")

    # Assert the secrets are correctly retrieved
    assert secrets["aws_access_key_id"] == "nvcf_key"
    assert secrets["aws_secret_access_key"] == "nvcf_secret"
    assert secrets["aws_region"] == "nvcf_region"
    assert secrets["endpoint_url"] == "http://nvcf.example.com"


@pytest.mark.asyncio
async def test_get_custom_aws_credentials_nvcf_not_found(monkeypatch):
    # Mock the environment variable
    monkeypatch.setenv("NGC_SECRETS_FILE_PATH", "/var/secrets/secrets.json")

    # Mock the NVCFFileBasedSecretsManager to raise an error
    class MockNVCFManager:
        def get_secrets(self):
            return None

    monkeypatch.setattr(
        "src.visual_search.v1.apis.search.NVCFFileBasedSecretsManager", MockNVCFManager
    )

    # Call the function and assert it raises SecretsNotFoundError
    with pytest.raises(SecretsNotFoundError):
        await get_custom_aws_credentials("test-secret")
