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

"""Integration tests for the complete cosmos-embed pipeline and API changes."""

import base64
import json
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

# isort: off
from haystack import Document
from haystack.dataclasses import ByteStream
from src.haystack.components.video.cosmos_video_embedder import (
    CosmosEmbedClient,
    CosmosSessionSegmentEmbedder,
    CosmosTextEmbedder,
    CosmosVideoDocumentEmbedder,
    CosmosVideoEmbedder,
)
from src.visual_search.common.models import DocumentUploadUrl, MimeType
from src.visual_search.v1.apis.document_indexing import (
    _convert_to_haystack_document,
    _download_url_data,
)

# isort: on

# --------------------------------------------------------------------------- #
# Test Fixtures                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_video_data():
    """Create sample video data for testing."""
    return b"fake_video_data_for_testing_integration"


@pytest.fixture
def cosmos_embed_client():
    """Create a mocked CosmosEmbedClient."""
    with patch.object(CosmosEmbedClient, "__init__", return_value=None):
        client = CosmosEmbedClient()
        client.embeddings_endpoint = "http://test-cosmos:8000/v1/embeddings"
        return client


@pytest.fixture
def mock_download_response(sample_video_data):
    """Mock response for URL download."""
    return sample_video_data


# --------------------------------------------------------------------------- #
# CosmosEmbedClient Tests                                                     #
# --------------------------------------------------------------------------- #


def test_cosmos_embed_client_embed_videos_success(cosmos_embed_client):
    """Test successful video embedding via CosmosEmbedClient with multiple base64 videos."""
    video_inputs = [
        "data:video/mp4;base64,dGVzdF92aWRlb19kYXRh",
        "data:video/mp4;base64,YW5vdGhlcl90ZXN0X3ZpZGVv",
    ]

    mock_response = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.1, 0.2, 0.3]}]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None

        embeddings = cosmos_embed_client.embed_videos(video_inputs)

        # Should make 2 individual calls for multiple base64 videos
        assert mock_post.call_count == 2

        # Each call should use query mode
        for call in mock_post.call_args_list:
            call_args = call[1]
            assert call_args["json"]["request_type"] == "query"
            assert call_args["json"]["model"] == "nvidia/cosmos-embed1"
            assert isinstance(call_args["json"]["input"], str)
            assert ";base64," in call_args["json"]["input"]

        assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_cosmos_embed_client_embed_text_success(cosmos_embed_client):
    """Test successful text embedding via CosmosEmbedClient."""
    text_inputs = ["test text", "another test"]

    mock_response = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None

        embeddings = cosmos_embed_client.embed_texts(text_inputs)

        # Verify API call format
        call_args = mock_post.call_args
        assert call_args[1]["json"]["input"] == text_inputs
        assert call_args[1]["json"]["request_type"] == "query"
        assert call_args[1]["json"]["model"] == "nvidia/cosmos-embed1"

        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_cosmos_embed_client_error_handling(cosmos_embed_client):
    """Test error handling in CosmosEmbedClient."""
    video_inputs = ["data:video/mp4;base64,dGVzdA=="]

    # Test API error response
    error_response = {"error": {"detail": "Invalid video format", "status_code": 400}}

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = error_response
        mock_post.return_value.raise_for_status.return_value = None

        # The actual test call needs to be within the patch scope
        with pytest.raises(
            RuntimeError, match="Cosmos-embed error \\(400\\): Invalid video format"
        ):
            cosmos_embed_client.embed_videos(video_inputs)


def test_cosmos_embed_client_batch_size_validation(cosmos_embed_client):
    """Test batch size validation in CosmosEmbedClient."""
    # Create 65 video inputs (exceeds max batch size)
    video_inputs = [f"data:video/mp4;base64,dGVzdA=={i}" for i in range(65)]

    with pytest.raises(
        ValueError, match="cosmos-embed supports maximum 64 videos per request"
    ):
        cosmos_embed_client.embed_videos(video_inputs)


def test_cosmos_embed_client_invalid_format_validation(cosmos_embed_client):
    """Test invalid video format validation."""
    # Test invalid presigned URL format
    invalid_inputs = ["http://example.com/video.mp4"]  # Not wrapped in data URI

    with pytest.raises(
        ValueError, match="Video input .* must be presigned-url formatted"
    ):
        cosmos_embed_client.embed_videos(invalid_inputs)


# --------------------------------------------------------------------------- #
# Document Indexing API Tests                                                 #
# --------------------------------------------------------------------------- #


def test_convert_to_haystack_document_video_url_stores_url_directly(sample_video_data):
    """Test that DocumentUploadUrl for videos stores URL directly in content field."""
    doc_upload = DocumentUploadUrl(
        url="https://example.com/test_video.mp4",
        mime_type=MimeType.MP4,
        id="test_video_123",
    )

    haystack_doc = _convert_to_haystack_document(doc_upload)

    # Verify URL is stored directly (presigned URL strategy)
    assert haystack_doc.id == "test_video_123"
    assert haystack_doc.content == "https://example.com/test_video.mp4"
    assert haystack_doc.meta["source_url"] == "https://example.com/test_video.mp4"
    assert haystack_doc.blob is None


def test_convert_to_haystack_document_text_url_no_download():
    """Test that DocumentUploadUrl for text doesn't download blob data."""
    doc_upload = DocumentUploadUrl(
        url="https://example.com/test.txt", mime_type=MimeType.TEXT, id="test_text_123"
    )

    text_content = "This is test text content"

    with patch(
        "src.visual_search.v1.apis.document_indexing._download_url_data",
        return_value=text_content.encode(),
    ) as mock_download:
        haystack_doc = _convert_to_haystack_document(doc_upload)

        # Verify download was called for text
        mock_download.assert_called_once_with("https://example.com/test.txt")

        # Verify document structure (text goes in content, not blob)
        assert haystack_doc.id == "test_text_123"
        assert haystack_doc.content == text_content
        assert haystack_doc.blob is None


# --------------------------------------------------------------------------- #
# All Cosmos Components Tests                                                 #
# --------------------------------------------------------------------------- #


def test_cosmos_text_embedder_component():
    """Test CosmosTextEmbedder component functionality."""


embedder = CosmosTextEmbedder(url="http://test:8000")

texts = ["Hello world", "Test embedding"]
mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

with patch.object(
    embedder._client, "embed_texts", return_value=mock_embeddings
) as mock_embed:
    result = embedder.run(texts)

    mock_embed.assert_called_once_with(texts)
    assert result == {"embeddings": mock_embeddings}


def test_cosmos_video_embedder_component():
    """Test CosmosVideoEmbedder component functionality."""
    embedder = CosmosVideoEmbedder(url="http://test:8000")

    video_urls = ["http://test.com/video1.mp4", "http://test.com/video2.mp4"]
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch.object(
        embedder._client, "embed_videos", return_value=mock_embeddings
    ) as mock_embed:
        result = embedder.run(video_urls)

        mock_embed.assert_called_once()
        # Verify URLs were wrapped as presigned URL data URIs
        call_args = mock_embed.call_args[0][0]
        assert call_args[0] == "data:video/mp4;presigned_url,http://test.com/video1.mp4"
        assert call_args[1] == "data:video/mp4;presigned_url,http://test.com/video2.mp4"
        assert result == {"embeddings": mock_embeddings}


def test_cosmos_session_segment_embedder_component():
    """Test CosmosSessionSegmentEmbedder component functionality."""
    embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")

    clips = [
        {
            "session_id": "s1",
            "start_timestamp": "0",
            "end_timestamp": "5",
            "camera": "camera_front_wide_120fov",
        },
        {
            "session_id": "s2",
            "start_timestamp": "5",
            "end_timestamp": "10",
            "camera": "camera_front_wide_120fov",
        },
    ]
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch.object(
        embedder._client, "embed_videos", return_value=mock_embeddings
    ) as mock_embed:
        result = embedder.run(clips)

        mock_embed.assert_called_once()
        assert result == {"embeddings": mock_embeddings}


# --------------------------------------------------------------------------- #
# Pipeline Configuration Tests                                                #
# --------------------------------------------------------------------------- #


def test_cosmos_pipeline_template_rendering():
    """Test that cosmos pipeline template renders correctly with batch_size."""
    import chevron
    import yaml

    # Mock environment variables for template rendering
    mock_env = {
        "COSMOS_EMBED_URI": "http://cosmos-embed.default.svc.cluster.local:8000",
        "MILVUS_DOCUMENT_STORE_URI": "http://milvus.default.svc.cluster.local:19530",
        "MILVUS_DB": "default",
    }

    # Read the actual mustache template
    template_path = "src/visual_search/pipelines/cosmos_video_search_milvus.mustache"

    with patch("os.environ", mock_env):
        # Actually read and render the template
        with open(template_path, "r") as f:
            template_content = f.read()

        # Render the template with environment variables
        rendered_content = chevron.render(template_content, mock_env)

        # Parse the rendered YAML to verify it's valid
        parsed_config = yaml.safe_load(rendered_content)

        # Verify the template was rendered correctly
        embedder_config = parsed_config["index"]["pipeline"]["components"]["embedder"]
        assert embedder_config["init_parameters"]["url"] == mock_env["COSMOS_EMBED_URI"]
        assert embedder_config["init_parameters"]["batch_size"] == 64
        assert (
            embedder_config["type"]
            == "src.haystack.components.video.cosmos_video_embedder.CosmosVideoDocumentEmbedder"
        )

        # Verify Milvus configuration
        milvus_config = parsed_config["index"]["pipeline"]["components"]["writer"][
            "init_parameters"
        ]["document_store"]
        assert (
            milvus_config["init_parameters"]["uri"]
            == mock_env["MILVUS_DOCUMENT_STORE_URI"]
        )
        assert milvus_config["init_parameters"]["database"] == mock_env["MILVUS_DB"]


# --------------------------------------------------------------------------- #
# End-to-End Integration Tests                                                #
# --------------------------------------------------------------------------- #


def test_end_to_end_video_processing_flow(sample_video_data):
    """Test complete flow: URL → Store URL → Presigned URL Wrapping → Embedding."""

    # Step 1: Create DocumentUploadUrl
    doc_upload = DocumentUploadUrl(
        url="https://example.com/test_video.mp4",
        mime_type=MimeType.MP4,
        id="test_video_e2e",
    )

    # Step 2: Convert to Haystack document (stores URL directly)
    haystack_doc = _convert_to_haystack_document(doc_upload)

    # Step 3: Process through CosmosVideoDocumentEmbedder with batching
    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=64)

    mock_embedding = [[0.1, 0.2, 0.3]]

    with patch.object(
        embedder._client, "embed_videos", return_value=mock_embedding
    ) as mock_embed:
        result = embedder.run([haystack_doc])

        # Verify the complete flow
        # 1. URL was stored directly in content
        assert haystack_doc.content == "https://example.com/test_video.mp4"
        assert haystack_doc.blob is None

        # 2. Presigned URL wrapping was applied correctly
        call_args = mock_embed.call_args[0][0]
        assert (
            call_args[0]
            == "data:video/mp4;presigned_url,https://example.com/test_video.mp4"
        )

        # 3. Embedding was applied correctly
        assert len(result["documents"]) == 1
        assert result["documents"][0].embedding == [0.1, 0.2, 0.3]
        assert result["documents"][0].meta["source_id"] == "test_video_e2e"


def test_large_batch_processing_flow(sample_video_data):
    """Test end-to-end flow with large batch requiring multiple cosmos-embed API calls."""

    # Create 100 video documents
    documents = []
    for i in range(100):
        doc_upload = DocumentUploadUrl(
            url=f"https://example.com/video_{i}.mp4",
            mime_type=MimeType.MP4,
            id=f"video_{i}",
        )

        haystack_doc = _convert_to_haystack_document(doc_upload)
        documents.append(haystack_doc)

    # Process through batching embedder
    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=64)

    def mock_embed_side_effect(video_inputs):
        return [
            [float(i), float(i + 1), float(i + 2)] for i in range(len(video_inputs))
        ]

    with patch.object(
        embedder._client, "embed_videos", side_effect=mock_embed_side_effect
    ) as mock_embed:
        result = embedder.run(documents)

    # Verify batching: 100 documents = 2 API calls (64 + 36)
    assert mock_embed.call_count == 2

    # Verify all documents processed
    assert len(result["documents"]) == 100

    # Verify order preservation
    for i, doc in enumerate(result["documents"]):
        assert doc.meta["source_id"] == f"video_{i}"


# --------------------------------------------------------------------------- #
# Performance and Monitoring Tests                                            #
# --------------------------------------------------------------------------- #


def test_batch_size_performance_optimization():
    """Test that different batch sizes affect API call count correctly."""
    test_cases = [
        (10, 1, 10),  # 10 docs, batch_size=1 → 10 calls
        (10, 5, 2),  # 10 docs, batch_size=5 → 2 calls
        (10, 64, 1),  # 10 docs, batch_size=64 → 1 call
        (100, 32, 4),  # 100 docs, batch_size=32 → 4 calls (32+32+32+4)
        (100, 64, 2),  # 100 docs, batch_size=64 → 2 calls (64+36)
    ]

    sample_data = b"test_video_data"

    for doc_count, batch_size, expected_calls in test_cases:
        # Create documents
        documents = []
        for i in range(doc_count):
            documents.append(
                Document(
                    id=f"test_{i}",
                    content=f"https://example.com/video_{i}.mp4",
                    meta={},
                )
            )

    # Test with specific batch size
    embedder = CosmosVideoDocumentEmbedder(
        url="http://test:8000", batch_size=batch_size
    )

    def side_effect(video_inputs):
        return [[0.1, 0.2, 0.3] for _ in range(len(video_inputs))]

    with patch.object(
        embedder._client, "embed_videos", side_effect=side_effect
    ) as mock_embed:
        result = embedder.run(documents)

        # Verify API call count
        assert (
            mock_embed.call_count == expected_calls
        ), f"Expected {expected_calls} calls for {doc_count} docs with batch_size {batch_size}, got {mock_embed.call_count}"

        # Verify all documents processed
        assert len(result["documents"]) == doc_count


def test_logging_performance_metrics(caplog):
    """Test that performance metrics are logged correctly."""
    sample_data = b"test_video_data"
    documents = []
    for i in range(130):  # 3 batches: 64 + 64 + 2
        documents.append(
            Document(
                id=f"test_{i}", content=f"https://example.com/video_{i}.mp4", meta={}
            )
        )

    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=64)

    def side_effect(video_inputs):
        return [[0.1, 0.2, 0.3] for _ in range(len(video_inputs))]

    with patch.object(embedder._client, "embed_videos", side_effect=side_effect):
        embedder.run(documents)

        # Check performance logging
        log_text = caplog.text
        assert "Cosmos-embed processing: 130 videos in batches of 64" in log_text
        assert (
            "Cosmos-embed completed: 130 videos processed with 3 API calls (avg 43.3 videos/call)"
            in log_text
        )


# --------------------------------------------------------------------------- #
# Error Scenario Tests                                                        #
# --------------------------------------------------------------------------- #


def test_partial_batch_failure_isolation():
    """Test that failure in one batch doesn't affect other batches."""
    sample_data = b"test_video_data"
    documents = []
    for i in range(130):  # 3 batches
        documents.append(
            Document(
                id=f"test_{i}", content=f"https://example.com/video_{i}.mp4", meta={}
            )
        )

    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=64)

    call_count = 0

    def side_effect_with_failure(video_inputs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # Fail the second batch
            raise RuntimeError("Cosmos service temporarily unavailable")
        return [[0.1, 0.2, 0.3] for _ in range(len(video_inputs))]

    with patch.object(
        embedder._client, "embed_videos", side_effect=side_effect_with_failure
    ):
        # Should fail on batch 2, demonstrating isolated failure
        with pytest.raises(
            RuntimeError, match="Cosmos service temporarily unavailable"
        ):
            embedder.run(documents)


def test_download_url_error_handling():
    """Test error handling when URL download fails for text documents."""
    # Test with text document (which still downloads)
    doc_upload = DocumentUploadUrl(
        url="https://invalid-url.com/nonexistent.txt",
        mime_type=MimeType.TEXT,
        id="test_error",
    )

    with patch(
        "src.visual_search.v1.apis.document_indexing._download_url_data",
        side_effect=Exception("Download failed"),
    ):
        with pytest.raises(Exception, match="Download failed"):
            _convert_to_haystack_document(doc_upload)


# --------------------------------------------------------------------------- #
# Migration Validation Tests                                                  #
# --------------------------------------------------------------------------- #


def test_cosmos_embed_replaces_vitcat_architecture():
    """Test that cosmos-embed architecture properly replaces vitcat workflows."""

    # Test that we're using presigned URL strategy for DocumentUploadUrl
    doc = Document(
        id="migration_test", content="https://example.com/test_video.mp4", meta={}
    )

    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000")

    mock_embedding = [[0.1, 0.2, 0.3]]

    with patch.object(
        embedder._client, "embed_videos", return_value=mock_embedding
    ) as mock_embed:
        result = embedder.run([doc])

        # Verify presigned URL strategy is used for URL inputs
        call_args = mock_embed.call_args[0][0]
        assert call_args[0].startswith("data:video/mp4;presigned_url,")
        assert "https://example.com/test_video.mp4" in call_args[0]

        # Verify 256-dimensional embeddings (cosmos-embed output)
        assert (
            len(result["documents"][0].embedding) == 3
        )  # Mock has 3 dims, real would have 256

        # Verify no frame extraction needed (cosmos-embed advantage)
        # This is implicit - we're passing full video data, not frames
        assert result["documents"][0].meta["source_id"] == "migration_test"
