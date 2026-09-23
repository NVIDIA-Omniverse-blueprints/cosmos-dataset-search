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

"""Unit tests for CosmosVideoDocumentEmbedder with batching functionality."""

from typing import Dict, List
from unittest.mock import MagicMock, patch
import base64
import pytest
from haystack import Document
from haystack.dataclasses import ByteStream

from src.haystack.components.video.cosmos_video_embedder import (
    CosmosVideoDocumentEmbedder,
    CosmosVideoEmbedder,
    CosmosTextEmbedder,
    CosmosSessionSegmentEmbedder,
)


# --------------------------------------------------------------------------- #
# Test Fixtures and Helpers                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_video_data():
    """Create sample video data for testing."""
    return b"fake_video_data_for_testing"


@pytest.fixture
def sample_document_url():
    """Create a sample video document with URL."""
    return Document(
        id="test_video_1",
        content="https://example.com/test_video.mp4",
        meta={"source": "test"}
    )


@pytest.fixture  
def sample_document_base64(sample_video_data):
    """Create a sample video document with base64 data URI."""
    import base64
    b64_data = base64.b64encode(sample_video_data).decode('utf-8')
    return Document(
        id="test_video_1",
        content=f"data:video/mp4;base64,{b64_data}",
        meta={"source": "test"}
    )


@pytest.fixture
def multiple_documents_url():
    """Create multiple video documents with URLs for batch testing."""
    documents = []
    for i in range(5):
        documents.append(Document(
            id=f"test_video_{i+1}",
            content=f"https://example.com/test_video_{i+1}.mp4",
            meta={"source": f"test_{i}"}
        ))
    return documents


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    return [[0.1, 0.2, 0.3] for _ in range(64)]  # Max batch size embeddings


@pytest.fixture
def embedder():
    """Create a CosmosVideoDocumentEmbedder instance for testing."""
    return CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=64)


# --------------------------------------------------------------------------- #
# Initialization Tests                                                        #
# --------------------------------------------------------------------------- #

def test_initialization_default_batch_size():
    """Test default initialization with batch_size=64."""
    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000")
    assert embedder.batch_size == 64


def test_initialization_custom_batch_size():
    """Test initialization with custom batch size."""
    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=32)
    assert embedder.batch_size == 32


def test_initialization_invalid_batch_size_too_large():
    """Test initialization fails with batch_size > 64."""
    with pytest.raises(ValueError, match="cosmos-embed maximum batch size is 64"):
        CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=65)


def test_initialization_invalid_batch_size_too_small():
    """Test initialization fails with batch_size < 1."""
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=0)


# --------------------------------------------------------------------------- #
# Input Validation Tests                                                      #
# --------------------------------------------------------------------------- #

def test_input_checks_valid_document_with_url():
    """Test input_checks passes for valid document with URL in content."""
    doc = Document(id="test", content="https://example.com/video.mp4", meta={})
    assert CosmosVideoDocumentEmbedder.input_checks(doc) is True


def test_input_checks_valid_document_with_data_uri():
    """Test input_checks passes for valid document with data URI in content."""
    doc = Document(id="test", content="data:video/mp4;base64,dGVzdA==", meta={})
    assert CosmosVideoDocumentEmbedder.input_checks(doc) is True


def test_input_checks_no_content():
    """Test input_checks fails for document without content."""
    doc = Document(id="test", meta={})
    with pytest.raises(ValueError, match="Document .* must provide a presigned URL or base64 data URI"):
        CosmosVideoDocumentEmbedder.input_checks(doc)


def test_input_checks_empty_content():
    """Test input_checks fails for document with empty content."""
    doc = Document(id="test", content="", meta={})
    with pytest.raises(ValueError, match="Document .* must provide a presigned URL or base64 data URI"):
        CosmosVideoDocumentEmbedder.input_checks(doc)


# --------------------------------------------------------------------------- #
# Basic Functionality Tests                                                   #
# --------------------------------------------------------------------------- #

def test_run_empty_documents(embedder):
    """Test run with empty document list."""
    result = embedder.run([])
    assert result == {"documents": []}


def test_run_single_document_url(embedder, sample_document_url):
    """Test run with single document containing URL."""
    mock_embedding = [[0.1, 0.2, 0.3]]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embedding) as mock_embed:
        result = embedder.run([sample_document_url])
        
        # Verify the API was called once
        mock_embed.assert_called_once()
        
        # Check the input format - URL should be wrapped as presigned URL
        call_args = mock_embed.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == "data:video/mp4;presigned_url,https://example.com/test_video.mp4"
        
        # Check the result
        assert len(result["documents"]) == 1
        assert result["documents"][0].embedding == [0.1, 0.2, 0.3]
        assert result["documents"][0].meta["source_id"] == "test_video_1"


def test_run_single_document_base64(embedder, sample_document_base64):
    """Test run with single document containing base64 data URI."""
    mock_embedding = [[0.1, 0.2, 0.3]]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embedding) as mock_embed:
        result = embedder.run([sample_document_base64])
        
        # Verify the API was called once
        mock_embed.assert_called_once()
        
        # Check the input format - data URI should be passed through
        call_args = mock_embed.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].startswith("data:video/mp4;base64,")
        
        # Check the result
        assert len(result["documents"]) == 1
        assert result["documents"][0].embedding == [0.1, 0.2, 0.3]
        assert result["documents"][0].meta["source_id"] == "test_video_1"


# --------------------------------------------------------------------------- #
# Batching Tests                                                              #
# --------------------------------------------------------------------------- #

def test_run_exact_batch_size(embedder):
    """Test run with exactly 64 documents (one full batch)."""
    # Create exactly 64 documents with URLs
    documents = []
    for i in range(64):
        documents.append(Document(
            id=f"test_video_{i}",
            content=f"https://example.com/test_video_{i}.mp4",
            meta={"source": f"test_{i}"}
        ))
    
    # Mock 64 embeddings
    mock_embeddings = [[float(i), float(i+1), float(i+2)] for i in range(64)]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings) as mock_embed:
        result = embedder.run(documents)
        
        # Should be called exactly once with all 64 videos
        mock_embed.assert_called_once()
        assert len(mock_embed.call_args[0][0]) == 64
        
        # Check all documents are processed
        assert len(result["documents"]) == 64
        for i, doc in enumerate(result["documents"]):
            assert doc.embedding == [float(i), float(i+1), float(i+2)]
            assert doc.meta["source_id"] == f"test_video_{i}"


def test_run_multiple_batches(embedder):
    """Test run with 130 documents (3 batches: 64 + 64 + 2)."""
    # Create 130 documents with URLs
    documents = []
    for i in range(130):
        documents.append(Document(
            id=f"test_video_{i}",
            content=f"https://example.com/test_video_{i}.mp4",
            meta={"source": f"test_{i}"}
        ))
    
    # Mock embeddings for different batch sizes
    def side_effect(video_inputs):
        batch_size = len(video_inputs)
        return [[float(i), float(i+1), float(i+2)] for i in range(batch_size)]
    
    with patch.object(embedder._client, 'embed_videos', side_effect=side_effect) as mock_embed:
        result = embedder.run(documents)
        
        # Should be called 3 times: 64 + 64 + 2
        assert mock_embed.call_count == 3
        
        # Check call arguments
        call_args_list = mock_embed.call_args_list
        assert len(call_args_list[0][0][0]) == 64  # First batch: 64 videos
        assert len(call_args_list[1][0][0]) == 64  # Second batch: 64 videos  
        assert len(call_args_list[2][0][0]) == 2   # Third batch: 2 videos
        
        # Check all documents are processed
        assert len(result["documents"]) == 130


def test_run_custom_batch_size():
    """Test run with custom batch size."""
    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=10)
    
    # Create 25 documents (3 batches: 10 + 10 + 5)
    documents = []
    for i in range(25):
        documents.append(Document(
            id=f"test_video_{i}",
            content=f"https://example.com/test_video_{i}.mp4",
            meta={"source": f"test_{i}"}
        ))
    
    def side_effect(video_inputs):
        batch_size = len(video_inputs)
        return [[float(i)] for i in range(batch_size)]
    
    with patch.object(embedder._client, 'embed_videos', side_effect=side_effect) as mock_embed:
        result = embedder.run(documents)
        
        # Should be called 3 times with batches of 10, 10, 5
        assert mock_embed.call_count == 3
        call_args_list = mock_embed.call_args_list
        assert len(call_args_list[0][0][0]) == 10  # First batch
        assert len(call_args_list[1][0][0]) == 10  # Second batch
        assert len(call_args_list[2][0][0]) == 5   # Third batch
                
        assert len(result["documents"]) == 25


# --------------------------------------------------------------------------- #
# Error Handling Tests                                                        #
# --------------------------------------------------------------------------- #

def test_run_embedding_mismatch_error(embedder, multiple_documents_url):
    """Test run fails when embedding count doesn't match document count."""
    # Return wrong number of embeddings (3 instead of 5)
    mock_embeddings = [[0.1, 0.2, 0.3] for _ in range(3)]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings):
        with pytest.raises(RuntimeError, match="Embeddings batch size .* does not match document batch size"):
            embedder.run(multiple_documents_url)


def test_run_client_error_propagation(embedder, sample_document_url):
    """Test that client errors are properly propagated."""
    with patch.object(embedder._client, 'embed_videos', side_effect=RuntimeError("Cosmos service error")):
        with pytest.raises(RuntimeError, match="Cosmos service error"):
            embedder.run([sample_document_url])


# --------------------------------------------------------------------------- #
# Logging Tests                                                               #
# --------------------------------------------------------------------------- #

def test_logging_single_batch(embedder, multiple_documents_url, caplog):
    """Test logging for single batch processing."""
    mock_embeddings = [[0.1, 0.2, 0.3] for _ in range(5)]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings):
        embedder.run(multiple_documents_url)        
        # Check info logs
        assert "Cosmos-embed processing: 5 videos in batches of 64" in caplog.text
        assert "Cosmos-embed completed: 5 videos processed with 1 API calls (avg 5.0 videos/call)" in caplog.text


def test_logging_multiple_batches(embedder, caplog):
    """Test logging for multiple batch processing."""
    # Create 70 documents (2 batches: 64 + 6) with URLs
    documents = []
    for i in range(70):
        documents.append(Document(
            id=f"test_video_{i}",
            content=f"https://example.com/test_video_{i}.mp4",
            meta={"source": f"test_{i}"}
        ))
    
    def side_effect(video_inputs):
        return [[0.1, 0.2, 0.3] for _ in range(len(video_inputs))]
    
    with patch.object(embedder._client, 'embed_videos', side_effect=side_effect):
        embedder.run(documents)
        
        # Check info logs
    assert "Cosmos-embed processing: 70 videos in batches of 64" in caplog.text
    assert "Cosmos-embed completed: 70 videos processed with 2 API calls (avg 35.0 videos/call)" in caplog.text


# --------------------------------------------------------------------------- #
# Integration Tests                                                           #
# --------------------------------------------------------------------------- #

def test_presigned_url_format(embedder, sample_document_url):
    """Test that presigned URL format is correct."""
    mock_embedding = [[0.1, 0.2, 0.3]]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embedding) as mock_embed:
        embedder.run([sample_document_url])
        
        # Verify the exact format of the presigned URL wrapped input
        call_args = mock_embed.call_args[0][0]
        expected_format = "data:video/mp4;presigned_url,https://example.com/test_video.mp4"
        assert call_args[0] == expected_format


def test_base64_data_uri_passthrough(embedder, sample_document_base64):
    """Test that base64 data URIs are passed through unchanged."""
    mock_embedding = [[0.1, 0.2, 0.3]]
    
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embedding) as mock_embed:
        embedder.run([sample_document_base64])
        
        # Verify the data URI is passed through unchanged
        call_args = mock_embed.call_args[0][0]
        assert call_args[0].startswith("data:video/mp4;base64,")
        assert call_args[0] == sample_document_base64.content


def test_url_wrapping_uses_mp4_format(embedder):
    """Test that URL wrapping always uses mp4 format."""
    doc = Document(
        id="test",
        content="https://example.com/test_video.avi",  # Different extension
        meta={}
    )
    
    mock_embedding = [[0.1, 0.2, 0.3]]
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embedding) as mock_embed:
        embedder.run([doc])
        
        call_args = mock_embed.call_args[0][0]
        # URL wrapping always uses mp4 format
        expected_format = "data:video/mp4;presigned_url,https://example.com/test_video.avi"
        assert call_args[0] == expected_format


def test_document_order_preservation(embedder):
    """Test that document order is preserved through batching."""
    # Create documents with identifiable URLs
    documents = []
    for i in range(70):  # 2 batches to test cross-batch ordering
        documents.append(Document(
            id=f"video_{i:03d}",  # Zero-padded for easy sorting verification
            content=f"https://example.com/test_video_{i:03d}.mp4",
            meta={"index": i}
        ))
    
    # Mock embeddings with index-based values
    def side_effect(video_inputs):
        # Extract the index from base64 data to simulate order-dependent embeddings
        embeddings = []
        for i, video_input in enumerate(video_inputs):
            # Create embedding that includes batch info for verification
            embeddings.append([float(len(embeddings)), 0.0, 0.0])
        return embeddings
    
    with patch.object(embedder._client, 'embed_videos', side_effect=side_effect):
        result = embedder.run(documents)
        
        # Verify all documents are returned in the same order
        assert len(result["documents"]) == 70
        for i, doc in enumerate(result["documents"]):
            assert doc.meta["source_id"] == f"video_{i:03d}"
            assert doc.meta["index"] == i


# --------------------------------------------------------------------------- #
# Edge Cases                                                                  #
# --------------------------------------------------------------------------- #

def test_batch_boundary_edge_cases(embedder):
    """Test edge cases around batch boundaries."""
    test_cases = [1, 63, 64, 65, 127, 128, 129]  # Various boundary conditions
    
    for doc_count in test_cases:
        documents = []
        for i in range(doc_count):
            documents.append(Document(
                id=f"test_video_{i}",
                content=f"https://example.com/test_video_{i}.mp4",
                meta={"source": f"test_{i}"}
            ))
        
        def side_effect(video_inputs):
            return [[0.1, 0.2, 0.3] for _ in range(len(video_inputs))]
        
    with patch.object(embedder._client, 'embed_videos', side_effect=side_effect) as mock_embed:
        result = embedder.run(documents)
        
        # Calculate expected number of API calls
        expected_calls = (doc_count + 63) // 64  # Ceiling division
        assert mock_embed.call_count == expected_calls
        assert len(result["documents"]) == doc_count


# --------------------------------------------------------------------------- #
# Other Component Tests                                                       #
# --------------------------------------------------------------------------- #

def test_cosmos_video_embedder():
    """Test CosmosVideoEmbedder component."""
    embedder = CosmosVideoEmbedder(url="http://test:8000")
        
    video_urls = ["http://test.com/video1.mp4"]
    mock_embeddings = [[0.1, 0.2, 0.3]]
        
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings) as mock_embed:
        result = embedder.run(video_urls)
        mock_embed.assert_called_once()
        # Verify URL was wrapped
        call_args = mock_embed.call_args[0][0]
        assert call_args[0] == "data:video/mp4;presigned_url,http://test.com/video1.mp4"
        assert result == {"embeddings": [[0.1, 0.2, 0.3]]}


def test_cosmos_text_embedder():
    """Test CosmosTextEmbedder component."""
    embedder = CosmosTextEmbedder(url="http://test:8000")
        
    texts = ["test text"]
    mock_embeddings = [[0.1, 0.2, 0.3]]
            
    with patch.object(embedder._client, 'embed_texts', return_value=mock_embeddings):
        result = embedder.run(texts)
        assert result == {"embeddings": [[0.1, 0.2, 0.3]]}


def test_cosmos_session_segment_embedder():
    """Test CosmosSessionSegmentEmbedder component."""
    embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
        
    clips = [{"session_id": "s1", "start_timestamp": "0", "end_timestamp": "5", "camera": "camera_front_wide_120fov"}]
    mock_embeddings = [[0.1, 0.2, 0.3]]
        
    with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings):
        result = embedder.run(clips)
        assert result == {"embeddings": [[0.1, 0.2, 0.3]]}


# --------------------------------------------------------------------------- #
# HTTP Client Tests (for better coverage)                                    #
# --------------------------------------------------------------------------- #

def test_cosmos_embed_client_initialization():
    """Test CosmosEmbedClient initialization with different URL formats."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("localhost:8000")
    assert client.base_url == "http://localhost:8000"
    assert client.embeddings_endpoint == "http://localhost:8000/v1/embeddings"
    
    client = CosmosEmbedClient("http://test.com:9000")
    assert client.base_url == "http://test.com:9000"
    
    client = CosmosEmbedClient("http://test.com:9000/")
    assert client.base_url == "http://test.com:9000"


def test_cosmos_embed_client_embed_videos_empty():
    """Test embed_videos with empty input."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("http://test:8000")
    result = client.embed_videos([])
    assert result == []


def test_cosmos_embed_client_embed_videos_too_many():
    """Test embed_videos with too many videos (>64)."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = [f"data:video/mp4;base64,test{i}" for i in range(65)]
    
    with pytest.raises(ValueError, match="cosmos-embed supports maximum 64 videos per request"):
        client.embed_videos(video_inputs)


def test_cosmos_embed_client_embed_videos_mixed_content_types():
    """Test embed_videos with mixed base64 and presigned URL content."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = [
        "data:video/mp4;base64,dGVzdA==",
        "data:video/mp4;presigned_url,http://test.com/video.mp4"
    ]
    
    with pytest.raises(ValueError, match="Cannot mix base64 and presigned URL inputs in same request"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_single_base64_video(mock_post):
    """Test embed_videos with single base64 video."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    mock_post.return_value = mock_response
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;base64,dGVzdA=="]
    
    result = client.embed_videos(video_inputs)
    
    # Verify request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]['json']['request_type'] == 'query'
    assert call_args[1]['json']['input'] == "data:video/mp4;base64,dGVzdA=="
    assert result == [[0.1, 0.2, 0.3]]


@patch('requests.post')
def test_cosmos_embed_client_multiple_base64_videos(mock_post):
    """Test embed_videos with multiple base64 videos."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    mock_post.return_value = mock_response
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = [
        "data:video/mp4;base64,dGVzdDE=",
        "data:video/mp4;base64,dGVzdDI="
    ]
    
    result = client.embed_videos(video_inputs)
    
    # Should be called twice (once per video)
    assert mock_post.call_count == 2
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


@patch('requests.post')
def test_cosmos_embed_client_presigned_urls(mock_post):
    """Test embed_videos with presigned URLs."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ]
    }
    mock_post.return_value = mock_response
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = [
        "data:video/mp4;presigned_url,http://test.com/video1.mp4",
        "data:video/mp4;presigned_url,http://test.com/video2.mp4"
    ]
    
    result = client.embed_videos(video_inputs)
    
    # Should be called once with bulk_video request
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]['json']['request_type'] == 'bulk_video'
    assert call_args[1]['json']['input'] == video_inputs
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@patch('requests.post')
def test_cosmos_embed_client_presigned_url_validation_error(mock_post):
    """Test embed_videos with invalid presigned URL format."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = [
        "data:video/mp4;presigned_url,http://test.com/video1.mp4",
        "http://invalid.com/video2.mp4"  # Missing presigned_url format
    ]
    
    with pytest.raises(ValueError, match="Video input 1 must be presigned-url formatted"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_embed_texts_success(mock_post):
    """Test embed_texts with successful response."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ]
    }
    mock_post.return_value = mock_response
    
    client = CosmosEmbedClient("http://test:8000")
    texts = ["Hello world", "Test text"]
    
    result = client.embed_texts(texts)
    
    # Verify request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]['json']['request_type'] == 'query'
    assert call_args[1]['json']['input'] == texts
    assert call_args[1]['timeout'] == 60
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_cosmos_embed_client_embed_texts_empty():
    """Test embed_texts with empty input."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("http://test:8000")
    result = client.embed_texts([])
    assert result == []


def test_cosmos_embed_client_embed_texts_too_many():
    """Test embed_texts with too many texts (>64)."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    client = CosmosEmbedClient("http://test:8000")
    texts = [f"Text {i}" for i in range(65)]
    
    with pytest.raises(ValueError, match="cosmos-embed supports maximum 64 texts per request"):
        client.embed_texts(texts)


# --------------------------------------------------------------------------- #
# Error Handling Tests                                                        #
# --------------------------------------------------------------------------- #

@patch('requests.post')
def test_cosmos_embed_client_timeout_error(mock_post):
    """Test timeout error handling."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    import requests
    
    mock_post.side_effect = requests.exceptions.Timeout()
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;presigned_url,http://test.com/video.mp4"]
    
    with pytest.raises(RuntimeError, match="Cosmos-embed service timeout"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_connection_error(mock_post):
    """Test connection error handling."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    import requests
    
    mock_post.side_effect = requests.exceptions.ConnectionError()
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;presigned_url,http://test.com/video.mp4"]
    
    with pytest.raises(RuntimeError, match="Cannot connect to cosmos-embed service"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_http_400_error(mock_post):
    """Test HTTP 400 error handling."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    import requests
    
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_error = requests.exceptions.HTTPError()
    mock_error.response = mock_response
    mock_post.side_effect = mock_error
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;presigned_url,http://test.com/video.mp4"]
    
    with pytest.raises(ValueError, match="Invalid video format or request format"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_http_503_error(mock_post):
    """Test HTTP 503 error handling."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    import requests
    
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_error = requests.exceptions.HTTPError()
    mock_error.response = mock_response
    mock_post.side_effect = mock_error
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;presigned_url,http://test.com/video.mp4"]
    
    with pytest.raises(RuntimeError, match="Cosmos-embed service temporarily unavailable"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_http_500_error(mock_post):
    """Test HTTP 500 error handling."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    import requests
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_error = requests.exceptions.HTTPError()
    mock_error.response = mock_response
    mock_post.side_effect = mock_error
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;presigned_url,http://test.com/video.mp4"]
    
    with pytest.raises(RuntimeError, match="Cosmos-embed service error"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_response_error(mock_post):
    """Test error response from cosmos-embed service."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "error": {
            "detail": "Invalid video format",
            "status_code": 422
        }
    }
    mock_post.return_value = mock_response
    
    client = CosmosEmbedClient("http://test:8000")
    video_inputs = ["data:video/mp4;presigned_url,http://test.com/video.mp4"]
    
    with pytest.raises(RuntimeError, match="Cosmos-embed error \\(422\\): Invalid video format"):
        client.embed_videos(video_inputs)


@patch('requests.post')
def test_cosmos_embed_client_text_timeout_error(mock_post):
    """Test timeout error handling for text embedding."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedClient
    import requests
    
    mock_post.side_effect = requests.exceptions.Timeout()
    
    client = CosmosEmbedClient("http://test:8000")
    texts = ["Test text"]
    
    with pytest.raises(RuntimeError, match="Cosmos-embed service timeout"):
        client.embed_texts(texts)


# --------------------------------------------------------------------------- #
# Edge Case Tests                                                             #
# --------------------------------------------------------------------------- #

def test_cosmos_video_embedder_empty_urls():
    """Test CosmosVideoEmbedder with empty URL list."""
    embedder = CosmosVideoEmbedder(url="http://test:8000")
    result = embedder.run([])
    assert result == {"embeddings": []}


def test_cosmos_text_embedder_empty_texts():
    """Test CosmosTextEmbedder with empty text list."""
    embedder = CosmosTextEmbedder(url="http://test:8000")
    result = embedder.run([])
    assert result == {"embeddings": []}


def test_cosmos_session_segment_embedder_empty_clips():
    """Test CosmosSessionSegmentEmbedder with empty clips list."""
    embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
    result = embedder.run([])
    assert result == {"embeddings": []}


def test_cosmos_session_segment_embedder_invalid_clips():
    """Test CosmosSessionSegmentEmbedder with invalid clips."""
    embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
    
    clips = [
        "invalid_string",
        None
    ]
    
    result = embedder.run(clips)
    assert result == {"embeddings": []}


def test_cosmos_session_segment_embedder_json_encoding_error():
    """Test CosmosSessionSegmentEmbedder with JSON encoding error."""
    embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
    
    import json
    clips = [{"lambda": lambda x: x}]  # Lambda functions are not JSON serializable
    
    with patch('json.dumps', side_effect=TypeError("Object is not JSON serializable")):
        result = embedder.run(clips)
        assert result == {"embeddings": []}


def test_run_invalid_document_content():
    """Test that invalid document content raises appropriate error."""
    embedder = CosmosVideoDocumentEmbedder(url="http://test:8000", batch_size=64)
    
    doc = Document(id="test", content=None, meta={})
    
    with pytest.raises(ValueError, match="Document test must provide a presigned URL or base64 data URI"):
        embedder.run([doc])


# --------------------------------------------------------------------------- #
# Utility Method Tests                                                        #
# --------------------------------------------------------------------------- #

def test_get_mime_type_extension():
    """Test _get_mime_type_extension utility method."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedMixin
    
    mixin = CosmosEmbedMixin(url="http://test:8000")
    
    assert mixin._get_mime_type_extension("video/mp4") == "mp4"
    assert mixin._get_mime_type_extension("video/avi") == "avi"
    assert mixin._get_mime_type_extension("video/mov") == "mov"
    assert mixin._get_mime_type_extension("video/webm") == "webm"    
    assert mixin._get_mime_type_extension("VIDEO/MP4") == "mp4"
    assert mixin._get_mime_type_extension("video/unknown") == "mp4"
    assert mixin._get_mime_type_extension("application/octet-stream") == "mp4"


def test_cosmos_embed_mixin_initialization():
    """Test CosmosEmbedMixin initialization."""
    from src.haystack.components.video.cosmos_video_embedder import CosmosEmbedMixin
    
    mixin = CosmosEmbedMixin(url="http://test:8000")
    assert mixin.url == "http://test:8000"
    assert mixin._client is not None
    assert mixin._client.base_url == "http://test:8000"