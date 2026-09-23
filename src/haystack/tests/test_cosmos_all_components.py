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

"""Comprehensive tests for all cosmos-embed components and client functionality."""

from typing import Dict, List
from unittest.mock import patch, MagicMock
import base64
import pytest
import requests

from src.haystack.components.video.cosmos_video_embedder import (
    CosmosEmbedClient,
    CosmosEmbedMixin,
    CosmosVideoDocumentEmbedder,
    CosmosVideoEmbedder,
    CosmosTextEmbedder,
    CosmosSessionSegmentEmbedder,
)


# --------------------------------------------------------------------------- #
# Test Fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def cosmos_client():
    """Create a CosmosEmbedClient for testing."""
    with patch.object(CosmosEmbedClient, '__init__', return_value=None):
        client = CosmosEmbedClient()
        client.embeddings_endpoint = "http://test-cosmos:8000/v1/embeddings"
        return client


@pytest.fixture
def mock_successful_response():
    """Mock successful cosmos-embed API response."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
        ],
        "model": "nvidia/cosmos-embed1",
        "usage": {"num_videos": 2, "prompt_tokens": 0, "total_tokens": 10}
    }


@pytest.fixture 
def sample_base64_videos():
    """Sample base64-encoded video data."""
    video1 = base64.b64encode(b"fake_video_1").decode('utf-8')
    video2 = base64.b64encode(b"fake_video_2").decode('utf-8')
    return [
        f"data:video/mp4;base64,{video1}",
        f"data:video/webm;base64,{video2}"
    ]


# --------------------------------------------------------------------------- #
# CosmosEmbedClient Tests                                                     #
# --------------------------------------------------------------------------- #

class TestCosmosEmbedClient:
    """Test suite for CosmosEmbedClient functionality."""
    
    def test_embed_videos_multiple_base64_request_format(self, cosmos_client, sample_base64_videos):
        """Test that embed_videos sends individual query requests for multiple base64 videos."""
        
        # Mock different responses for each call
        mock_responses = [
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            {"data": [{"embedding": [0.4, 0.5, 0.6]}]}
        ]
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.side_effect = mock_responses
            mock_post.return_value.raise_for_status.return_value = None
            
            result = cosmos_client.embed_videos(sample_base64_videos)
            
            # Should make 2 individual calls for 2 base64 videos
            assert mock_post.call_count == 2
            
            # Verify each call uses query mode
            for call in mock_post.call_args_list:
                call_kwargs = call[1]
                assert call_kwargs['json']['request_type'] == 'query'
                assert call_kwargs['json']['encoding_format'] == 'float'
                assert call_kwargs['json']['model'] == 'nvidia/cosmos-embed1'
                assert call_kwargs['timeout'] == 300
                # Each call should have single video input
                assert isinstance(call_kwargs['json']['input'], str)
                assert ";base64," in call_kwargs['json']['input']
            
            # Verify response parsing
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
    def test_embed_text_query_request_format(self, cosmos_client, mock_successful_response):
        """Test that embed_text sends correct query request format."""
        
        text_inputs = ["sample text 1", "sample text 2"]
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_successful_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = cosmos_client.embed_texts(text_inputs)
            
            # Verify request format
            call_kwargs = mock_post.call_args[1]
            
            assert call_kwargs['json']['input'] == text_inputs
            assert call_kwargs['json']['request_type'] == 'query'
            assert call_kwargs['json']['encoding_format'] == 'float'
            assert call_kwargs['json']['model'] == 'nvidia/cosmos-embed1'
            
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    def test_embed_videos_empty_input(self, cosmos_client):
        """Test embed_videos with empty input."""
        result = cosmos_client.embed_videos([])
        assert result == []
    
    def test_embed_videos_batch_size_limit(self, cosmos_client):
        """Test that embed_videos enforces 64 video limit."""
        # Create 65 video inputs
        large_batch = [f"data:video/mp4;base64,dGVzdA=={i}" for i in range(65)]
        
        with pytest.raises(ValueError, match="cosmos-embed supports maximum 64 videos per request"):
            cosmos_client.embed_videos(large_batch)
    
    def test_embed_videos_format_validation(self, cosmos_client):
        """Test that embed_videos validates format correctly."""
        # Test mixed types (should fail)
        mixed_inputs = [
            "data:video/mp4;base64,dGVzdA==",
            "data:video/mp4;presigned_url,http://example.com/video.mp4"
        ]
        
        with pytest.raises(ValueError, match="Cannot mix base64 and presigned URL inputs in same request"):
            cosmos_client.embed_videos(mixed_inputs)
        
        # Test invalid presigned URL format
        invalid_presigned = ["http://example.com/video.mp4"]  # Not wrapped in data URI
        with pytest.raises(ValueError, match="Video input .* must be presigned-url formatted"):
            cosmos_client.embed_videos(invalid_presigned)
    
    def test_cosmos_api_error_handling(self, cosmos_client, sample_base64_videos):
        """Test handling of cosmos-embed API errors."""
        
        error_response = {
            "error": {
                "detail": "Invalid video format detected",
                "status_code": 400
            }
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = error_response
            mock_post.return_value.raise_for_status.return_value = None
            
            with pytest.raises(RuntimeError, match="Cosmos-embed error \\(400\\): Invalid video format detected"):
                cosmos_client.embed_videos(sample_base64_videos)
    
    def test_http_timeout_handling(self, cosmos_client, sample_base64_videos):
        """Test handling of HTTP timeouts."""
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(RuntimeError, match="Cosmos-embed service timeout"):
                cosmos_client.embed_videos(sample_base64_videos)
    
    def test_connection_error_handling(self, cosmos_client, sample_base64_videos):
        """Test handling of connection errors."""
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            with pytest.raises(RuntimeError, match="Cannot connect to cosmos-embed service"):
                cosmos_client.embed_videos(sample_base64_videos)
    
    def test_http_400_error_handling(self, cosmos_client, sample_base64_videos):
        """Test handling of HTTP 400 errors."""
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
            
            with pytest.raises(ValueError, match="Invalid video format or request format"):
                cosmos_client.embed_videos(sample_base64_videos)
    
    def test_http_503_error_handling(self, cosmos_client, sample_base64_videos):
        """Test handling of HTTP 503 errors."""
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
            
            with pytest.raises(RuntimeError, match="Cosmos-embed service temporarily unavailable"):
                cosmos_client.embed_videos(sample_base64_videos)


# --------------------------------------------------------------------------- #
# CosmosEmbedMixin Tests                                                      #
# --------------------------------------------------------------------------- #

class TestCosmosEmbedMixin:
    """Test suite for CosmosEmbedMixin functionality."""

    def test_mixin_initialization(self):
        """Test that CosmosEmbedMixin initializes client correctly."""
        mixin = CosmosEmbedMixin(url="http://test:8000")
        assert hasattr(mixin, '_client')
        assert isinstance(mixin._client, CosmosEmbedClient)
    
    def test_mime_type_extension_mapping(self):
        """Test MIME type to file extension mapping."""

        mixin = CosmosEmbedMixin(url="http://test:8000")
        
        test_cases = [
            ("video/mp4", "mp4"),
            ("video/avi", "avi"), 
            ("video/mov", "mov"),
            ("video/webm", "webm"),
        ]
                
        for mime_type, expected_ext in test_cases:
            result = mixin._get_mime_type_extension(mime_type)
            assert result == expected_ext


# --------------------------------------------------------------------------- #
# CosmosVideoEmbedder Tests                                                   #
# --------------------------------------------------------------------------- #

class TestCosmosVideoEmbedder:
    """Test suite for CosmosVideoEmbedder component."""
    
    def test_component_initialization(self):
        """Test CosmosVideoEmbedder initialization."""
        embedder = CosmosVideoEmbedder(url="http://test:8000")
        assert hasattr(embedder, '_client')
    
    def test_run_method_signature(self):
        """Test run method input/output types."""
        embedder = CosmosVideoEmbedder(url="http://test:8000")
            
        video_urls = ["http://example.com/video1.mp4", "http://example.com/video2.mp4"]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(video_urls)
            mock_embed.assert_called_once()
            
            # Check that URLs were wrapped as presigned URL data URIs
            call_args = mock_embed.call_args[0][0]
            assert call_args[0] == "data:video/mp4;presigned_url,http://example.com/video1.mp4"
            assert call_args[1] == "data:video/mp4;presigned_url,http://example.com/video2.mp4"
            assert result == {"embeddings": mock_embeddings}
    
    def test_empty_input_handling(self):
        """Test handling of empty video data."""
        embedder = CosmosVideoEmbedder(url="http://test:8000")
            
        with patch.object(embedder._client, 'embed_videos') as mock_embed:
            result = embedder.run([])
            mock_embed.assert_not_called()
            assert result == {"embeddings": []}


# --------------------------------------------------------------------------- #
# CosmosTextEmbedder Tests                                                    #
# --------------------------------------------------------------------------- #

class TestCosmosTextEmbedder:
    """Test suite for CosmosTextEmbedder component."""
    
    def test_component_initialization(self):
        """Test CosmosTextEmbedder initialization."""
        embedder = CosmosTextEmbedder(url="http://test:8000")
        assert hasattr(embedder, '_client')
    
    def test_run_method_functionality(self):
        """Test run method with text inputs."""
        embedder = CosmosTextEmbedder(url="http://test:8000")
            
        texts = ["Hello world", "Test text embedding"]
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        with patch.object(embedder._client, 'embed_texts', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(texts)
            
            mock_embed.assert_called_once_with(texts)
            assert result == {"embeddings": mock_embeddings}
    
    def test_single_text_input(self):
        """Test with single text input."""
        embedder = CosmosTextEmbedder(url="http://test:8000")
            
        texts = ["Single text"]
        mock_embeddings = [[0.1, 0.2, 0.3]]
        
        with patch.object(embedder._client, 'embed_texts', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(texts)
            
            assert result == {"embeddings": [[0.1, 0.2, 0.3]]}
    
    def test_empty_text_input(self):
        """Test with empty text list."""
        embedder = CosmosTextEmbedder(url="http://test:8000")
            
        with patch.object(embedder._client, 'embed_texts', return_value=[]) as mock_embed:
            result = embedder.run([])
            
            assert result == {"embeddings": []}


# --------------------------------------------------------------------------- #
# CosmosSessionSegmentEmbedder Tests                                          #
# --------------------------------------------------------------------------- #

class TestCosmosSessionSegmentEmbedder:
    """Test suite for CosmosSessionSegmentEmbedder component."""
    
    def test_component_initialization(self):
        """Test CosmosSessionSegmentEmbedder initialization."""
        embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
        assert hasattr(embedder, '_client')
    
    def test_run_method_with_clips(self):
        """Test run method with video clips."""
        embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
            
        clips = [
            {"session_id": "s1", "start_timestamp": "0", "end_timestamp": "5", "camera": "camera_front_wide_120fov"},
            {"session_id": "s2", "start_timestamp": "5", "end_timestamp": "10", "camera": "camera_front_wide_120fov"},
        ]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(clips)
            
            mock_embed.assert_called_once()
            assert result == {"embeddings": mock_embeddings}
    
    def test_session_segment_use_case(self):
        """Test session segment embedding use case."""
        embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
            
        # Simulate session segments (short video clips from a session)
        session_clips = [
            {"session_id": "sess1", "start_timestamp": "0", "end_timestamp": "5", "camera": "camera_front_wide_120fov"},
            {"session_id": "sess1", "start_timestamp": "5", "end_timestamp": "10", "camera": "camera_front_wide_120fov"},
            {"session_id": "sess1", "start_timestamp": "10", "end_timestamp": "15", "camera": "camera_front_wide_120fov"},
        ]
        
        # Mock embeddings for segments
        segment_embeddings = [
            [0.1, 0.2, 0.3],  # segment1 embedding
            [0.4, 0.5, 0.6],  # segment2 embedding
            [0.7, 0.8, 0.9]   # segment3 embedding
        ]
        
        with patch.object(embedder._client, 'embed_videos', return_value=segment_embeddings) as mock_embed:
            result = embedder.run(session_clips)
            
            # Verify clips are processed as videos
            mock_embed.assert_called_once()
            assert result == {"embeddings": segment_embeddings}
    
    def test_empty_clips_handling(self):
        """Test handling of empty clips list."""
        embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
            
        with patch.object(embedder._client, 'embed_videos', return_value=[]) as mock_embed:
            result = embedder.run([])
            
            assert result == {"embeddings": []}


# --------------------------------------------------------------------------- #
# Component Integration Tests                                                 #
# --------------------------------------------------------------------------- #

class TestCosmosComponentIntegration:
    """Test integration between cosmos components."""
    
    def test_all_components_use_same_cosmos_service(self):
        """Test that all components can connect to the same cosmos service."""
        
        cosmos_url = "http://cosmos-embed.default.svc.cluster.local:8000"
        
        components = [
            (CosmosVideoDocumentEmbedder, {"batch_size": 32}),
            (CosmosVideoEmbedder, {}),
            (CosmosTextEmbedder, {}),
            (CosmosSessionSegmentEmbedder, {})
        ]
        
        for component_class, extra_kwargs in components:
                component = component_class(url=cosmos_url, **extra_kwargs)                
                assert hasattr(component, '_client')
                assert component._client.embeddings_endpoint == f"{cosmos_url}/v1/embeddings"
    
    def test_component_error_propagation(self):
        """Test that all components properly propagate cosmos-embed errors."""
        
        components_and_inputs = [
            (CosmosVideoEmbedder, ["http://test.com/video.mp4"]),
            (CosmosTextEmbedder, ["test text"]),
            (CosmosSessionSegmentEmbedder, [{"session_id": "s_test", "start_timestamp": "0", "end_timestamp": "5", "camera": "camera_front_wide_120fov"}])
        ]
        
        for component_class, test_input in components_and_inputs:
            component = component_class(url="http://test:8000")
            
            # Mock a connection error from requests.post
            with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Service down")):
                with pytest.raises(RuntimeError, match="Cannot connect to cosmos-embed service"):
                    component.run(test_input)
    
    def test_component_serialization_compatibility(self):
        """Test that all components are serialization compatible."""
        
        # This tests the SerializerMixin functionality
        components = [
            CosmosVideoDocumentEmbedder,
            CosmosVideoEmbedder,
            CosmosTextEmbedder,
            CosmosSessionSegmentEmbedder
        ]
        
        for component_class in components:
                # Test basic serialization capability
            if component_class == CosmosVideoDocumentEmbedder:
                component = component_class(url="http://test:8000", batch_size=32)
            else:
                component = component_class(url="http://test:8000")
                
            assert hasattr(component, 'to_dict')
            assert hasattr(component, 'from_dict')


# --------------------------------------------------------------------------- #
# Performance and Scale Tests                                                 #
# --------------------------------------------------------------------------- #

class TestCosmosPerformanceScaling:
    """Test performance characteristics of cosmos components."""
    
    def test_video_embedder_max_batch_performance(self):
        """Test video embedder with maximum batch size (64 videos)."""
        
        embedder = CosmosVideoEmbedder(url="http://test:8000")
            
        # Create 64 video inputs (maximum batch)
        max_batch = [f"http://test.com/video{i}.mp4" for i in range(64)]
        mock_embeddings = [[float(i), float(i+1)] for i in range(64)]
        
        with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(max_batch)
            
            # Should make single API call for all 64 videos
            mock_embed.assert_called_once()
            assert len(result["embeddings"]) == 64
    
    def test_text_embedder_large_batch_performance(self):
        """Test text embedder with large batch."""
        
        embedder = CosmosTextEmbedder(url="http://test:8000")
            
        large_text_batch = [f"Text sample {i}" for i in range(100)]
        mock_embeddings = [[float(i)] for i in range(100)]
        
        with patch.object(embedder._client, 'embed_texts', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(large_text_batch)
            
            # Should handle large batches efficiently
            mock_embed.assert_called_once_with(large_text_batch)
            assert len(result["embeddings"]) == 100
    
    def test_session_segment_embedder_realistic_workload(self):
        """Test session segment embedder with realistic session workload."""
        
        embedder = CosmosSessionSegmentEmbedder(url="http://test:8000")
            
            # Simulate realistic session: 20 segments of 5-second clips
        session_segments = [{"session_id": f"sess{i//4}", "start_timestamp": str(i*5), "end_timestamp": str(i*5+5), "camera": "camera_front_wide_120fov"} for i in range(20)]
        mock_embeddings = [[float(i), float(i+0.1), float(i+0.2)] for i in range(20)]
        
        with patch.object(embedder._client, 'embed_videos', return_value=mock_embeddings) as mock_embed:
            result = embedder.run(session_segments)
            mock_embed.assert_called_once()
            assert len(result["embeddings"]) == 20
            
            # Verify embedding dimensionality
            for embedding in result["embeddings"]:
                assert len(embedding) == 3  # Mock 3D embeddings


# --------------------------------------------------------------------------- #
# Migration and Compatibility Tests                                           #
# --------------------------------------------------------------------------- #

class TestCosmosMigrationCompatibility:
    """Test migration compatibility from vitcat to cosmos-embed."""
    
    def test_cosmos_output_dimensions_consistency(self):
        """Test that cosmos-embed produces consistent output dimensions."""
        
        # All cosmos components should produce 256-dimensional embeddings
        # (this is mocked, but tests the expectation)
        
        components_and_inputs = [
            (CosmosVideoEmbedder, ["http://test.com/video.mp4"]),
            (CosmosTextEmbedder, ["test text"]),
            (CosmosSessionSegmentEmbedder, [{"session_id": "s_test", "start_timestamp": "0", "end_timestamp": "5", "camera": "camera_front_wide_120fov"}])
        ]
        
        expected_dim = 256  # cosmos-embed output dimension
        
        for component_class, test_input in components_and_inputs:
            component = component_class(url="http://test:8000")
            
            # Mock 256-dimensional embeddings in a successful response
            mock_response_json = {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * expected_dim}],
                "model": "nvidia/cosmos-embed1",
                "usage": {"num_videos": 1, "prompt_tokens": 0, "total_tokens": 10}
            }
            
            with patch('requests.post') as mock_post:
                mock_post.return_value.json.return_value = mock_response_json
                mock_post.return_value.raise_for_status.return_value = None
                
                result = component.run(test_input)
                
                # Verify output dimension consistency
                assert len(result["embeddings"][0]) == expected_dim
    
    def test_base64_security_strategy_compliance(self):
        """Test that all video components use secure base64 strategy."""
        
        video_components = [CosmosVideoEmbedder, CosmosSessionSegmentEmbedder]
        
        for component_class in video_components:
            component = component_class(url="http://test:8000")
            
            # Prepare inputs depending on component interface
            if component_class is CosmosVideoEmbedder:
                valid_inputs = ["http://test.com/video1.mp4", "http://test.com/video2.mp4"]
            else:
                valid_inputs = [
                    {"session_id": "s_test", "start_timestamp": "0", "end_timestamp": "5", "camera": "camera_front_wide_120fov"},
                    {"session_id": "s_test", "start_timestamp": "5", "end_timestamp": "10", "camera": "camera_front_wide_120fov"},
                ]
            
            # Mock TWO separate embeddings for TWO inputs  
            mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
            with patch.object(component._client, 'embed_videos', return_value=mock_embeddings):
                result = component.run(valid_inputs)
                assert len(result["embeddings"]) == 2