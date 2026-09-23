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

"""
Unit tests for document deletion fix.

Tests the enhanced delete_document function
"""

from typing import List
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException
from haystack import Document as HaystackDocument

from src.visual_search.common.models import Collection
from src.visual_search.common.pipelines import EnabledPipeline
from src.visual_search.v1.apis.document_indexing import delete_document
from src.haystack.components.milvus.document_store import MilvusDocumentStore


# --------------------------------------------------------------------------- #
# Test Fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_collection():
    """Mock Collection for testing."""
    collection = MagicMock(spec=Collection)
    collection.id = "test_collection_123"
    collection.pipeline = "cosmos_video_search_milvus"
    return collection


@pytest.fixture
def mock_pipeline():
    """Mock EnabledPipeline for testing."""
    pipeline = MagicMock(spec=EnabledPipeline)
    pipeline.id = "cosmos_video_search_milvus"
    pipeline.index_pipeline = MagicMock()  # Add the missing index_pipeline attribute
    return pipeline


@pytest.fixture
def mock_milvus_document_store():
    """Mock MilvusDocumentStore for testing."""
    store = MagicMock(spec=MilvusDocumentStore)
    return store


@pytest.fixture
def mock_non_milvus_document_store():
    """Mock non-Milvus document store for testing."""
    store = MagicMock()
    # Ensure it's not identified as Milvus
    store.__class__.__name__ = "SomeOtherDocumentStore"
    return store


# --------------------------------------------------------------------------- #
# Test Cases for Document Deletion Fix                                       #
# --------------------------------------------------------------------------- #

def test_delete_document_by_source_id_success(mock_collection, mock_pipeline, mock_milvus_document_store):
    """Test successful deletion using source_id (original behavior)."""
    # Arrange
    test_doc_id = "api_001_video_segment-test123"
    mock_document = HaystackDocument(
        id="milvus_generated_uuid_12345",
        content="test content",
        meta={"source_id": test_doc_id, "other_meta": "value"}
    )
    
    # Mock document store returns document when filtered by source_id
    mock_milvus_document_store.filter_documents.return_value = [mock_document]
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [mock_milvus_document_store]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act
        result = delete_document(
            collection=mock_collection, 
            pipeline=mock_pipeline, 
            document_id=test_doc_id
        )
        
        # Assert
        assert result.id == test_doc_id
        assert result.message == "Resource deleted successfully."
        
        # Verify the correct filter was used (source_id lookup)
        mock_milvus_document_store.filter_documents.assert_called_with(
            collection_name="safe_collection_name",
            filters={"field": "meta.source_id", "operator": "==", "value": test_doc_id}
        )
        
        # Verify deletion was called with the actual document ID
        mock_milvus_document_store.delete_documents.assert_called_once_with(
            collection_name="safe_collection_name",
            document_ids=["milvus_generated_uuid_12345"]
        )


def test_delete_document_by_document_id_success(mock_collection, mock_pipeline, mock_milvus_document_store):
    """Test successful deletion using document.id (new behavior - the fix!)."""
    # Arrange
    ui_displayed_id = "638f6f9af2d2902255881590baaf8805ac490ed5c4b39849ed537f0c6c5b3404"
    mock_document = HaystackDocument(
        id=ui_displayed_id,
        content="test content",
        meta={"source_id": "original_api_id", "other_meta": "value"}
    )
    
    # Mock document store returns empty for source_id, but finds by document.id
    mock_milvus_document_store.filter_documents.side_effect = [
        [],  # First call (source_id filter) returns empty
        [mock_document]  # Second call (document.id filter) returns the document
    ]
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [mock_milvus_document_store]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act
        result = delete_document(
            collection=mock_collection, 
            pipeline=mock_pipeline, 
            document_id=ui_displayed_id
        )
        
        # Assert
        assert result.id == ui_displayed_id
        assert result.message == "Resource deleted successfully."
        
        # Verify both filters were attempted
        assert mock_milvus_document_store.filter_documents.call_count == 2
        
        # First call: source_id lookup
        first_call = mock_milvus_document_store.filter_documents.call_args_list[0]
        assert first_call[1]["filters"]["field"] == "meta.source_id"
        assert first_call[1]["filters"]["value"] == ui_displayed_id
        
        # Second call: document.id lookup (the fix!)
        second_call = mock_milvus_document_store.filter_documents.call_args_list[1]
        assert second_call[1]["filters"]["field"] == "id"
        assert second_call[1]["filters"]["value"] == ui_displayed_id
        
        # Verify deletion was called
        mock_milvus_document_store.delete_documents.assert_called_once_with(
            collection_name="safe_collection_name",
            document_ids=[ui_displayed_id]
        )


def test_delete_document_not_found_both_lookups_fail(mock_collection, mock_pipeline, mock_milvus_document_store):
    """Test 404 error when document is not found by either lookup method."""
    # Arrange
    nonexistent_id = "does_not_exist_123"
    
    # Mock document store returns empty for both lookups
    mock_milvus_document_store.filter_documents.return_value = []
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [mock_milvus_document_store]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            delete_document(
                collection=mock_collection, 
                pipeline=mock_pipeline, 
                document_id=nonexistent_id
            )
        
        assert exc_info.value.status_code == 404
        assert "Not Found" in exc_info.value.detail
        assert nonexistent_id in exc_info.value.detail
        assert "searched both by source_id and document id" in exc_info.value.detail
        
        # Verify both lookups were attempted
        assert mock_milvus_document_store.filter_documents.call_count == 2


def test_delete_document_with_non_milvus_store(mock_collection, mock_pipeline, mock_non_milvus_document_store):
    """Test deletion works with non-Milvus document stores (backward compatibility)."""
    # Arrange
    test_doc_id = "test_doc_123"
    mock_document = HaystackDocument(
        id="some_id",
        content="test content",
        meta={"source_id": test_doc_id}
    )
    
    # Mock non-Milvus store behavior
    mock_non_milvus_document_store.filter_documents.return_value = [mock_document]
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [mock_non_milvus_document_store]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act
        result = delete_document(
            collection=mock_collection, 
            pipeline=mock_pipeline, 
            document_id=test_doc_id
        )
        
        # Assert
        assert result.id == test_doc_id
        
        # Verify non-Milvus store was called correctly (no collection_name parameter)
        mock_non_milvus_document_store.filter_documents.assert_called_with(
            filters={"field": "meta.source_id", "operator": "==", "value": test_doc_id}
        )
        mock_non_milvus_document_store.delete_documents.assert_called_once_with(
            document_ids=["some_id"]
        )


def test_delete_document_fallback_exception_handling(mock_collection, mock_pipeline, mock_milvus_document_store):
    """Test that exceptions in the document.id lookup are handled gracefully."""
    # Arrange
    test_doc_id = "test_doc_123"
    
    # First call (source_id) returns empty, second call (document.id) raises exception
    mock_milvus_document_store.filter_documents.side_effect = [
        [],  # source_id lookup returns empty
        Exception("Milvus error")  # document.id lookup fails
    ]
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [mock_milvus_document_store]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act & Assert - should get 404, not the Milvus exception
        with pytest.raises(HTTPException) as exc_info:
            delete_document(
                collection=mock_collection, 
                pipeline=mock_pipeline, 
                document_id=test_doc_id
            )
        
        assert exc_info.value.status_code == 404
        assert "Not Found" in exc_info.value.detail


def test_delete_document_no_document_store_found(mock_collection, mock_pipeline):
    """Test error when no document store is found in pipeline."""
    # Arrange - empty document stores list
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = []  # No document stores
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            delete_document(
                collection=mock_collection, 
                pipeline=mock_pipeline, 
                document_id="any_id"
            )
        
        assert exc_info.value.status_code == 501  # NOT_IMPLEMENTED
        assert "Document Store Not Found" in exc_info.value.detail
        assert mock_collection.pipeline in exc_info.value.detail


# --------------------------------------------------------------------------- #
# Integration Test Cases                                                      #
# --------------------------------------------------------------------------- #

def test_delete_document_multiple_stores_success(mock_collection, mock_pipeline):
    """Test deletion works when multiple document stores are present."""
    # Arrange
    test_doc_id = "multi_store_test_123"
    mock_document = HaystackDocument(
        id="found_id",
        content="test content",
        meta={"source_id": test_doc_id}
    )
    
    # Create multiple stores - first one empty, second one has the document
    store1 = MagicMock(spec=MilvusDocumentStore)
    store1.filter_documents.return_value = []
    
    store2 = MagicMock(spec=MilvusDocumentStore)
    store2.filter_documents.return_value = [mock_document]
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [store1, store2]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act
        result = delete_document(
            collection=mock_collection, 
            pipeline=mock_pipeline, 
            document_id=test_doc_id
        )
        
        # Assert
        assert result.id == test_doc_id
        
        # Verify both stores were queried
        store1.filter_documents.assert_called()
        store2.filter_documents.assert_called()
        
        # Verify only the store with the document performed deletion
        store1.delete_documents.assert_not_called()
        store2.delete_documents.assert_called_once()


# --------------------------------------------------------------------------- #
# Edge Cases and Error Conditions                                            #
# --------------------------------------------------------------------------- #

def test_delete_document_edge_case_empty_string_id(mock_collection, mock_pipeline, mock_milvus_document_store):
    """Test behavior with edge case inputs like empty string."""
    # Arrange
    empty_id = ""
    mock_milvus_document_store.filter_documents.return_value = []
    
    with patch('src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection') as mock_get_pipeline, \
         patch('src.visual_search.v1.apis.document_indexing.get_document_stores') as mock_get_stores, \
         patch('src.visual_search.v1.apis.document_indexing.create_safe_name') as mock_safe_name:
        
        mock_get_pipeline.return_value = mock_pipeline
        mock_get_stores.return_value = [mock_milvus_document_store]
        mock_safe_name.return_value = "safe_collection_name"
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            delete_document(
                collection=mock_collection, 
                pipeline=mock_pipeline, 
                document_id=empty_id
            )
        
        assert exc_info.value.status_code == 404
