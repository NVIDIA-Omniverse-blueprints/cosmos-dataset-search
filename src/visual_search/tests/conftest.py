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

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from faker import Faker
from fastapi import FastAPI
from fastapi.testclient import TestClient
from haystack import Document as HaystackDocument
from haystack import Pipeline as HaystackPipeline
from haystack import component

from src.visual_search.common.models import TopKSearch
from src.visual_search.v1.apis.collections import router as collections_router
from src.visual_search.v1.apis.search import router as search_router


# --------------------------------------------------------------------------- #
# FakeStore - A comprehensive fake MilvusDocumentStore for testing            #
# --------------------------------------------------------------------------- #
class _FakeClient:
    """Fake Milvus client with basic collection management."""

    _using = "default"

    def __init__(self):
        self._registry: Dict[str, Dict] = {}

    def list_collections(self, using=None):
        # Pretend the registry already exists
        return ["_collection_registry"] + list(self._registry.keys())


class FakeStore(MagicMock):
    """Ultra-thin fake MilvusDocumentStore for testing purposes."""

    def __init__(self, **kwargs):
        # Don't pass unexpected kwargs to parent
        super().__init__()
        self.client = _FakeClient()
        # Pre-set the methods that _safe_is_milvus checks for
        self.bulk_insert_files = MagicMock()
        self.get_bulk_insert_state = MagicMock()
        self.list_bulk_insert_tasks = MagicMock()

    def __bool__(self):
        # Always return True so "if not store:" works correctly
        return True

    # ---- registry helpers used by db_models -----------------------------
    def register_collection(self, cid, data):  # create
        if cid in self.client._registry:
            raise ValueError("exists")
        self.client._registry[cid] = data

    def update_collection_meta(self, cid, updates):  # patch
        if cid not in self.client._registry:
            raise KeyError
        self.client._registry[cid].update(updates)

    def fetch_collection_meta(self, cid):  # get
        result = []
        if isinstance(cid, list):
            for c in cid:
                if c in self.client._registry:
                    result.append(self.client._registry[c])
        return result

    def delete_collection(self, cid):  # delete
        self.client._registry.pop(cid, None)

    def list_collection_meta(self):  # list
        return list(self.client._registry.values())

    # called by /collections/<id> route for stats
    def get_embedding_count(self, _collection_name):
        return 0


# Configuration of the FastAPI application

# Create the FastAPI application and link it to the search.py router
app = FastAPI()
app.include_router(search_router)
app.include_router(collections_router)

# Create a test client
client = TestClient(app)


class HaystackResult(HaystackDocument):
    """
    Class representing a result from the Haystack search. Used in all of the
    retrieval tests.
    """

    def __init__(self):
        fake = Faker()
        self.collection_id = fake.word()
        self.score = fake.pyfloat()
        self.content = fake.word()
        self.id = fake.word()
        self.mime_type = fake.mime_type()
        self.meta = {"d": 4, "e": "5", "f": 6.0}
        self.embedding = fake.pylist(2, value_types=(float,))


def TopKSearchJson():
    """
    Utility function for creating a JSON representation of TopKSearch object.
    """

    return {
        "nb_neighbors": 1,
        "nb_probes": 1,
        "min_similarity": 0.0,
        "reconstruct": True,
    }


def TopKSearchObj():
    """
    Utility function for creating an instance of TopKSearch object.
    """

    return TopKSearch(nb_neighbors=1, nb_probes=1, min_similarity=0.0, reconstruct=True)


def RadiusSearchJson():
    """
    Utility function for creating a JSON representation of RadiusSearch object.
    """

    return {
        "min_similarity": 0.0,
        "nb_probes": 1,
        "max_results": 1,
    }


def good_retrieval_request_body(params):
    """
    Utility function for creating a retrieval request body with given parameters.
    """

    return {
        "collections": ["collection-a", "collection-b"],
        "query": [
            {
                "text": "a car",
            },
            {"text": "a house"},
        ],
        "params": params,
        "payload_keys": ["score"],
        "rerank": True,
        "num_results": 2,
    }


@pytest.fixture()
def mock_retrieval_query():
    """
    Fixture for mocking a retrieval query. This is a RetrievalRequest object,
    used in the retrieval tests.
    """

    mock_retrieval_query = MagicMock()
    mock_retrieval_query.collections = ["1", "2"]
    mock_retrieval_query.query = {"text": "a car"}
    mock_retrieval_query.params = TopKSearchObj()
    mock_retrieval_query.num_results = 10
    mock_retrieval_query.rerank = True
    mock_retrieval_query.payload_keys = None
    mock_retrieval_query.generate_asset_url = True
    yield mock_retrieval_query


@pytest.fixture()
def mock_pipeline_handler():
    """
    Fixture for mocking the pipeline handler. Used in the retrieval tests to
    return an object that has a query_pipeline method. Mocks elements of the
    search function.
    """

    with patch(
        "src.visual_search.v1.apis.search.pipeline_handler"
    ) as mock_pipeline_handler:
        mock_pipeline_handler.return_value.query_pipeline.return_value = "mocked"
        mock_pipeline_handler.return_value.query_pipeline_inputs = {}
        yield mock_pipeline_handler


@pytest.fixture()
def mock_run_query_pipeline():
    """
    Fixture for mocking the run_query_pipeline function. Mocks the
    run_query_pipeline function in the search request, returning a list of two
    HaystackResult objects. Used in the retrieval tests. Return value is
    sometimes modified in the tests.
    """

    with patch(
        "src.visual_search.v1.apis.search.run_query_pipeline"
    ) as mock_run_query_pipeline:
        mock_run_query_pipeline.return_value = [HaystackResult(), HaystackResult()]
        yield mock_run_query_pipeline


@component
class MockIndexer:
    def run(self, output: dict, index_name: str) -> dict:
        return {"output": {"status": "ok"}}


@component
class DumpIndexer:
    def run(self, documents: List[HaystackDocument], index_name: str) -> dict:
        return {
            "output": {
                "status": f"Indexed {len(documents)} documents into '{index_name}'"
            }
        }


@component
class MockRetriever:
    def run(self, query: str, top_k: int, index_name: str) -> dict:
        return [
            HaystackDocument(content="dummy result"),
            HaystackDocument(content="another dummy result"),
        ]


@component
class DumpRetriever:
    def __init__(self, config: dict):
        self.config = config

    @component.output_types(documents=List[HaystackDocument])
    def run(self, query: str, top_k: int) -> List[HaystackDocument]:
        return {
            "documents": [
                HaystackDocument(
                    content="mocked result 1",
                    meta={"score": 0.99, "collection_id": "collection-a"},
                ),
                HaystackDocument(
                    content="mocked result 2",
                    meta={"score": 0.97, "collection_id": "collection-b"},
                ),
            ]
        }


@pytest.fixture(autouse=True)
def inject_test_pipeline(monkeypatch):
    from src.visual_search.common.pipelines import (
        EnabledPipeline,
        IndexPipelineInputs,
        QueryPipelineInputs,
        enabled_pipelines,
    )

    index_pipe = HaystackPipeline()
    fake_store = FakeStore()  # Use the FakeStore class

    # Replace the helper so it never touches .graph
    import src.visual_search.common.pipelines as _pl

    monkeypatch.setattr(
        _pl,
        "get_document_stores",
        lambda _pipeline: [fake_store],
        raising=False,
    )

    # Do the same in db_models – it captured the symbol at import time
    import src.visual_search.common.db_models as _db

    monkeypatch.setattr(
        _db,
        "get_document_stores",
        lambda _pipeline: [fake_store],
        raising=False,
    )

    # Set up additional methods that might be needed
    fake_store.insert = MagicMock()
    fake_store.query = MagicMock(return_value=[])
    fake_store.get = MagicMock(return_value=[])
    fake_store.delete = MagicMock()

    indexer = DumpIndexer()
    setattr(indexer, "document_store", fake_store)
    index_pipe.add_component("indexer", indexer)

    query_pipe = HaystackPipeline()
    retriever = DumpRetriever({"index_name": "test-pipeline"})
    setattr(retriever, "_document_store", fake_store)
    query_pipe.add_component("retriever", retriever)

    enabled_pipelines["test-pipeline"] = EnabledPipeline(
        id="test-pipeline",
        config={},
        index_pipeline=index_pipe,
        index_pipeline_inputs=IndexPipelineInputs(index_name=["indexer.index_name"]),
        query_pipeline=query_pipe,
        query_pipeline_inputs=QueryPipelineInputs(
            query=["retriever.query"],
            top_k=["retriever.top_k"],
            # Note: No longer mapping 'index_name' here, as removed from component run signature
        ),
    )
    yield
    enabled_pipelines.pop("test-pipeline", None)
