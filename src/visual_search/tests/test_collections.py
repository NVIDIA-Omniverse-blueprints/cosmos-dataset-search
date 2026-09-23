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

# Copyright (C) 2025, NVIDIA CORPORATION.
"""
End‑to‑end tests for the Collections API.

A lightweight in‑memory FakeStore is injected (via an autouse fixture) so that
`db_models._global_store()` resolves without touching Milvus or a Haystack
pipeline graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.visual_search.tests.conftest import client, FakeStore
from src.visual_search.common import db_models, models


# --------------------------------------------------------------------------- #
# Fixture: fake global pipeline + fake store                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def mock_global_pipeline_store(monkeypatch):
    """
    • Provide a dummy `enabled_pipelines` with ONE entry ("test‑pipeline").
    • Patch BOTH `get_document_stores` references (in pipelines **and**
      db_models) to return the same in‑memory FakeStore.
    """

    fake_store = FakeStore()

    # --------------------------------------------------------------------- #
    # Patch pipelines‑side                                                  #
    # --------------------------------------------------------------------- #
    import src.visual_search.common.pipelines as _pl

    monkeypatch.setattr(
        _pl, "enabled_pipelines", {"test-pipeline": MagicMock()}, raising=False
    )
    monkeypatch.setattr(
        _pl,
        "get_document_stores",
        lambda _pipeline: [fake_store],
        raising=False,
    )

    # --------------------------------------------------------------------- #
    # Patch db_models‑side (they captured the symbols at import time)       #
    # --------------------------------------------------------------------- #
    import src.visual_search.common.db_models as _db

    monkeypatch.setattr(_db, "enabled_pipelines", _pl.enabled_pipelines, raising=False)
    monkeypatch.setattr(
        _db,
        "get_document_stores",
        lambda _pipeline: [fake_store],
        raising=False,
    )

    yield  # run the tests


# --------------------------------------------------------------------------- #
#                         CREATE                                              #
# --------------------------------------------------------------------------- #
def test_create_collection_success():
    response = client.post(
        "/collections",
        json={
            "pipeline": "test-pipeline",
            "name": "test-name",
            "collection_config": {},
            "index_config": {},
            "tags": {"key": "value"},
            "metadata_config": {},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["collection"]["pipeline"] == "test-pipeline"
    assert body["collection"]["name"] == "test-name"
    assert body["collection"]["tags"] == {"key": "value"}


def test_create_collection_with_id():
    response = client.post(
        "/collections",
        json={
            "pipeline": "test-pipeline",
            "name": "test-name",
            "tags": {"a": "b"},
            "init_params": {
                "collection_config": {},
                "index_config": {},
                "metadata_config": {},
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["collection"]["tags"] == {"a": "b"}


def test_create_collection_invalid_pipeline():
    response = client.post(
        "/collections",
        json={"pipeline": "invalid", "name": "foo"},
    )
    assert response.status_code == 400
    assert "Pipeline 'invalid' not loaded" in response.json()["detail"]


# --------------------------------------------------------------------------- #
#                         LIST                                                #
# --------------------------------------------------------------------------- #
def test_get_collections():
    # add two
    for n in ("one", "two"):
        client.post(
            "/collections",
            json={
                "pipeline": "test-pipeline",
                "name": f"name-{n}",
                "collection_config": {},
                "index_config": {},
                "metadata_config": {},
            },
        )

    res = client.get("/collections")
    assert res.status_code == 200
    assert len(res.json()["collections"]) == 2


# --------------------------------------------------------------------------- #
#             GET / PATCH / DELETE single collection                          #
# --------------------------------------------------------------------------- #
def test_get_collection_success():
    created = client.post(
        "/collections",
        json={
            "pipeline": "test-pipeline",
            "name": "single",
            "collection_config": {},
            "index_config": {},
            "metadata_config": {},
        },
    ).json()["collection"]

    res = client.get(f"/collections/{created['id']}")
    assert res.status_code == 200
    assert res.json()["collection"]["id"] == created["id"]


def test_update_collection_success():
    created = client.post(
        "/collections",
        json={
            "pipeline": "test-pipeline",
            "name": "old",
            "collection_config": {},
            "index_config": {},
            "metadata_config": {},
        },
    ).json()["collection"]

    upd = client.patch(
        f"/collections/{created['id']}",
        json={"name": "new-name", "tags": {"z": "y"}},
    )
    assert upd.status_code == 200
    body = upd.json()
    assert body["collection"]["name"] == "new-name"
    assert body["collection"]["tags"] == {"z": "y"}


def test_update_collection_persistence():
    """Test that collection updates are properly persisted to the database."""
    # Create a collection
    created = client.post(
        "/collections",
        json={
            "pipeline": "test-pipeline",
            "name": "original-name",
            "collection_config": {},
            "index_config": {},
            "metadata_config": {},
        },
    ).json()["collection"]
    collection_id = created["id"]
    original_name = created["name"]
    original_tags = created.get("tags", {})

    # Update the collection
    updated_name = "updated-name"
    updated_tags = {"test": "unit", "persistence": "verified"}
    
    upd = client.patch(
        f"/collections/{collection_id}",
        json={"name": updated_name, "tags": updated_tags},
    )
    assert upd.status_code == 200
    body = upd.json()
    assert body["collection"]["name"] == updated_name
    assert body["collection"]["tags"] == updated_tags
    
    # Verify persistence by fetching the collection again
    get_resp = client.get(f"/collections/{collection_id}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()["collection"]
    assert fetched["name"] == updated_name
    assert fetched["tags"] == updated_tags
    assert fetched["name"] != original_name
    assert fetched["tags"] != original_tags
    
    # Verify the collection appears in list with updated data
    list_resp = client.get("/collections")
    assert list_resp.status_code == 200
    collections = list_resp.json()["collections"]
    updated_collection = next((c for c in collections if c["id"] == collection_id), None)
    assert updated_collection is not None
    assert updated_collection["name"] == updated_name
    assert updated_collection["tags"] == updated_tags


def test_delete_collection_success():
    created = client.post(
        "/collections",
        json={
            "pipeline": "test-pipeline",
            "name": "todel",
            "collection_config": {},
            "index_config": {},
            "metadata_config": {},
        },
    ).json()["collection"]

    res = client.delete(f"/collections/{created['id']}")
    assert res.status_code == 200
    assert res.json()["id"] == created["id"]


def test_get_collection_not_found():
    res = client.get("/collections/non-existent")
    assert res.status_code == 404


def test_update_collection_not_found():
    res = client.patch("/collections/non-existent", json={"name": "boom"})
    assert res.status_code == 404


def test_delete_collection_not_found():
    res = client.delete("/collections/non-existent")
    assert res.status_code == 404


# --------------------------------------------------------------------------- #
#                         db_model internals                                  #
# --------------------------------------------------------------------------- #S

def test_get_collections_empty_string_returns_404():
    with pytest.raises(Exception) as excinfo:
        db_models.get_collections("")
    assert "Collection not found" in str(excinfo.value)


def test_get_collections_empty_list_returns_404():
    with pytest.raises(Exception) as excinfo:
        db_models.get_collections([])
    assert "Collection not found" in str(excinfo.value)


def test_get_collections_with_single_id_string():
    created = db_models.create_collection(
        models.Collection(
            id="by-id-string",
            created_at="2024-01-01T00:00:00Z",
            pipeline="test-pipeline",
            name="by-id-string",
            collection_config={},
            index_config={},
            metadata_config={},
        )
    )
    result = db_models.get_collections(created.id)
    assert result.id == created.id
    assert result.name == "by-id-string"


def test_get_collections_with_single_id_list():
    created = db_models.create_collection(
        models.Collection(
            id="by-id-string",
            created_at="2024-01-01T00:00:00Z",
            pipeline="test-pipeline",
            name="by-id-list",
            collection_config={},
            index_config={},
            metadata_config={},
        )
    )
    result = db_models.get_collections([created.id])
    assert result.id == created.id
    assert result.name == "by-id-list"


def test_get_collections_with_two_ids_returns_first():
    created1 = db_models.create_collection(
        models.Collection(
            id="by-id-string-1",
            created_at="2024-01-01T00:00:00Z",
            pipeline="test-pipeline",
            name="first",
            collection_config={},
            index_config={},
            metadata_config={},
        )
    )
    created2 = db_models.create_collection(
        models.Collection(
            id="by-id-string-2",
            created_at="2024-01-01T00:00:00Z",
            pipeline="test-pipeline",
            name="second",
            collection_config={},
            index_config={},
            metadata_config={},
        )
    )
    result = db_models.get_collections([created1.id, created2.id])
    assert result.id == created1.id
    assert result.name == "first"
