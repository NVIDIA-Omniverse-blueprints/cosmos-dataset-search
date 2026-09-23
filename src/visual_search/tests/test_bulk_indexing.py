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

import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.visual_search.v1.apis.bulk_indexing as bi
from src.visual_search.common.models import Collection
from src.visual_search.tests.conftest import FakeStore

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def make_store(**overrides):
    """Return a FakeStore with any overrides applied."""
    store = FakeStore()
    # Apply any overrides to the store's mock methods
    for key, value in overrides.items():
        if hasattr(store, key):
            # If it's already a MagicMock, configure it
            if isinstance(getattr(store, key), MagicMock):
                getattr(store, key).configure_mock(**{'side_effect': value} if callable(value) else {'return_value': value})
            else:
                setattr(store, key, value)
        else:
            setattr(store, key, value)
    return store


class FakeStored:
    def __init__(self, pipeline):
        self.pipeline = pipeline


class FakeState:
    def __init__(self, state, state_name="", failed_reason=None):
        self.state = state
        self.state_name = state_name
        self.failed_reason = failed_reason or ""


class FakeTask:
    def __init__(
        self,
        task_id,
        state,
        state_name="",
        failed_reason=None,
        progress=None,
        collection_name="",
        files=None,
    ):
        self.task_id = task_id
        self.state = state
        self.state_name = state_name
        # always a string for Pydantic
        self.failed_reason = failed_reason if failed_reason is not None else ""
        self.progress = progress
        self.collection_name = collection_name
        self.files = files or []


def setup_pipelines(monkeypatch, pipelines):
    # pipelines: dict of pipeline_name -> fake document-store
    pipes = {}
    for name in pipelines:
        # Create a mock index_pipeline with the necessary attributes
        index_pipeline = MagicMock()
        index_pipeline.name = name
        pipes[name] = types.SimpleNamespace(name=name, id=name, index_pipeline=index_pipeline)
    
    monkeypatch.setattr(bi, "enabled_pipelines", pipes)
    # Updated to match the fix - get_document_stores is called with index_pipeline
    # Use idx_pl.name to get the correct store from pipelines dict
    monkeypatch.setattr(
        bi, "get_document_stores", 
        lambda idx_pl: [pipelines.get(idx_pl.name)] if hasattr(idx_pl, 'name') and idx_pl.name in pipelines else []
    )
    monkeypatch.setattr(bi, "create_safe_name", lambda name: name)
    monkeypatch.setattr(bi, "build_storage_options", lambda ak, sk, eu: {})

    # avoid real Milvus connection/schema checks
    async def _noop_validate_parquet_schema(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bi, "validate_parquet_schema", _noop_validate_parquet_schema)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def clear_s3_endpoint_policy(monkeypatch):
    """Do not let developer or CI endpoint settings authorize a test request."""
    for variable in (
        "CDS_FETCH_S3_ENDPOINTS",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(bi.router)
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_get_collections(monkeypatch):
    def mock_get(coll_id):
        return Collection(
            id=coll_id, pipeline="coll", name="test", created_at=datetime.utcnow()
        )

    monkeypatch.setattr(
        "src.visual_search.v1.apis.bulk_indexing.get_collections", mock_get
    )
    return mock_get


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_insert_data_success(monkeypatch, client, mock_get_collections):  # noqa: ARG001
    ds = make_store(bulk_insert_files=MagicMock(side_effect=[1, 2]))
    setup_pipelines(monkeypatch, {"coll": ds})
    monkeypatch.setenv("CDS_FETCH_S3_ENDPOINTS", "http://example.com")

    payload = {
        "collection_name": "coll",
        "parquet_paths": ["s3://test-bucket/file1.parquet", "s3://test-bucket/file2.parquet"],
        "access_key": "ak",
        "secret_key": "sk",
        "endpoint_url": "http://example.com",
    }
    resp = client.post("/insert-data", json=payload)
    assert resp.status_code == 202
    assert resp.json() == {
        "status": "success",
        "message": "Data insertion started",
        "job_id": "1",
    }
    ds.bulk_insert_files.assert_has_calls(
        [
            call(collection_name="coll", file_paths=["file1.parquet"]),
            call(collection_name="coll", file_paths=["file2.parquet"]),
        ]
    )


def test_insert_data_pipeline_not_found(monkeypatch, client, mock_get_collections):  # noqa: ARG001
    ds = make_store(bulk_insert_files=MagicMock())
    setup_pipelines(monkeypatch, {"x": ds})
    monkeypatch.setattr(
        "src.visual_search.v1.apis.bulk_indexing.get_collections", lambda x: None
    )

    payload = {
        "collection_name": "does_not_exist",
        "parquet_paths": ["s3://test-bucket/test.parquet"],
        "access_key": None,
        "secret_key": None,
        "endpoint_url": None,
    }
    resp = client.post("/insert-data", json=payload)
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "invalid_path",
    [
        "s3://bucket-only",
        "s3://bucket-only/",
        "s3://",
        "s3:///file.parquet",
        "s3://[invalid]/file.parquet",
        "",
        "/local/path/file.parquet",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
        "https://example.com/file.parquet",
        "file.parquet",
        "simplecache::s3://test-bucket/file.parquet",
        "s3://test-bucket/file.parquet::file:///etc/passwd",
        "zip://file.parquet::s3://test-bucket/archive.zip",
        "s3://user:password@test-bucket/file.parquet",
        "s3://test-bucket:9000/file.parquet",
        "s3://test-bucket/*.parquet",
        "s3://test-bucket/file?.parquet",
        "s3://test-bucket/file[12].parquet",
        "s3://test-bucket/file.parquet?other=value",
        "s3://test-bucket/file.parquet#fragment",
        "s3://test-bucket/../file.parquet",
        "s3://test-bucket/./file.parquet",
        "s3://test-bucket/path\\file.parquet",
        " s3://test-bucket/file.parquet",
        "s3://test-bucket/file.parquet\n",
    ],
)
@pytest.mark.parametrize("valid_path_first", [False, True], ids=["invalid-only", "mixed-batch"])
def test_insert_data_invalid_s3_path(
    monkeypatch, client, mock_get_collections, invalid_path, valid_path_first
):  # noqa: ARG001
    """Reject the entire batch before even reading the first valid file."""
    ds = make_store(bulk_insert_files=MagicMock())
    setup_pipelines(monkeypatch, {"coll": ds})
    resolve_pipeline = MagicMock()
    validate_schema = AsyncMock()
    build_options = MagicMock()
    monkeypatch.setattr(bi, "_resolve_pipeline", resolve_pipeline)
    monkeypatch.setattr(bi, "validate_parquet_schema", validate_schema)
    monkeypatch.setattr(bi, "build_storage_options", build_options)
    paths = ["s3://test-bucket/valid.parquet"] if valid_path_first else []
    paths.append(invalid_path)

    resp = client.post(
        "/insert-data",
        json={"collection_name": "coll", "parquet_paths": paths},
    )

    assert resp.status_code == 400
    assert "s3://bucket/key" in resp.json()["detail"]
    resolve_pipeline.assert_not_called()
    build_options.assert_not_called()
    validate_schema.assert_not_called()
    ds.bulk_insert_files.assert_not_called()


def test_insert_data_empty_batch_rejected_before_io(monkeypatch, client):
    ds = make_store()
    setup_pipelines(monkeypatch, {"coll": ds})
    resolve_pipeline = MagicMock()
    validate_schema = AsyncMock()
    monkeypatch.setattr(bi, "_resolve_pipeline", resolve_pipeline)
    monkeypatch.setattr(bi, "validate_parquet_schema", validate_schema)

    resp = client.post(
        "/insert-data", json={"collection_name": "coll", "parquet_paths": []}
    )

    assert resp.status_code == 400
    resolve_pipeline.assert_not_called()
    validate_schema.assert_not_called()
    ds.bulk_insert_files.assert_not_called()


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://unapproved.example.com",
        "http://127.0.0.1:9000",
        "http://169.254.169.254",
        "http://[::1]:9000",
        "http://10.0.0.12:9000",
        "https://storage.example.com.attacker.example",
        "https://sub.storage.example.com",
        "http://storage.example.com",
        "https://storage.example.com:9443",
        "https://storage.example.com/other-path",
        "https://storage.example.com?redirect=internal",
        "https://storage.example.com#fragment",
        "https://user:password@storage.example.com",
    ],
)
def test_insert_data_unapproved_endpoint_rejected_before_io(
    monkeypatch, client, endpoint_url
):
    ds = make_store()
    setup_pipelines(monkeypatch, {"coll": ds})
    monkeypatch.setenv("CDS_FETCH_S3_ENDPOINTS", "https://storage.example.com")
    resolve_pipeline = MagicMock()
    validate_schema = AsyncMock()
    build_options = MagicMock()
    monkeypatch.setattr(bi, "_resolve_pipeline", resolve_pipeline)
    monkeypatch.setattr(bi, "validate_parquet_schema", validate_schema)
    monkeypatch.setattr(bi, "build_storage_options", build_options)

    resp = client.post(
        "/insert-data",
        json={
            "collection_name": "coll",
            "parquet_paths": ["s3://test-bucket/file.parquet"],
            "access_key": "ak",
            "secret_key": "sk",
            "endpoint_url": endpoint_url,
        },
    )

    assert resp.status_code == 400
    resolve_pipeline.assert_not_called()
    build_options.assert_not_called()
    validate_schema.assert_not_called()
    ds.bulk_insert_files.assert_not_called()


def test_insert_data_request_cannot_configure_its_own_endpoint(monkeypatch, client):
    ds = make_store()
    setup_pipelines(monkeypatch, {"coll": ds})
    resolve_pipeline = MagicMock()
    validate_schema = AsyncMock()
    monkeypatch.setattr(bi, "_resolve_pipeline", resolve_pipeline)
    monkeypatch.setattr(bi, "validate_parquet_schema", validate_schema)

    resp = client.post(
        "/insert-data",
        json={
            "collection_name": "coll",
            "parquet_paths": ["s3://test-bucket/file.parquet"],
            "endpoint_url": "https://storage.example.com",
        },
    )

    assert resp.status_code == 400
    assert "operator-configured" in resp.json()["detail"]
    resolve_pipeline.assert_not_called()
    validate_schema.assert_not_called()
    ds.bulk_insert_files.assert_not_called()


@pytest.mark.parametrize(
    "endpoint_variable",
    ["CDS_FETCH_S3_ENDPOINTS", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"],
)
def test_insert_data_operator_configured_endpoint_allowed(
    monkeypatch, client, mock_get_collections, endpoint_variable
):  # noqa: ARG001
    """An explicitly configured private MinIO endpoint remains supported."""
    ds = make_store(bulk_insert_files=MagicMock(return_value=42))
    setup_pipelines(monkeypatch, {"coll": ds})
    monkeypatch.setenv(endpoint_variable, "http://minio.internal:9000")
    validate_schema = AsyncMock()
    monkeypatch.setattr(bi, "validate_parquet_schema", validate_schema)

    resp = client.post(
        "/insert-data",
        json={
            "collection_name": "coll",
            "parquet_paths": ["s3://test-bucket/path/file.parquet"],
            "access_key": "ak",
            "secret_key": "sk",
            "endpoint_url": "http://minio.internal:9000/",
        },
    )

    assert resp.status_code == 202
    assert resp.json()["job_id"] == "42"
    validate_schema.assert_awaited_once_with(
        "s3://test-bucket/path/file.parquet", "coll", {}, ds.client._using
    )
    ds.bulk_insert_files.assert_called_once_with(
        collection_name="coll", file_paths=["path/file.parquet"]
    )


def test_insert_data_default_s3_endpoint_allowed(monkeypatch, client, mock_get_collections):  # noqa: ARG001
    ds = make_store(bulk_insert_files=MagicMock(return_value=42))
    setup_pipelines(monkeypatch, {"coll": ds})
    validate_schema = AsyncMock()
    monkeypatch.setattr(bi, "validate_parquet_schema", validate_schema)

    resp = client.post(
        "/insert-data",
        json={"collection_name": "coll", "parquet_paths": ["s3://test-bucket/file.parquet"]},
    )

    assert resp.status_code == 202
    validate_schema.assert_awaited_once_with(
        "s3://test-bucket/file.parquet", "coll", {}, ds.client._using
    )
    ds.bulk_insert_files.assert_called_once_with(
        collection_name="coll", file_paths=["file.parquet"]
    )


def test_job_status_success(monkeypatch, client):
    ds1 = make_store(
        get_bulk_insert_state=MagicMock(
            side_effect=bi.MilvusException("can't find task")
        )
    )
    state = FakeState(bi.BulkInsertState.ImportStarted, state_name="ImportStarted")
    ds2 = make_store(get_bulk_insert_state=MagicMock(return_value=state))
    setup_pipelines(monkeypatch, {"p1": ds1, "p2": ds2})

    resp = client.get("/job-status/42")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "42"
    assert data["status"] == "in_progress"
    assert data["details"] == "ImportStarted"


def test_job_status_not_found(monkeypatch, client):
    ds = make_store(
        get_bulk_insert_state=MagicMock(
            side_effect=bi.MilvusException("can't find task")
        )
    )
    setup_pipelines(monkeypatch, {"p": ds})

    resp = client.get("/job-status/100")
    assert resp.status_code == 404


def test_job_status_milvus_error(monkeypatch, client):
    ds = make_store(
        get_bulk_insert_state=MagicMock(side_effect=bi.MilvusException("fatal error"))
    )
    setup_pipelines(monkeypatch, {"p": ds})

    resp = client.get("/job-status/1")
    assert resp.status_code == 500


def test_job_status_unknown_state(monkeypatch, client):
    state = FakeState(state="UNKNOWN_STATE", state_name="UNK")
    ds = make_store(get_bulk_insert_state=MagicMock(return_value=state))
    setup_pipelines(monkeypatch, {"p": ds})

    resp = client.get("/job-status/7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unknown"
    assert data["details"] == "UNK"


def test_list_jobs_success(monkeypatch, client):
    task1 = FakeTask(
        task_id=1,
        state=bi.BulkInsertState.ImportCompleted,
        state_name="Comp",
        progress=50,
        collection_name="col1",
        files=["a"],
    )
    ds1 = make_store(list_bulk_insert_tasks=MagicMock(return_value=[task1]))

    task2 = FakeTask(
        task_id=2,
        state=999,
        state_name="St",
        failed_reason="fail",
        progress=100,
        collection_name="col2",
        files=["b", "c"],
    )
    ds2 = make_store(list_bulk_insert_tasks=MagicMock(return_value=[task2]))

    setup_pipelines(monkeypatch, {"p1": ds1, "p2": ds2})

    resp = client.get("/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) == 2
    assert any(j["job_id"] == "1" and j["status"] == "completed" for j in data)
    assert any(
        j["job_id"] == "2" and j["status"] == "unknown" and j["details"] == "fail"
        for j in data
    )


def test_list_jobs_limit_and_filter(monkeypatch, client):
    tasks = [
        FakeTask(
            task_id=i,
            state=bi.BulkInsertState.ImportPending,
            state_name="Pending",
            progress=None,
            collection_name="xyz",
            files=[],
        )
        for i in range(5)
    ]
    ds = make_store(list_bulk_insert_tasks=MagicMock(return_value=tasks))
    setup_pipelines(monkeypatch, {"pdr": ds})

    resp = client.get("/jobs?limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get("/jobs?collection_name=collname")
    assert resp.status_code == 200
    ds.list_bulk_insert_tasks.assert_called_with(limit=None, collection_name="collname")


def test_list_jobs_exception(monkeypatch, client):
    ds = make_store(
        list_bulk_insert_tasks=MagicMock(side_effect=bi.MilvusException("oops"))
    )
    setup_pipelines(monkeypatch, {"pp": ds})

    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json() == []
