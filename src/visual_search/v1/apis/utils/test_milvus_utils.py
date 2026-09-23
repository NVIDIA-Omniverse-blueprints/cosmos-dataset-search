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

# pylint: disable=import-error,missing-module-docstring,invalid-name,line-too-long
import sys
import types
from unittest.mock import call, patch

import pyarrow as pa
import pytest  # type: ignore


class _StubParquetFile:
    """Minimal replacement for `pyarrow.parquet.ParquetFile` used by tests."""

    def __init__(self, *_args, **_kwargs):
        pass

    # helper to build a one-row RecordBatch
    def _batch(self, columns, dim: int = 3):
        col = columns[0] if columns else "vector"
        arr = pa.array([[float(i) for i in range(dim)]])
        return pa.record_batch([arr], names=[col])

    # **new** API used in production code
    def iter_batches(self, columns=None, batch_size=None):
        yield self._batch(columns)

    # keep the **old** API around so nothing breaks if someone still calls it
    def read_row_groups(self, *_a, columns=None, **_k):
        return self._batch(columns)


# pyarrow & pyarrow.parquet -----------------------------------------------------
if "pyarrow" not in sys.modules:
    pa_stub = types.ModuleType("pyarrow")
    pa_parquet_stub = types.ModuleType("pyarrow.parquet")

    def _unimplemented_read_schema(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("pyarrow.parquet.read_schema was not patched")

    pa_parquet_stub.read_schema = _unimplemented_read_schema  # type: ignore[attr-defined]

    # Provide a default minimal ParquetFile stub to satisfy calls when tests
    # don't explicitly patch it.  It returns a single-row table with a vector
    # of length 3 so that the embedding-dimension check passes by default.

    class _DefaultPF:  # pylint: disable=too-few-public-methods
        def __init__(self, _file):  # noqa: D401
            pass

        def read_row_groups(self, _row_groups, columns=None, n_rows=None):  # noqa: D401
            class _Table:
                def __init__(self, col_name):
                    self._name = col_name

                def column(self, _idx):  # noqa: D401
                    return [[0, 1, 2]]

            return _Table(columns[0] if columns else "vector")

    pa_parquet_stub.ParquetFile = _DefaultPF  # type: ignore[attr-defined]
    pa_stub.parquet = pa_parquet_stub  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "pyarrow": pa_stub,
            "pyarrow.parquet": pa_parquet_stub,
        }
    )

# pymilvus ----------------------------------------------------------------------
if "pymilvus" not in sys.modules:
    pm_stub = types.ModuleType("pymilvus")

    class _DummyConnections:  # pragma: no cover – replaced in tests
        def has_connection(self, _alias):
            return False

        def connect(self, *args, **kwargs):
            pass

    pm_stub.Collection = object  # type: ignore
    pm_stub.CollectionSchema = object  # type: ignore
    pm_stub.connections = _DummyConnections()  # type: ignore
    pm_stub.utility = types.SimpleNamespace(has_collection=lambda _name: True)  # type: ignore

    class MilvusException(Exception):
        pass

    pm_stub.MilvusException = MilvusException  # type: ignore
    sys.modules["pymilvus"] = pm_stub

# -----------------------------------------------------------------------------
# Now it is safe to import the module under test.
# -----------------------------------------------------------------------------
from src.visual_search.v1.apis.utils import milvus_utils as utils  # noqa: E402

# -----------------------------------------------------------------------------
# Helpers used across multiple tests
# -----------------------------------------------------------------------------


class DummyField:
    """Minimal stub mimicking pymilvus FieldSchema."""

    def __init__(self, name: str, dim: int | None = None):
        self.name = name
        self.params = {"dim": dim} if dim is not None else {}


class DummySchema:
    """Stub for pymilvus CollectionSchema."""

    def __init__(self, field_names, dim: int | None = None):
        self.fields = [
            DummyField(n, dim=dim if n == "vector" else None) for n in field_names
        ]


class DummyCollection:
    """Stub for pymilvus Collection that only exposes .schema."""

    def __init__(self, collection_name, using=None, dim: int = 3):  # noqa: D401
        self.schema = DummySchema(["id", "vector"], dim=dim)


class DummyParquetSchema:
    """Stub for pyarrow.parquet schema that exposes .names."""

    def __init__(self, names):
        self.names = names


class DummyS3ImportCtx:
    """Context manager returned by patched open_s3_import."""

    def __enter__(self):  # noqa: D401
        return object()

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False  # propagate exceptions


# -----------------------------------------------------------------------------
# create_safe_name ----------------------------------------------------------------
# -----------------------------------------------------------------------------


def test_create_safe_name_basic():
    assert utils.create_safe_name("my-collection") == "my_collection"


def test_create_safe_name_leading_digit():
    assert utils.create_safe_name("123abc") == "a123abc"


def test_create_safe_name_empty_raises():
    with pytest.raises(ValueError):
        utils.create_safe_name("")


def test_create_safe_name_length_trim():
    long_name = "a" * 300
    safe = utils.create_safe_name(long_name)
    assert len(safe) == 255
    assert safe == "a" * 255


# -----------------------------------------------------------------------------
# get_milvus_connection_details -------------------------------------------------
# -----------------------------------------------------------------------------


def test_get_milvus_connection_details_plain():
    details = utils.get_milvus_connection_details("localhost:19530")
    assert details == {"host": "localhost", "port": 19530}


def test_get_milvus_connection_details_with_scheme():
    details = utils.get_milvus_connection_details("tcp://milvus.example.com:19130")
    assert details == {"host": "milvus.example.com", "port": 19130}


def test_get_milvus_connection_details_invalid():
    with pytest.raises(ValueError):
        utils.get_milvus_connection_details("missing_port")


# -----------------------------------------------------------------------------
# build_storage_options ---------------------------------------------------------
# -----------------------------------------------------------------------------


def test_build_storage_options_no_creds():
    assert utils.build_storage_options() == {}


def test_build_storage_options_key_secret_only():
    opts = utils.build_storage_options("AKIA", "SECRET")
    assert opts == {"key": "AKIA", "secret": "SECRET"}


def test_build_storage_options_with_endpoint():
    opts = utils.build_storage_options("AKIA", "SECRET", "https://s3.example.com")
    assert opts == {
        "key": "AKIA",
        "secret": "SECRET",
        "client_kwargs": {"endpoint_url": "https://s3.example.com"},
    }


# -----------------------------------------------------------------------------
# ensure_milvus_connection ------------------------------------------------------
# -----------------------------------------------------------------------------


def test_ensure_milvus_connection_already_connected():
    with patch.object(
        utils.connections, "has_connection", return_value=True
    ) as mock_has, patch.object(utils.connections, "connect") as mock_connect:
        utils.ensure_milvus_connection("localhost:19530", alias="test")
        mock_has.assert_called_once_with("test")
        mock_connect.assert_not_called()


def test_ensure_milvus_connection_first_time():
    with patch.object(
        utils.connections, "has_connection", return_value=False
    ), patch.object(utils.connections, "connect") as mock_connect, patch.object(
        utils,
        "get_milvus_connection_details",
        return_value={"host": "localhost", "port": 19530},
    ) as mock_parse, patch(
        "src.visual_search.v1.apis.utils.milvus_utils.utility.has_collection",
        return_value=True,
    ):
        utils.ensure_milvus_connection("localhost:19530", alias="test")
        mock_parse.assert_called_once_with("localhost:19530")
        mock_connect.assert_called_once_with(alias="test", host="localhost", port=19530)


def test_ensure_milvus_connection_connect_error():
    with patch.object(
        utils.connections, "has_connection", return_value=False
    ), patch.object(
        utils.connections, "connect", side_effect=utils.MilvusException("boom")
    ), patch.object(
        utils,
        "get_milvus_connection_details",
        return_value={"host": "localhost", "port": 19530},
    ):
        with pytest.raises(utils.MilvusServiceError):
            utils.ensure_milvus_connection("localhost:19530", alias="test")


# -----------------------------------------------------------------------------
# validate_parquet_schema -------------------------------------------------------
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_parquet_schema_success():
    with (
        patch.object(utils, "ensure_milvus_connection"),
        patch.object(utils, "Collection", DummyCollection),
        patch.object(utils, "open_s3_import", return_value=DummyS3ImportCtx()) as mock_open,
        patch.object(
            utils.pq,
            "read_schema",
            return_value=DummyParquetSchema(["id", "vector", "extra"]),
        ),
        patch.object(utils.pq, "ParquetFile", _StubParquetFile),
    ):
        # Should not raise
        await utils.validate_parquet_schema("s3://bucket/file.parquet", "collection")
        # Both schema and embedding-dimension reads must use the guarded opener.
        assert mock_open.call_args_list == [
            call("s3://bucket/file.parquet", {}),
            call("s3://bucket/file.parquet", {}),
        ]


@pytest.mark.asyncio
async def test_validate_parquet_schema_missing_fields():
    with patch.object(utils, "ensure_milvus_connection"), patch.object(
        utils, "Collection", DummyCollection
    ), patch.object(utils, "open_s3_import", return_value=DummyS3ImportCtx()), patch.object(
        utils.pq, "read_schema", return_value=DummyParquetSchema(["id"])
    ):
        with pytest.raises(utils.MilvusServiceError):
            await utils.validate_parquet_schema(
                "s3://bucket/file.parquet", "collection"
            )


@pytest.mark.asyncio
async def test_validate_parquet_schema_file_not_found():
    with patch.object(utils, "ensure_milvus_connection"), patch.object(
        utils, "Collection", DummyCollection
    ), patch.object(utils, "open_s3_import", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            await utils.validate_parquet_schema(
                "s3://bucket/missing.parquet", "collection"
            )


# -----------------------------------------------------------------------------
# Embedding dimension validation
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_parquet_schema_embedding_dim_match():
    class _PFMatch(_StubParquetFile):
        """Return a 3-dim embedding so the dimension check passes."""

        def iter_batches(self, columns=None, batch_size=None):
            yield self._batch(columns, dim=3)

    with (
        patch.object(utils, "ensure_milvus_connection"),
        patch.object(
            utils, "Collection", lambda *a, **k: DummyCollection(*a, **k, dim=3)
        ),
        patch.object(utils, "open_s3_import", return_value=DummyS3ImportCtx()),
        patch.object(
            utils.pq, "read_schema", return_value=DummyParquetSchema(["id", "vector"])
        ),
        patch.object(utils.pq, "ParquetFile", _PFMatch),  # <-- NEW
    ):
        await utils.validate_parquet_schema("s3://bucket/file.parquet", "collection")


@pytest.mark.asyncio
async def test_validate_parquet_schema_embedding_dim_mismatch():
    class PFMis(DummyParquetFile):
        def read_row_groups(self, _row_groups, columns=None, n_rows=None):  # noqa: D401
            class _Table:
                def column(self, _idx):
                    return [[0.0, 1.0, 2.0]]  # 3-dim

            return _Table()

    with patch.object(utils, "ensure_milvus_connection"), patch.object(
        utils, "Collection", lambda *a, **k: DummyCollection(*a, **k, dim=4)
    ), patch.object(utils, "open_s3_import", return_value=DummyS3ImportCtx()), patch.object(
        utils.pq, "read_schema", return_value=DummyParquetSchema(["id", "vector"])
    ), patch.object(
        utils.pq, "ParquetFile", PFMis
    ):
        with pytest.raises(utils.MilvusServiceError):
            await utils.validate_parquet_schema(
                "s3://bucket/file.parquet", "collection"
            )


# -----------------------------------------------------------------------------
# Collections with embedding dimension
# -----------------------------------------------------------------------------


class DummyParquetFile:
    """Stub for pq.ParquetFile returning a single-row table."""

    def __init__(self, _file):  # noqa: D401
        pass

    def read_row_groups(self, _row_groups, columns=None, n_rows=None):  # noqa: D401
        # The result needs to have a .column method which returns a sequence
        class _Table:
            def __init__(self, cols):
                self._cols = cols

            def column(self, idx):
                if isinstance(idx, int):
                    # Return first element of stored list for embedding
                    col_name = list(self._cols.keys())[idx]
                else:
                    col_name = idx
                return [self._cols[col_name][0]]

        # Build single-row embedding data
        emb_len = len(columns) if columns else 3  # default 3-dim
        emb = list(range(emb_len))  # e.g., [0,1,2]
        cols = {
            (columns[0] if columns else "vector"): [emb],
        }
        return _Table(cols)
