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

"""Unit tests for document indexing."""

from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pytest

from src.haystack.components.milvus.document_store import ID_FIELD, VECTOR_FIELD
from src.haystack.components.milvus.schema_utils import MetadataConfig, MetadataField
from src.visual_search.v1.apis.document_indexing import create_parquet_table


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
            VECTOR_FIELD: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        }
    )


def check_fixed_fields(table: pa.Table):
    assert ID_FIELD in table.column_names
    assert table.field(ID_FIELD).type == pa.string()
    assert VECTOR_FIELD in table.column_names
    assert table.field(VECTOR_FIELD).type == pa.list_(pa.float32())


@pytest.mark.parametrize("id_cols", [["col1", "col2"], None])
def test_create_parquet_table_dynamic(
    id_cols: Optional[List[str]], df: pd.DataFrame
) -> None:
    table = create_parquet_table(
        df,
        id_cols=["col1", "col2"],
        embeddings_col=VECTOR_FIELD,
        metadata_cols=["col1", "col2"],
        fillna=False,
        metadata_config=MetadataConfig(
            allow_dynamic_schema=True,
            fields=[],
        ),
    )

    check_fixed_fields(table)
    assert "$meta" in table.column_names
    assert table.field("$meta").type == pa.string()
    assert len(table.column_names) == 3


@pytest.mark.parametrize("id_cols", [["col1", "col2"], None])
def test_create_parquet_table_static(
    id_cols: Optional[List[str]], df: pd.DataFrame
) -> None:
    table = create_parquet_table(
        df,
        id_cols=["col1", "col2"],
        embeddings_col=VECTOR_FIELD,
        metadata_cols=["col1", "col2"],
        fillna=False,
        metadata_config=MetadataConfig(
            allow_dynamic_schema=False,
            fields=[
                MetadataField(
                    name="col1",
                    dtype="INT32",
                ),
                MetadataField(
                    name="col2",
                    dtype="VARCHAR",
                    max_length=100,
                ),
            ],
        ),
    )

    check_fixed_fields(table)
    assert "col1" in table.column_names
    assert table.field("col1").type == pa.int32()
    assert "col2" in table.column_names
    assert table.field("col2").type == pa.string()
    assert "$meta" not in table.column_names
    assert len(table.column_names) == 4


@pytest.mark.parametrize("id_cols", [["col1", "col2"], None])
def test_create_parquet_table_mixed(
    id_cols: Optional[List[str]], df: pd.DataFrame
) -> None:
    table = create_parquet_table(
        df,
        id_cols=["col1", "col2"],
        embeddings_col=VECTOR_FIELD,
        metadata_cols=["col1", "col2"],
        fillna=False,
        metadata_config=MetadataConfig(
            allow_dynamic_schema=True,
            fields=[
                MetadataField(
                    name="col1",
                    dtype="INT32",
                ),
            ],
        ),
    )

    check_fixed_fields(table)
    assert "col1" in table.column_names
    assert "$meta" in table.column_names
    assert table.field("$meta").type == pa.string()
    assert len(table.column_names) == 4
