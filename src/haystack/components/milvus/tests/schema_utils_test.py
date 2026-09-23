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

"""Unit tests for schema utils models."""

import pytest
from pydantic import ValidationError

from src.haystack.components.milvus.schema_utils import DataType, MetadataField


def test_valid_metadata_field() -> None:
    field = MetadataField(
        name="validField", dtype="VARCHAR", is_partition_key=True, max_length=255
    )
    assert field.name == "validField"
    assert field.dtype == DataType.VARCHAR
    assert field.is_partition_key is True
    assert field.max_length == 255


def test_invalid_max_length_none_for_varchar() -> None:
    with pytest.raises(ValidationError) as excinfo:
        MetadataField(
            name="invalidField",
            dtype="VARCHAR",
            is_partition_key=False,
            max_length=None,
        )
    assert "max_length cannot be None for VARCHAR type" in str(excinfo.value)


def test_invalid_max_length_out_of_range() -> None:
    with pytest.raises(ValidationError) as excinfo:
        MetadataField(
            name="invalidField",
            dtype="VARCHAR",
            is_partition_key=False,
            max_length=70000,
        )
    assert "max_length must be between 1 and 65535 for VARCHAR type" in str(
        excinfo.value
    )


def test_valid_partition_key_with_int64() -> None:
    field = MetadataField(
        name="validPartitionField",
        dtype="INT64",
        is_partition_key=True,
        max_length=None,
    )
    assert field.dtype == DataType.INT64
    assert field.is_partition_key is True


def test_invalid_partition_key_with_invalid_dtype() -> None:
    with pytest.raises(ValidationError) as excinfo:
        MetadataField(
            name="invalidPartitionField",
            dtype="INT32",
            is_partition_key=True,
            max_length=None,
        )
    assert "dtype must be either INT64 or VARCHAR when is_partition_key is True" in str(
        excinfo.value
    )


def test_valid_non_partition_key_with_other_dtypes() -> None:
    field = MetadataField(
        name="nonPartitionField",
        dtype="FLOAT",
        is_partition_key=False,
        max_length=None,
    )
    assert field.dtype == DataType.FLOAT
    assert field.is_partition_key is False
