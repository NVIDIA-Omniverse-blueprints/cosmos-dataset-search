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

"""Pydantic model definitions for metadata schema."""

from enum import Enum
from typing import Any, Dict, List, Optional

import pyarrow as pa
from pydantic import BaseModel, Field, validator


class DataType(str, Enum):
    """
    Data type of metadata fields in collection.
    """

    BOOL = "BOOL"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    VARCHAR = "VARCHAR"
    JSON = "JSON"
    ARRAY = "ARRAY"


class MetadataField(BaseModel):
    """Metadata field attributes."""

    name: str = Field(description="Name of metadata field.")
    dtype: DataType = Field(description="Data type of metadata field.")
    is_partition_key: bool = Field(
        default=False,
        description="Whether field is a partition key. Supports only VARCHAR and INT64.",
    )
    element_dtype: Optional[DataType] = Field(
        default=None, description="Element data type if dtype ARRAY."
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length if dtype is VARCHAR. Between [1, 65535].",
    )
    max_capacity: Optional[int] = Field(
        default=None, description="Maximum size of array if dtype is ARRAY."
    )

    @validator("max_length", always=True)
    def validate_max_length(
        cls: "MetadataField", v: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        # Validate max_length for VARCHAR dtype or element_dtype
        dtype = values.get("dtype")
        element_dtype = values.get("element_dtype")
        if dtype == DataType.VARCHAR or element_dtype == DataType.VARCHAR:
            if v is None:
                raise ValueError("max_length cannot be None for VARCHAR type")
            if not (1 <= v <= 65535):
                raise ValueError(
                    "max_length must be between 1 and 65535 for VARCHAR type"
                )
        else:
            if v is not None:
                raise ValueError(f"max_length should be None for dtype {dtype}")
        return v

    @validator("max_capacity", always=True)
    def validate_max_capacity(
        cls: "MetadataField", v: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        # Validate max_capacity for ARRAY dtype
        dtype = values.get("dtype")
        if dtype == DataType.ARRAY:
            if v is None:
                raise ValueError("max_capacity cannot be None for ARRAY type")
        else:
            if v is not None:
                raise ValueError(f"max_capacity should be None for dtype {dtype}")
        return v

    @validator("element_dtype", always=True)
    def validate_element_dtype(
        cls: "MetadataField", v: Optional[DataType], values: Dict[str, Any]
    ) -> Optional[DataType]:
        # Validate element_type for ARRAY dtype
        dtype = values.get("dtype")
        if dtype == DataType.ARRAY:
            if v is None:
                raise ValueError("element_dtype cannot be None for ARRAY dtype")
            supported_array_dtypes = {
                DataType.BOOL,
                DataType.INT8,
                DataType.INT16,
                DataType.INT32,
                DataType.INT64,
                DataType.FLOAT,
                DataType.DOUBLE,
                DataType.VARCHAR,
                DataType.JSON,
            }
            if v not in supported_array_dtypes:
                raise ValueError(
                    f"ARRAY element_type supports {supported_array_dtypes}. Got {v}."
                )
        else:
            if v is not None:
                raise ValueError(f"element_type should be None for dtype {dtype}")
        return v

    @validator("is_partition_key")
    def validate_partition_key_dtype(
        cls: "MetadataField", v: bool, values: Dict[str, Any]
    ) -> bool:
        # Validate that partition key is of type INT64 or VARCHAR
        dtype = values.get("dtype")
        if v and dtype not in (DataType.INT64, DataType.VARCHAR):
            raise ValueError(
                "dtype must be either INT64 or VARCHAR when is_partition_key is True"
            )
        return v

    def to_pyarrow_type(self) -> pa.DataType:
        """Maps dtype to pyarrow data type."""
        type_map = {
            DataType.BOOL: pa.bool_(),
            DataType.INT8: pa.int8(),
            DataType.INT16: pa.int16(),
            DataType.INT32: pa.int32(),
            DataType.INT64: pa.int64(),
            DataType.FLOAT: pa.float32(),
            DataType.DOUBLE: pa.float64(),
            DataType.VARCHAR: pa.string(),
            DataType.JSON: pa.string(),
        }
        if self.dtype == DataType.ARRAY:
            assert self.element_dtype  # silence mypy
            return pa.list_(type_map[self.element_dtype])
        return type_map[self.dtype]


class MetadataConfig(BaseModel):
    """Schema of user metadata of collection, associated with the ingested embeddings.

    If defaults used, no static fields are specified and a dynamic schema will be used.
    This means all metadata is added into a JSON string.

    For very large collections, it is recommended to specify the metadata fields statically,
    especially if you want to pre-filter based on the metadata.

    If one wants to use a metadata field as a partition key, it is necessary to specify this.
    """

    allow_dynamic_schema: bool = True
    fields: List[MetadataField] = []
