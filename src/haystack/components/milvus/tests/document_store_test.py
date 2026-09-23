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

import tempfile
from pathlib import Path

import pytest

from src.haystack.components.milvus.document_store import (
    VECTOR_FIELD,
    MilvusDocumentStore,
)
from src.haystack.components.milvus.schema_utils import MetadataConfig, MetadataField

# Mark all tests in this module to run on the same worker (serially)
# This prevents parallel milvus-lite initialization conflicts
pytestmark = pytest.mark.xdist_group(name="milvus_serial")


@pytest.fixture
def milvus_store(tmp_path: Path):
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmp_dir_str:
        db_file = Path(tmp_dir_str) / "milvus.db"
        milvus_store = MilvusDocumentStore(
            uri=str(db_file),
            embedding_dim=768,
            similarity="dot_product",
        )
        yield milvus_store


@pytest.mark.parametrize(
    "metadata_config",
    [
        MetadataConfig(),
        MetadataConfig(
            allow_dynamic_schema=False,
            fields=[
                MetadataField(
                    name="session_id",
                    dtype="VARCHAR",
                    max_length=36,
                    is_partition_key=True,
                ),
                MetadataField(
                    name="latlon",
                    dtype="ARRAY",
                    max_capacity=2,
                    element_dtype="FLOAT",
                    is_partition_key=False,
                ),
                MetadataField(
                    name="tags",
                    dtype="ARRAY",
                    max_length=256,
                    max_capacity=10,
                    element_dtype="VARCHAR",
                    is_partition_key=False,
                ),
            ],
        ),
    ],
)
def test_create_collection(
    milvus_store: MilvusDocumentStore, metadata_config: MetadataConfig
) -> None:
    collection_name = "test_collection"
    milvus_store.create_index(
        index=collection_name,
        collection_config={
            "num_partitions": 10,
            "num_shards": 2,
            "properties": {
                "mmap.enabled": True,
            },
        },
        index_config={
            "index_type": "FLAT",
            "params": {
                "mmap.enabled": True,
            },
        },
        metadata_config=metadata_config,
    )

    # some fields are not supported by milvus-lite so only asserting a few fields
    collection = milvus_store.client.describe_collection(collection_name)
    assert collection["collection_name"] == collection_name
    # one primary key, one embedding column and some metadata columns
    assert len(collection["fields"]) == 2 + len(metadata_config.fields)

    index = milvus_store.client.describe_index(
        collection_name=collection_name,
        index_name=VECTOR_FIELD,
    )
    assert index["index_type"] == "FLAT"
    assert index["metric_type"] == "IP"

    retrieved_metadata_config = milvus_store.get_metadata_schema(
        collection_name,
    )
    assert metadata_config == retrieved_metadata_config


def test_serialization(milvus_store: MilvusDocumentStore) -> None:
    """Test serialization of milvus haystack document store."""

    component_dict = milvus_store.to_dict()
    new_component = MilvusDocumentStore.from_dict(component_dict)
    assert new_component.to_dict() == component_dict
