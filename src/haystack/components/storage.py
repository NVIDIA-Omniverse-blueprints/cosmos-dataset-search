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
Design a StorageBase base class that defines an interface for writing a tuple a numpy float array of embeddings and a metadata dictionary into a parquet tabular file.
The StorageBase class caches each row or rows as they are fed into it, and writes a new file when each file is written when its size is greater than 1GB.

This base class is a haystack component that integrates with a haystack pipeline.
"""

import json
import logging
from io import BytesIO
from typing import Any, Dict, List

import pyarrow as pa
import requests  # type: ignore
from haystack import Document, component, default_from_dict, default_to_dict

from src.haystack.components.milvus.document_store import MilvusDocumentStore
from src.visual_search.common.models import IngestRequest

logger = logging.getLogger(__name__)


@component
class StorageClient:
    def __init__(
        self, storage_server_url: str, document_store: MilvusDocumentStore
    ) -> None:
        self.storage_server_url = storage_server_url
        self.document_store = document_store

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            storage_server_url=self.storage_server_url,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageClient":
        init_parameters = data["init_parameters"]
        document_store = MilvusDocumentStore.from_dict(
            init_parameters["document_store"]
        )
        return default_from_dict(
            cls,
            {
                **data,
                "init_parameters": {
                    **init_parameters,
                    "document_store": document_store,
                },
            },
        )

    @component.output_types(output=IngestRequest)
    def run(
        self,
        documents: List[Document],
        index_name: str,
    ) -> Dict[str, bool]:

        source_ids, embeddings, meta_json = [], [], []

        for doc in documents:
            source_ids.append(doc.id)
            embeddings.append(doc.embedding)
            meta_json.append(json.dumps(doc.meta))

        data = {
            "id": pa.array(source_ids, type=pa.string()),
            "embedding": pa.array(embeddings, pa.list_(pa.float32())),
            "$meta": pa.array(meta_json, type=pa.string()),
        }

        table = pa.Table.from_pydict(data)
        buffer = BytesIO()
        with pa.ipc.new_stream(buffer, table.schema) as writer:
            writer.write_table(table)

        buffer.seek(0)

        # create_safe_name appends 'a' letter to the UUID and changes '-' to '_'
        collection_name = index_name.replace("_", "-")[1:]

        response = requests.post(
            self.storage_server_url + "/write",
            files={"file": ("table.stream", buffer, "application/octet-stream")},
            headers={
                "Accept": "application/json",
                "Collection": collection_name,
            },
        )

        return response.json()
