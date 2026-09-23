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

"""Custom document writer for writing to multiple collections in a thread-safe manner."""

from typing import Any, Dict, List, Optional
import logging

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DuplicatePolicy

from src.haystack.components.milvus.document_store import MilvusDocumentStore

logger = logging.getLogger(__name__)


@component
class MilvusDocumentWriter:
    def __init__(
        self,
        document_store: MilvusDocumentStore,
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        index_name: str = "",
    ):
        self.document_store = document_store
        self.policy = policy
        self.index_name = index_name

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            policy=self.policy,
            index_name=self.index_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusDocumentWriter":
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

    @component.output_types(documents_written=int)
    def run(
        self,
        documents: List[Document],
        policy: Optional[DuplicatePolicy] = None,
        index_name: str = "",
    ):
        if policy is None:
            policy = self.policy
        logger.debug("MilvusWriter received %d docs", len(documents))

        documents_written = self.document_store.write_documents(
            documents=documents,
            policy=policy,
            collection_name=index_name,
        )
        return {"documents_written": documents_written}
