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

from typing import Any, Dict, List, Literal, Optional

from haystack import component
from haystack.core.serialization import default_from_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from src.haystack.serializer import SerializerMixin


class NoOpDocumentStore(SerializerMixin):
    """Document store that does nothing."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        embedding_similarity_function: Optional[
            List[Literal["cosine", "inner_product", "l2"]]
        ] = None,
    ):
        pass

    def _count_documents(self) -> int:
        return 100

    def count_documents(self) -> int:
        return 100

    def filter_documents(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        return []

    def create_index(self, index: str) -> None:
        pass

    def delete_index(self, index: str) -> None:
        pass

    def use_index(self, index: str) -> None:
        self.table_name = index

    def write_documents(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        return len(documents)

    def delete_documents(self, document_ids: List[str]) -> None:
        pass

    def delete_documents_by_filter(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> int:
        pass

    def get_metadata_schema(self, index: str) -> None:
        pass


@component
class NoOpEmbeddingRetriever:
    def __init__(
        self,
        *,
        document_store: NoOpDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ):
        if not isinstance(document_store, NoOpDocumentStore):
            msg = "document_store must be an instance of NoOpDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoOpEmbeddingRetriever":
        data["init_parameters"]["document_store"] = NoOpDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self, query_embedding: List[float], top_k: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        return {"documents": []}
