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

"""Embedding retriever for VastDB vector tables."""

from typing import Any, Dict, List, Optional

import numpy as np
from haystack import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import Document

from src.haystack.components.vastdb.document_store import VastDocumentStore


@component
class VastEmbeddingRetriever:
    """Retrieve documents from VastDB using native vector distance functions."""

    def __init__(
        self,
        *,
        document_store: VastDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        reconstruct: bool = False,
    ):
        if not isinstance(document_store, VastDocumentStore):
            raise ValueError("document_store must be an instance of VastDocumentStore")

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = int(top_k)
        self._reconstruct = reconstruct

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            reconstruct=self._reconstruct,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VastEmbeddingRetriever":
        init_parameters = data["init_parameters"]
        document_store = VastDocumentStore.from_dict(
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

    @component.output_types(documents=List[List[Document]])
    def run(
        self,
        query_embeddings: List[List[float]],
        index_name: str = "",
        top_k: Optional[int] = None,
        search_params: Dict = {},
        filters: Dict = {},
        reconstruct: Optional[bool] = None,
    ) -> Dict[str, List[List[Document]]]:
        docs = []
        for embedding in query_embeddings:
            docs.append(
                self._document_store._embedding_retrieval(
                    collection_name=index_name,
                    query_embedding=np.array(embedding),
                    filters=filters or self._filters,
                    search_params=search_params,
                    top_k=top_k or self._top_k,
                    return_embedding=reconstruct
                    if reconstruct is not None
                    else self._reconstruct,
                )
            )
        return {"documents": docs}
