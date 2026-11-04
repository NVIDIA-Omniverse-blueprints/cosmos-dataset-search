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

"""VastDB-backed Haystack components for Cosmos Dataset Search.

Import concrete symbols from submodules to avoid circular imports::

    from src.haystack.components.vastdb.document_store import VastDocumentStore
"""

from typing import Any

__all__ = [
    "COLL_META",
    "VastDocumentStore",
    "VastDocumentWriter",
    "VastEmbeddingRetriever",
]


def __getattr__(name: str) -> Any:
    if name == "COLL_META":
        from src.haystack.components.vastdb.document_store import COLL_META

        return COLL_META
    if name == "VastDocumentStore":
        from src.haystack.components.vastdb.document_store import VastDocumentStore

        return VastDocumentStore
    if name == "VastDocumentWriter":
        from src.haystack.components.vastdb.document_writer import VastDocumentWriter

        return VastDocumentWriter
    if name == "VastEmbeddingRetriever":
        from src.haystack.components.vastdb.embedding_retriever import (
            VastEmbeddingRetriever,
        )

        return VastEmbeddingRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
