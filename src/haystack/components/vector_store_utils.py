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

"""Helpers for the VastDB document-store call sites."""

from __future__ import annotations

from typing import Any, List, Tuple, Type


def vector_document_store_types() -> Tuple[Type[Any], ...]:
    from src.haystack.components.vastdb.document_store import VastDocumentStore

    return (VastDocumentStore,)


def is_vector_document_store(store: Any) -> bool:
    return isinstance(store, vector_document_store_types())


def list_store_collections(store: Any) -> List[str]:
    if hasattr(store, "list_collection_names"):
        return list(store.list_collection_names())
    if hasattr(store, "list_collections") and callable(store.list_collections):
        return list(store.list_collections())
    return []
