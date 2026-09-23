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

"""Haystack components for routing logic."""

import base64
from typing import Any, Dict, List

from haystack import Document, component

from src.haystack.serializer import SerializerMixin
from src.visual_search.common.models import (
    EmbeddingQuery,
    EpisodeQuery,
    QueryType,
    SessionFrameQuery,
    SessionSegmentQuery,
    TextQuery,
    VideoQuery,
)
from src.visual_search.exceptions import InputValidationError


def string_to_bytes(string: str) -> bytes:
    """Convert string in base64 to raw bytes."""
    return base64.b64decode(string)


@component
class QueryTypeRouter(SerializerMixin):
    @component.output_types(
        frames=List[Dict[str, str]],
        clips=List[Dict[str, str]],
        texts=List[str],
        videos=List[str],
        episodes=List[bytes],
        embeddings=List[List[float]],
    )
    def run(self, query: QueryType) -> Dict[str, Any]:
        """Determine the type of query and route accordingly."""

        if not isinstance(query, (list, tuple)):
            query = [query]

        output: Dict[str, Any] = dict()
        for q in query:
            if isinstance(q, TextQuery):
                if "texts" not in output:
                    output["texts"] = []
                output["texts"].append(str(q.text))
            elif isinstance(q, VideoQuery):
                if "videos" not in output:
                    output["videos"] = []
                # Expect presigned URL string
                output["videos"].append(str(q.video))
            elif isinstance(q, EpisodeQuery):
                if "episodes" not in output:
                    output["episodes"] = []
                output["episodes"].append(string_to_bytes(q.episode))
            elif isinstance(q, EmbeddingQuery):
                if "embeddings" not in output:
                    output["embeddings"] = []
                output["embeddings"].append(list(q.embedding))
            elif isinstance(q, SessionSegmentQuery):
                if "clips" not in output:
                    output["clips"] = []
                output["clips"].append(q.session_segment.dict())
            elif isinstance(q, SessionFrameQuery):
                if "frames" not in output:
                    output["frames"] = []
                output["frames"].append(q.session_frame.dict())
            else:
                raise InputValidationError(f"Unsupported query type {type(q)}")
        return output


@component
class IndexTypeRouter(SerializerMixin):
    @component.output_types(
        to_index=List[Document],
        embedded=List[Document],
    )
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """Determine the type of document and route accordingly."""

        to_index, embedded = [], []

        for doc in documents:
            if doc.embedding and not doc.content:
                embedded.append(doc)
            elif doc.embedding is None and (doc.blob is not None or bool(doc.content)):
                to_index.append(doc)
            else:
                raise ValueError(
                    "Ambiguous content, cannot route to appropriate indexer! Got "
                    f"embedding={doc.embedding}, "
                    f"content={doc.content}, "
                    f"blob={doc.blob}"
                )

        output = {}
        if to_index:
            output["to_index"] = to_index
        if embedded:
            output["embedded"] = embedded
        return output
