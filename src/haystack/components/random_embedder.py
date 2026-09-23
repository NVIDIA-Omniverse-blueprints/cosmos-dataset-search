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

from typing import Dict, List, Optional

import numpy as np
from haystack import Document, component

from src.haystack.serializer import SerializerMixin


@component
class RandomTextEmbedder(SerializerMixin):
    """
    A component for generarting random embeddings from strings.
    """

    def __init__(self, dimension: int, *, max_length: Optional[int] = None):
        self.dimension = dimension
        self.max_length = max_length

    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, List[float]]:
        if self.max_length and len(text) > self.max_length:
            raise ValueError(
                f"String to embed (len{len(text)}) exceed "
                f"max length of {self.max_length}"
            )
        embedding = np.random.rand(self.dimension).tolist()
        return {"embedding": embedding}


@component
class RandomDocumentEmbedder(SerializerMixin):
    """
    A component for embedding documents with random values.
    """

    def __init__(self, dimension: int, *, max_length: Optional[int] = None):
        self.dimension = dimension
        self.max_length = max_length

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        embeddings = np.random.rand(len(documents), self.dimension).tolist()

        for doc, emb in zip(documents, embeddings):
            if self.max_length and len(doc.content or "") > self.max_length:
                raise ValueError(
                    f"String to embed (length {len(doc.content or '')})"
                    f"exceed max length of {self.max_length}"
                )

            doc.embedding = emb

        return {"documents": documents}
