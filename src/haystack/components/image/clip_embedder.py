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

"""Haystack components for embedding images and texts with CLIP."""

import logging
from typing import Dict, List

import numpy as np
import numpy.typing as npt
from haystack import Document, component

from src.haystack.serializer import SerializerMixin
from src.triton.model_repository.clip.client import TritonClipClient


class TritonServerMixin:
    def __init__(
        self,
        url: str = "localhost:8000",
        model_name: str = "clip",
        enable_nvcf_http: bool = False,
        enable_nvcf_grpc: bool = False,
        num_threads: int = 100,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize a component with `TritonServerMixin`.

        You can start the model server locally with:
        """
        self.url = url
        self.model_name = model_name
        self.enable_nvcf_http = enable_nvcf_http
        self.enable_nvcf_grpc = enable_nvcf_grpc
        self.num_threads = num_threads
        self.batch_size = batch_size
        self._client = TritonClipClient(
            url=url,
            model_name=model_name,
            enable_nvcf_grpc=enable_nvcf_grpc,
            enable_nvcf_http=enable_nvcf_http,
            num_threads=num_threads,
            batch_size=batch_size,
        )

    def _encode_images(self, images: List[bytes]) -> List[List[float]]:
        result: npt.NDArray[np.float32] = self._client.encode_images(images)
        result /= np.linalg.norm(result, axis=-1, keepdims=True)
        return result.tolist()

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        result: npt.NDArray[np.float32] = self._client.encode_texts(texts)
        result /= np.linalg.norm(result, axis=-1, keepdims=True)
        return result.tolist()


@component
class CLIPTextEmbedder(SerializerMixin, TritonServerMixin):
    """
    A component for embedding text using NVCLIP.
    """

    @component.output_types(embeddings=List[List[float]])
    def run(self, texts: str | List[str]) -> Dict[str, List[List[float]]]:
        if isinstance(texts, str):
            texts = [texts]
        return {"embeddings": self._encode_texts(texts)}


@component
class CLIPImageEmbedder(SerializerMixin, TritonServerMixin):
    """
    A component for embedding image using NVCLIP.
    """

    @component.output_types(embeddings=List[List[float]])
    def run(self, images: bytes | List[bytes]) -> Dict[str, List[List[float]]]:
        if isinstance(images, bytes):
            images = [images]
        return {"embeddings": self._encode_images(images)}


@component
class CLIPTextDocumentEmbedder(SerializerMixin, TritonServerMixin):
    """
    A component for embedding text documents using NVCLIP.
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        embeddings = self._encode_texts([doc.content or "" for doc in documents])
        if len(embeddings) != len(documents):
            raise RuntimeError(
                f"Embeddings batch size {len(embeddings)} does not match "
                f"document batch size {len(documents)}."
            )
        for doc, emb in zip(documents, embeddings):
            doc.content, doc.blob = None, None
            doc.embedding = emb
        return {"documents": documents}


@component
class CLIPImageDocumentEmbedder(SerializerMixin, TritonServerMixin):
    """
    A component for embedding image documents using NVCLIP.
    """

    @staticmethod
    def input_checks(doc: Document) -> bool:
        """Checks on input documents."""
        if doc.blob is None:
            msg = f"Document {doc.id} does not contain a `blob` attribute for image content."
            logging.error(msg)
            raise ValueError(msg)
        return True

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        if not documents:
            return {"documents": documents}

        assert all(self.input_checks(inp) for inp in documents)
        embeddings = self._encode_images([doc.blob.data for doc in documents])
        if len(embeddings) != len(documents):
            raise RuntimeError(
                f"Embeddings batch size {len(embeddings)} does not match "
                f"document batch size {len(documents)}."
            )
        for doc, emb in zip(documents, embeddings):
            doc.blob = None
            doc.content = None
            doc.embedding = emb
        return {"documents": documents}
