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

"""Component wrappers for OpenAI Haystack components"""

import logging
import os
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.components.embedders import (
    OpenAIDocumentEmbedder as _OpenAIDocumentEmbedder,
)
from haystack.components.embedders import OpenAITextEmbedder as _OpenAITextEmbedder
from haystack.utils import Secret

from src.haystack.serializer import SerializerMixin


@component
class NVIDIADocumentEmbedder(SerializerMixin, _OpenAIDocumentEmbedder):
    """Component for using NVIDIA model catalog.

    Requires `NVIDIA_API_KEY` to be set in environment.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.embedders import OpenAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = NVIDIADocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        model: str = "nvidia/nv-embed-v1",
        dimensions: Optional[int] = None,
        api_base_url: str = "https://integrate.api.nvidia.com/v1",
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not os.getenv("NVIDIA_API_KEY"):
            logging.warning(
                "`NVIDIA_API_KEY` not found! "
                "Please provide environment variable for correct authentication to NVIDIA catalog.",
            )
        _OpenAIDocumentEmbedder.__init__(
            self,
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            model=model,
            dimensions=dimensions,
            api_base_url=api_base_url,
            organization=organization,
            prefix=prefix,
            suffix=suffix,
            batch_size=batch_size,
            progress_bar=progress_bar,
            meta_fields_to_embed=meta_fields_to_embed,
            embedding_separator=embedding_separator,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.timeout = timeout
        self.max_retries = max_retries


@component
class NVIDIATextEmbedder(SerializerMixin, _OpenAITextEmbedder):
    """Component for using NVIDIA model catalog.

    Requires `NVIDIA_API_KEY` to be set in environment.

    Usage example:
    ```python
    from haystack.components.embedders import NVIDIATextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = NVIDIATextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embeddings': [[0.017020374536514282, -0.023255806416273117, ...]],
    # 'meta': [{'model': 'nvidia/nv-embed-v1',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}]}
    ```
    """

    def __init__(
        self,
        model: str = "nvidia/nv-embed-v1",
        dimensions: Optional[int] = None,
        api_base_url: str = "https://integrate.api.nvidia.com/v1",
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not os.getenv("NVIDIA_API_KEY"):
            logging.warning(
                "`NVIDIA_API_KEY` not found! "
                "Please provide environment variable for correct authentication to NVIDIA catalog.",
            )
        _OpenAITextEmbedder.__init__(
            self,
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            model=model,
            dimensions=dimensions,
            api_base_url=api_base_url,
            organization=organization,
            prefix=prefix,
            suffix=suffix,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(embeddings=List[List[float]], meta=List[Dict[str, Any]])
    def run(self, text: str | List[str]) -> Dict[str, Any]:
        """Embed strings.

        :param text:
            Text to embed. Either a single text or multiple texts.

        :returns:
            A dictionary with the following keys:
            - `embeddings`: List of embeddings of input texts.
            - `meta`: Information about the usage of the model.
        """

        if isinstance(text, str):
            text = [text]
        embeddings, meta = [], []
        for txt in text:
            text_to_embed = self.prefix + txt + self.suffix
            # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
            # replace newlines, which can negatively affect performance.
            text_to_embed = text_to_embed.replace("\n", " ")
            if self.dimensions is not None:
                response = self.client.embeddings.create(
                    model=self.model, dimensions=self.dimensions, input=text_to_embed
                )
            else:
                response = self.client.embeddings.create(
                    model=self.model, input=text_to_embed
                )
            meta.append({"model": response.model, "usage": dict(response.usage)})
            embeddings.append(response.data[0].embedding)
        return {"embeddings": embeddings, "meta": meta}
