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

"""Unit tests for Haystack filtering models."""

import pytest
from haystack import Document

from src.haystack.components.filtering_models import LinearClassifierFilter


def test_linear_classifier_filter_uses_weights() -> None:
    docs = [
        Document(content="drop", embedding=[-1.0, 0.0]),
        Document(content="keep", embedding=[1.0, 0.0]),
    ]
    clf = {
        "weights": {"coef": [[1.0, 0.0]], "intercept": [0.0]},
        "model": "legacy-model-field-is-ignored",
    }

    result = LinearClassifierFilter().run(docs, clf=clf)

    assert [doc.content for doc in result["documents"]] == ["keep"]
    assert result["documents"][0].embedding is None


def test_linear_classifier_filter_rejects_model_only_payload() -> None:
    docs = [
        Document(content="doc", embedding=[1.0, 0.0]),
    ]

    with pytest.raises(ValueError, match="weights"):
        LinearClassifierFilter().run(
            docs, clf={"model": "legacy-model-field-is-not-deserialized"}
        )
