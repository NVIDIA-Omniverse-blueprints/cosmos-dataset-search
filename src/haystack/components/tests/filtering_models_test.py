# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
