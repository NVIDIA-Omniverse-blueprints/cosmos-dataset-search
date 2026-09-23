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

"""Haystack components for document filtering models."""

from typing import Any, Dict, List

import numpy as np
from haystack import Document, component

from src.haystack.serializer import SerializerMixin


def _linear_classifier_predictions(
    embeddings: np.ndarray, clf: Dict[str, Any]
) -> np.ndarray:
    weights = clf.get("weights", clf)
    try:
        coef = np.asarray(weights["coef"], dtype=np.float32)
        intercept = np.asarray(weights["intercept"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "clf must contain linear classifier weights with coef and intercept"
        ) from exc

    if coef.ndim != 2 or intercept.ndim != 1:
        raise ValueError("clf weights must contain 2D coef and 1D intercept")
    if coef.shape[0] != 1 or intercept.shape[0] != 1:
        raise ValueError("clf weights must describe a binary linear classifier")
    if embeddings.shape[1] != coef.shape[1]:
        raise ValueError(
            "clf coef dimension "
            f"{coef.shape[1]} does not match embedding dimension "
            f"{embeddings.shape[1]}"
        )

    scores = embeddings @ coef.T + intercept
    return scores[:, 0] > 0


@component
class LinearClassifierFilter(SerializerMixin):
    """Class definition for a linear classifier which is used to filter results based on the classification score.

    The classifier weights are passed as coefficients and intercept terms.
    """

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
        clf: Dict[str, Any] = {},
    ):
        if not clf:
            return {"documents": documents}

        if not documents:
            return {"documents": []}

        if documents[0].embedding is None:
            raise TypeError("To enable filtering, please set reconstuct=True")

        all_embeddings = np.array([doc.embedding for doc in documents])
        preds = _linear_classifier_predictions(all_embeddings, clf)

        filtered_documents = []
        for doc, pred in zip(documents, preds):
            if pred:
                doc.embedding = None
                filtered_documents.append(doc)

        return {"documents": filtered_documents}
