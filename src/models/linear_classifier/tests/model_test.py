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

"""Unit tests for LinearClassifier model."""


import numpy as np
import pytest

from src.models.linear_classifier.model import LinearClassifier


@pytest.mark.parametrize("n_features", [2, 256])
@pytest.mark.parametrize("n_samples", [10, 20])
def test_model_train(n_features: int, n_samples: int) -> None:
    """Test the `train` method of the `LinearClassifier` class."""

    clf = LinearClassifier()

    inputs = np.random.rand(n_samples, n_features)
    targets = np.empty(n_samples)
    targets[: (n_samples // 2)] = True
    targets[(n_samples // 2) :] = False

    clf.train(inputs=inputs, targets=targets)
    coef = clf.grid_search.best_estimator_.coef_
    intercept = clf.grid_search.best_estimator_.intercept_

    assert coef.shape == (1, n_features)
    assert intercept.shape == (1,)
    assert not np.isnan(coef).any()
    assert not np.isnan(intercept).any()
