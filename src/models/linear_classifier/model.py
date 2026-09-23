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

"""Component to learn a binary classifier from grounding embeddings (from prompts) plus labelled embeddings."""

import logging
from typing import Dict, Final, List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.visual_search.common.models import LinearClassifierBase

RANDOM_SEED: Final[int] = 42
logger: Final = logging.getLogger(__name__)


class LinearClassifier:
    def __init__(
        self,
        penalty: str = "l2",
        solver: str = "liblinear",
        class_weight: str = "balanced",
        scoring_metric: str = "f1",
        max_iter: int = 1000,
        param_grid: Dict[str, npt.NDArray[np.float32]] = {"C": np.logspace(-5, 5, 100)},
    ):
        """Wrapper for a Logistic Regression with L2 regularization.

        Args:
            penalty (str, optional): Regularization type. Defaults to "l2".
            solver (str, optional): Solver backend. Defaults to "liblinear".
            class_weight (str, optional): Weighting between classes. Defaults to "balanced".
            scoring_metric (str, optional): Scoring function for cross-validation. Defaults to "f1".
            max_iter (int, optional): Number of optimization (training) steps. Defaults to 1000.
            param_grid (Dict, optional): Parameters grid for grid search. Defaults to {"C": np.logspace(-5, 5, 100)}.
        """

        clf = LogisticRegression(
            penalty=penalty,
            solver=solver,
            random_state=RANDOM_SEED,
            class_weight=class_weight,
            max_iter=max_iter,
        )

        self.grid_search = GridSearchCV(
            clf,
            param_grid,
            cv=StratifiedKFold(),
            scoring=scoring_metric,
        )

    def train(
        self, inputs: npt.NDArray[np.float32], targets: npt.NDArray[np.bool_]
    ) -> None:
        """This function performs grid search, fits multiple models, and returns the best model.


        Args:
            inputs (npt.NDArray[np.float32]): Training input embeddings of shape (n_samples, n_features).
            targets (npt.NDArray[np.bool_]): Training binary target class of shape (n_samples, )

        Raises:
            ValueError: Raised if inputs and targets shapes don't match.
        """
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"inputs and targets shapes don't match! inputs.shape={inputs.shape}, targets.shape: {targets.shape}"
            )

        self.grid_search.fit(inputs, targets)

    def get_best_model(self) -> LogisticRegression:
        """This method returns the best fitted model.

        Returns:
            LogisticRegression: Best fitted model.
        """
        return self.grid_search.best_estimator_


def run_linear_classifier_training(
    labelled_embeddings: List[Tuple[List[float], bool]] = [],
) -> Tuple[LinearClassifier, LinearClassifierBase]:
    """This function runs linear classifier training.

    Args:
        labelled_embeddings (List[Tuple[List[float], bool]], optional): Human labelled embeddings. Defaults to [].

    Returns:
        Tuple[LogisticRegression, LinearClassifierBase]: The trained model.
    """
    clf = LinearClassifier()

    inputs = np.array([elem[0] for elem in labelled_embeddings])
    targets = np.array([elem[1] for elem in labelled_embeddings])

    clf.train(inputs, targets)
    best_model = clf.get_best_model()
    weights = LinearClassifierBase(
        coef=best_model.coef_.tolist(),
        intercept=best_model.intercept_.tolist(),
    )

    return clf, weights
