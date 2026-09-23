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

"""Component to learn a query from grounding embeddings (from prompts) plus labelled embeddings."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from haystack import component
from numpy import dtype, ndarray
from torch.utils.data import DataLoader, TensorDataset

from src.haystack.serializer import SerializerMixin


class LinearProbe(nn.Module):
    def __init__(
        self,
        grounding_embeddings: List[List[float] | npt.NDArray[np.float32]],
        regularization_strength: float = 0.05,
    ) -> None:
        super(LinearProbe, self).__init__()
        if not grounding_embeddings:
            raise ValueError("Provide at least one grounding embedding.")
        self.embedding_size = len(grounding_embeddings[0])
        if not all(len(emb) == self.embedding_size for emb in grounding_embeddings):
            raise ValueError(
                f"Grounding embeddings need to all be the same dimensionality {self.embedding_size}!"
            )
        self.linear = nn.Linear(self.embedding_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.device = torch.device("cpu")

        self.weights_loss_fac = torch.tensor(
            regularization_strength, device=self.device
        )
        self.anchor = torch.tensor(
            np.ones((self.embedding_size,)) * np.inf * -1, device=self.device
        )
        for emb in grounding_embeddings:
            self.anchor = torch.maximum(self.anchor, torch.Tensor(emb).to(self.device))
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.embedding_size)
        out = self.linear(x)
        return out

    def train_model(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.bool_],
        nb_epochs: int = 1000,
    ) -> None:
        loader = self._create_dataloaders(X=X, y=y)
        self.train()

        for _ in range(nb_epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self(data)

                loss_target = self.criterion(outputs.squeeze(1), target)
                loss_weights = torch.sum((self.linear.weight - self.anchor) ** 2)

                loss = loss_target + self.weights_loss_fac * loss_weights

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_weights(self) -> npt.NDArray[np.float32]:
        weights = self.linear.weight.detach().cpu().numpy().flatten()
        return weights

    def _create_dataloaders(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.bool_],
    ) -> DataLoader:
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        return train_loader


@component
class LinearProbeQueryLearner(SerializerMixin):
    """Component that learns an optimized query based on grounding embeddings and labelled embeddings.

    The grounding embeddings are used as regularization for the linear regression model.
    The labelled embeddings are used for the binary classification loss term.
    """

    @component.output_types(query_embeddings=List[List[float]])
    def run(
        self,
        grounding_embeddings: List[Union[List[float], ndarray[Any, dtype[np.float32]]]],
        labelled_embeddings: List[Tuple[List[float], bool]] = [],
        regularization_strength: float = 0.05,
        nb_epochs: int = 1000,
    ) -> Dict[str, List[Union[List[float], ndarray[Any, dtype[np.float32]]]]]:
        if not grounding_embeddings and not labelled_embeddings:
            return {"query_embeddings": []}
        if grounding_embeddings and not labelled_embeddings:
            return {"query_embeddings": grounding_embeddings}
        probe = LinearProbe(
            grounding_embeddings=grounding_embeddings,
            regularization_strength=regularization_strength,
        )
        X = np.array([emb for emb, _ in labelled_embeddings], dtype=np.float32)
        y = np.array([lab for _, lab in labelled_embeddings], dtype=np.bool_)
        probe.train_model(X=X, y=y, nb_epochs=nb_epochs)
        return {"query_embeddings": [probe.get_weights().tolist()]}
