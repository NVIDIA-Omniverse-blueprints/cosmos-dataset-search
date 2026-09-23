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

"""Model definition for simple MLP that does aesthetic image prediction based on fixed-size input embeddings."""

from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from src.models.mdn import MixtureDensityNetwork


class AestheticPredictor(pl.LightningModule):
    """Linear MLP for aesthetic prediction given CLIP image embeddings."""

    def __init__(self, input_size: int, probabilistic: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            (
                MixtureDensityNetwork(
                    dim_in=16,
                    dim_out=1,
                    n_components=1,
                )
                if probabilistic
                else nn.Linear(16, 1)
            ),
        )
        self.probabilistic = probabilistic
        self.r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()

    def _validate_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Validate dimensions of input tensors."""
        x, y = batch
        if x.ndim != 2:
            raise ValueError(
                f"Expected input tensor `x` to be of rank 2. Got {x.ndim}!"
            )
        if y.ndim != 2:
            raise ValueError(
                f"Expected input tensor `y` to be of rank 2. Got {y.ndim}!"
            )
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input tensor `x` to be of shape [N, {self.input_size}]. Got {x.shape}!"
            )
        if y.shape[1] != 1:
            raise ValueError(
                f"Expected input tensor `y` to be of shape [N, 1]. Got {y.shape}!"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Expected x and y to be of same length, got {x.shape[0]} and {y.shape[0]}!"
            )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward function of model."""
        return self.layers(torch.nn.functional.normalize(x))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        self._validate_batch(batch)
        x, y = batch

        if self.probabilistic:
            log_pi, mu, sigma = self.forward(x)
            loss = self.layers[-1].loss(log_pi, mu, sigma, y)
            x_hat = mu.squeeze(1)
        else:
            x_hat = self.forward(x)
            loss = nn.functional.mse_loss(x_hat, y)

        assert x_hat.shape == (x.shape[0], 1)
        loss = torch.mean(loss)
        self.log("loss", loss, prog_bar=True)
        r2 = self.r2(x_hat, y)
        self.log("r2", r2, prog_bar=True)
        mape = self.mape(x_hat, y)
        self.log("mape", mape, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        self._validate_batch(batch)
        x, y = batch

        if self.probabilistic:
            log_pi, mu, sigma = self.forward(x)
            loss = self.layers[-1].loss(log_pi, mu, sigma, y)
            x_hat = mu.squeeze(1)
        else:
            x_hat = self.forward(x)
            loss = nn.functional.mse_loss(x_hat, y)

        assert x_hat.shape == (x.shape[0], 1)
        loss = torch.mean(loss)
        self.log("val_loss", loss, prog_bar=True)

        r2 = self.val_r2(x_hat, y)
        self.log("val_r2", r2, prog_bar=True)
        mape = self.val_mape(x_hat, y)
        self.log("val_mape", mape, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)
