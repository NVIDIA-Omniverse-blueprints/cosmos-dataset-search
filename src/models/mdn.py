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

"""Model definition for mixture density network (Bishop, 1994)."""

from enum import Enum, auto
from typing import Tuple

import torch
import torch.nn as nn


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    """Mixture density network, from Bishop, 1994.

    Attrs:
        dim_in: (int) dimensionality of the covariates
        dim_out: (int) dimensionality of the response variable
        n_components: (int) number of components in the mixture model
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        n_components,
        noise_type=NoiseType.DIAGONAL,
        fixed_noise_level=None,
    ):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        assert n_components > 0, "Expected n_components to be at least 1."
        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        if n_components > 1:
            self.pi_network = nn.Sequential(
                nn.Linear(dim_in, n_components),
            )
        else:
            self.pi_network = None
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, dim_out * n_components + num_sigma_channels)
        )

    def forward(
        self, x: torch.Tensor, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function of mixture density model.

        Returns:
            log_pi: (batch size, number of components)
            mu: (batch size, number of components, output dim)
            sigma: (batch size, number of components, output dim)
        """
        if self.pi_network is not None:
            log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        else:
            log_pi = torch.zeros((x.shape[0], 1), device=x.device)
        normal_params = self.normal_network(x)
        mu = normal_params[..., : self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components :]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def loss(
        self,
        log_pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss."""
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = -0.5 * torch.einsum(
            "bij,bij->bi", z_score, z_score
        ) - torch.sum(torch.log(sigma), dim=-1)
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample from mixture of gaussians."""
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(
            rand_normal, indices=rand_pi.unsqueeze(-1), dim=1
        ).squeeze(dim=1)
        return samples
