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

"""Inference utilities for aesthetic predictor.

Works on CLIP pre-computed embeddings as input representation.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from src.models.aesthetic import AestheticPredictor

_bazel_models = {
    (
        "ViT-L-14",
        "openai",
        False,
    ): "models/aesthetic/variants/clip_ViT-L-14_openai_aesthetic_head.pth",
    (
        "ViT-H-14",
        "laion2b_s32b_b79k",
        False,
    ): "models/aesthetic/variants/clip_ViT-H-14_laion2b_s32b_b79k_aesthetic_head.pth",
    (
        "ViT-L-14",
        "datacomp_xl_s13b_b90k",
        False,
    ): "models/aesthetic/variants/clip_ViT-L-14_datacomp_xl_s13b_b90k_aesthetic_head.pth",
}


def load_model(
    clip_model_name: str,
    clip_model_weights: str,
    probabilistic: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Load pretrained aesthetic scoring heads."""

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    )

    if clip_model_name == "ViT-L-14":
        head = AestheticPredictor(768, probabilistic)
    elif clip_model_name == "ViT-H-14":
        head = AestheticPredictor(1024, probabilistic)
    else:
        raise NotImplementedError(f"CLIP model {clip_model_name} not implemented!")

    weights_key = (clip_model_name, clip_model_weights, probabilistic)
    if weights_key not in _bazel_models:
        raise KeyError(f"Only {_bazel_models.keys()} are available! Got {weights_key}.")
    head_weights = _bazel_models[weights_key]
    
    # Ensure Triton compatibility by creating linear_mlp.pth symlink if needed
    linear_mlp_path = Path("models/aesthetic/1/linear_mlp.pth")
    if not linear_mlp_path.exists() and weights_key == ("ViT-L-14", "openai", False):
        # Create symlink for Triton compatibility
        head_weights_path = Path(head_weights)
        if head_weights_path.exists():
            try:
                linear_mlp_path.symlink_to(head_weights_path.name)
            except (OSError, FileExistsError):
                pass  # Symlink already exists or can't be created
    
    weights = torch.load(head_weights, weights_only=True, map_location=device)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    head.load_state_dict(weights)
    head.eval()
    head.to(device)

    return head
