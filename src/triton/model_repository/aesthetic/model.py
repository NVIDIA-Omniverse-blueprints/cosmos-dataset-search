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

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Aesthetic predictor exposed as a Triton Python model."""

from pathlib import Path
from typing import Mapping

import torch
from torch.nn.functional import normalize

from src.models.aesthetic import AestheticPredictor
from src.triton.model_repository.clip.model import (
    TritonPythonModel as CLIPTritonPythonModel,
)

try:
    import src.triton.triton_python_backend_utils as pb_utils
except ImportError:
    # when called from triton server image, this import should work!
    import triton_python_backend_utils as pb_utils  # type: ignore


class TritonPythonModel(CLIPTritonPythonModel):
    """Aesthetic score triton model."""

    def initialize(self, args: Mapping[str, str]) -> None:
        super().initialize(args)
        if self.clip_variant != "ViT-L-14":
            raise ValueError(
                "Aesthetic predictor weights are supported only with ViT-L-14 CLIP model. "
                f"Got {self.clip_variant}"
            )
        if self.pretrained_weights != "openai":
            raise ValueError(
                "Aesthetic predictor weights are supported only with openai CLIP weights. "
                f"Got {self.pretrained_weights}"
            )
        default_weights = Path(__file__).parent / "linear_mlp.pth"
        weights_file = self.model_params.get("aesthetic_weights", {}).get(
            "string_value", default_weights.as_posix()
        )
        weights = torch.load(weights_file, map_location=torch.device("cpu"))

        self.predictor = AestheticPredictor(768, probabilistic=False)
        self.predictor.load_state_dict(weights)
        self.predictor.to(self.device)
        self.predictor.eval()
        self.predictor = torch.compile(self.predictor)

    def execute(self, requests):
        """Implementation of `execute` function for `TritonPythonModel`."""
        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            output_tensors = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                img_embeddings = self.encode_image(image_tensor.as_numpy())
                img_embeddings = normalize(img_embeddings, dim=-1)
                aesthetic_score = self.predictor(img_embeddings)
            output_tensors.append(
                pb_utils.Tensor("aesthetic_score", aesthetic_score.cpu().numpy())
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)
        return responses
