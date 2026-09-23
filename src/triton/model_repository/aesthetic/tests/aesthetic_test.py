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

"""Unit tests for Triton CLIP model."""

import json
import os
from pathlib import Path

import numpy as np

import src.triton.triton_python_backend_utils as pb_utils
from src.triton.model_repository.aesthetic.model import TritonPythonModel


def test_aesthetic_default(tmp_path: Path) -> None:
    """Test aesthetic model inference."""

    os.environ["HOME"] = tmp_path.as_posix()
    model_config = {
        "parameters": {
            "clip_variant": {"string_value": "ViT-L-14"},
            "clip_weights": {"string_value": "openai"},
            "device": {"string_value": "cuda"},
            "cache_dir": {"string_value": tmp_path.as_posix()},
            "aesthetic_weights": {
                "string_value": "models/aesthetic/1/linear_mlp.pth"
            },
        }
    }
    model_config_str = json.dumps(model_config)
    model = TritonPythonModel()
    model.initialize({"model_config": model_config_str})
    assert model.cache_dir == tmp_path.as_posix()
    assert model.clip_variant == "ViT-L-14"
    assert model.pretrained_weights == "openai"

    with open("src/triton/model_repository/aesthetic/tests/data/test_image.png", "rb") as fp:
        image_bytes = fp.read()

    requests = [
        pb_utils.InferenceRequest(
            inputs=[
                pb_utils.Tensor(
                    "image", np.array([image_bytes, image_bytes], dtype=object)
                )
            ],
            request_id="test_image",
            correlation_id="",
            requested_output_names=["aesthetic_score"],
        ),
    ]
    responses = model.execute(requests)
    assert len(responses) == len(requests)
    image_response = responses

    score_outputs = image_response[0].output_tensors()
    assert len(score_outputs) == 1
    score_outputs[0].as_numpy().shape == (2, 1)
