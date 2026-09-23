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

"""Triton client for aesthetic model."""

import time

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


class TritonAestheticClient:
    def __init__(self, url: str = "localhost:8000", model_name: str = "clip") -> None:
        self.url = url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=url)

    def encode_image(self, batch: np.ndarray) -> np.ndarray:
        """Encode image batch into aesthetic scores."""

        input_tensors = [
            httpclient.InferInput(
                "image", batch.shape, datatype=np_to_triton_dtype(batch.dtype)
            )
        ]
        input_tensors[0].set_data_from_numpy(batch)

        response = self.client.infer(
            model_name=self.model_name,
            inputs=input_tensors,
            outputs=[httpclient.InferRequestedOutput("aesthetic_score")],
        )
        return response.as_numpy("aesthetic_score")


def test() -> None:
    client = TritonAestheticClient(model_name="aesthetic")
    images = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.uint8)
    timings = []
    for i in range(30):
        start = time.time()
        aesthetic_scores = client.encode_image(images)
        print(aesthetic_scores)
        stop = time.time()
        if i > 3:
            timings.append(stop - start)
    print(f"Average time: {np.mean(timings)}")


if __name__ == "__main__":
    test()
