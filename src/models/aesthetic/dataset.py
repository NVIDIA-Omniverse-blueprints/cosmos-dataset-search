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

"""Dataloader for aesthetic predictor."""

from logging import getLogger
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds

logger = getLogger(__name__)


def nodesplitter(src, group=None):
    """Node splitter function for distributed setups."""
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        logger.info(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        logger.info(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


def transform(tupl: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Batch transforms function."""
    x, y = np.asarray(tupl[0]), np.asarray(tupl[1])
    x = x.squeeze(1)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
    y = y.reshape(-1, 1) / 10.0  # predict score from 0-1 as opposed to 0-10
    return x.astype(np.float32), y.astype(np.float32)


def make_loader(
    urls: Union[List[str], str],
    mode: Literal["train", "val"] = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    resampled: bool = True,
    shuffle_buffer: int = 5000,
) -> wds.WebLoader:
    """Create data loader from urls."""

    is_training = mode == "train"

    dataset = (
        wds.WebDataset(
            urls,
            repeat=is_training,
            cache_dir=cache_dir,
            shardshuffle=shuffle_buffer if is_training else False,
            resampled=resampled if is_training else False,
            handler=wds.ignore_and_continue,
            nodesplitter=None if (is_training and resampled) else nodesplitter,
        )
        .shuffle(shuffle_buffer if is_training else 0)
        .decode()
        .to_tuple("embedding.pyd aesthetic_score.pyd")
        .batched(batch_size, partial=False)
        .map(transform)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader
