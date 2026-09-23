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

"""Script for preparing embedding data given AVA or SAC dataset."""

from pathlib import Path
from typing import Any, Dict

import braceexpand
import click
import open_clip
import torch
from webdataset import ShardWriter, WebDataset


@click.command
@click.option(
    "--url",
    help="URL of AVA or SAC image webdataset (i.e. in team-vision-language bucket).",
    required=True,
)
@click.option(
    "--output-dir",
    help="Output directory of AVA or SAC processed webdataset.",
    required=True,
)
@click.option(
    "--clip-model-name",
    help="Identifier for CLIP model.",
    default="ViT-H-14",
    show_default=True,
)
@click.option(
    "--clip-model-weights",
    help="Identifier for CLIP weights.",
    default="laion2b_s32b_b79k",
    show_default=True,
)
@click.option(
    "--batch-size",
    help="Batch size for processing images",
    default=100,
    show_default=True,
)
@click.option(
    "--num-workers",
    help="Number of workers for webdataset reader",
    default=4,
    show_default=True,
)
@click.option(
    "--device",
    help="Device to run inference on",
    default="cuda:0",
    show_default=True,
)
@click.option(
    "--precision",
    help="Inference precision",
    default="fp32",
    type=click.Choice(["fp32", "fp16"]),
    show_default=True,
)
@click.option(
    "--profile",
    help="S3 credentials profile (only for `s3://` prefix url inputs)",
    default="team-vision-language",
    show_default=True,
)
@click.option(
    "--endpoint-url",
    help="S3 endpoint URL (only for `s3://` prefix url inputs)",
    default="https://pbss.s8k.io",
    show_default=True,
)
def cli(
    url: str,
    output_dir: str,
    clip_model_name: str,
    clip_model_weights: str,
    batch_size: int,
    num_workers: int,
    device: str,
    precision: str,
    profile: str,
    endpoint_url: str,
) -> None:
    """CLI for data pre-processing of SAC/AVA datasets for aesthetic model training."""

    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Directory {output_path} was not found!")
    output_path = output_path / clip_model_name / clip_model_weights
    output_path.mkdir(parents=True, exist_ok=False)

    if url.startswith("s3://"):
        urls = [
            f"pipe:aws s3 cp --endpoint-url {endpoint_url} --profile {profile} {f} -"
            for f in braceexpand.braceexpand(url)
        ]
    else:
        urls = [url]

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, clip_model_weights, device=device, precision=precision
    )
    model.eval()

    def image_preprocess_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image.png"] = preprocess(sample["image.png"])
        return sample

    dataset = (
        WebDataset(urls)
        .decode("pilrgb")
        .map(image_preprocess_fn)
        .to_tuple("__key__", "image.png", "filename.txt", "aesthetic_score.pyd")
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    writer = ShardWriter(output_path.as_posix() + "/shard-%06d.tar", maxcount=1000)

    for key, images, txt, scores in dataloader:
        images = images.to(device)
        with torch.no_grad():
            embeddings = model.encode_image(images).cpu().numpy()
        for idx in range(len(images)):
            sample = {
                "__key__": key[idx],
                "filename.txt": txt[idx],
                "embedding.pyd": embeddings[idx : idx + 1],
                "aesthetic_score.pyd": scores[idx],
            }
            writer.write(sample)


if __name__ == "__main__":
    cli()
