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

"""Haystack components for patching images."""


import io
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import chain, repeat
from typing import Dict, Generator, List, Sequence, Tuple

from haystack import Document, component
from haystack.dataclasses import ByteStream
from PIL import Image

from src.haystack.serializer import SerializerMixin

BoundingBoxTLBR = Tuple[float, float, float, float]
H = int
W = int


def input_checks(doc: Document) -> bool:
    """Checks on input documents."""
    if doc.blob is None:
        msg = (
            f"Document {doc.id} does not contain a `blob` attribute for image content."
        )
        logging.error(msg)
        raise ValueError(msg)
    return True


def get_image_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert pillow image into bytes."""
    bytes_arr = io.BytesIO()
    image.save(bytes_arr, format=format)
    return bytes_arr.getvalue()


def patch_image(
    img: Image.Image, bboxes_norm: Sequence[BoundingBoxTLBR]
) -> Generator[Tuple[Image.Image, Dict[str, float]], None, None]:
    """Patch images based on normalized bboxes."""
    width, height = img.size
    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes_norm):
        x_min_pt = int(x_min * width)
        y_min_pt = int(y_min * height)
        x_max_pt = int(x_max * width)
        y_max_pt = int(y_max * height)
        meta = {
            "patch_id": i,
            "bbox_h": y_max - y_min,
            "bbox_w": x_max - x_min,
            "bbox_x": (x_max + x_min) * 0.5,
            "bbox_y": (y_max + y_min) * 0.5,
        }
        crop = img.crop((x_min_pt, y_min_pt, x_max_pt, y_max_pt))
        yield crop, meta


def patch_document(
    document: Document, bboxes_norm: Tuple[BoundingBoxTLBR, ...]
) -> List[Document]:
    """Patch document content based on normalized bboxes."""
    new_docs = []
    if document.blob is None:
        raise ValueError(f"Document {document.id} has a null blob attribute!")
    with io.BytesIO(document.blob.data) as fp:
        img = Image.open(fp)
        for new_img, new_meta in patch_image(img, bboxes_norm):
            meta = deepcopy(document.meta)
            meta.update(**new_meta)
            meta["source_id"] = document.id
            new_docs.append(
                Document(
                    content=None,
                    meta=meta,
                    embedding=None,
                    blob=ByteStream(
                        data=get_image_bytes(new_img),
                        mime_type=document.blob.mime_type,
                    ),
                )
            )
    return new_docs


class PatchingMixin:
    def __init__(
        self,
        bboxes_norm: Tuple[BoundingBoxTLBR, ...],
        max_workers: int = 4,
    ) -> None:
        self.bboxes_norm = bboxes_norm
        self.max_workers = max_workers

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        if not documents:
            return {"documents": documents}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_docs = list(
                chain.from_iterable(
                    executor.map(
                        patch_document,
                        documents,
                        repeat(self.bboxes_norm, len(documents)),
                    )
                )
            )

        expected_documents = len(documents) * len(self.bboxes_norm)
        if len(new_docs) != expected_documents:
            raise RuntimeError(
                f"Document size expected {expected_documents} does not match "
                f"document size {len(new_docs)}."
            )

        return {"documents": new_docs}


@component
class BoundingBoxPatchSplitting(SerializerMixin, PatchingMixin):
    """
    A component for breaking up images into patches from a list of normalised bounding boxes.
    """


@component
class MultiScalePatchSplitting(SerializerMixin, PatchingMixin):
    """
    A component for breaking up images into multi-resolution patches.
    """

    def __init__(
        self,
        num_patches: Tuple[Tuple[H, W], ...],
    ) -> None:
        """Constructor for `MultiScalePatchSplitting`.

        Args:
            num_patches: for each height and width tuple, will split image by that number.
                For example, ((1, 1), (3, 3), (5, 5)) will split image into 1, 9 and 25 patches.
        """
        PatchingMixin.__init__(self, bboxes_norm=self.patches_to_boxes(num_patches))
        self.num_patches = num_patches

    @staticmethod
    def patches_to_boxes(
        num_patches: Sequence[Tuple[H, W]]
    ) -> Tuple[BoundingBoxTLBR, ...]:
        bboxes = []
        for h, w in num_patches:
            patch_width = 1.0 / w
            patch_height = 1.0 / h
            for i in range(h):
                for j in range(w):
                    top_left_x = j * patch_width
                    top_left_y = i * patch_height
                    bottom_right_x = (j + 1.0) * patch_width
                    bottom_right_y = (i + 1.0) * patch_height
                    bboxes.append(
                        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                    )
        return tuple(bboxes)
