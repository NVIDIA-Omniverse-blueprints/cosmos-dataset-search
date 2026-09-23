# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Safe media inspection before entering native GPU video decoding."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FFPROBE_TIMEOUT_SECONDS = 15.0
GPU_DECODER_MAX_HEIGHT = 2160
GPU_DECODER_MAX_WIDTH = 4096
UNSAFE_RANDOM_ACCESS_FORMATS = {"matroska", "webm"}
# Match FFmpeg2NvCodecId in the pinned PyNvVideoCodec 2.2.3 public source.
# Reject unmapped codecs before reconfiguring a cached native decoder: a failed
# H.264-to-FFV1 reconfiguration can crash later while that decoder is destroyed.
GPU_DECODER_CODECS = frozenset({
    "mpeg1video", "mpeg2video", "mpeg4", "wmv3", "vc1", "h264", "hevc",
    "vp8", "vp9", "mjpeg", "av1",
})


class GpuDecodeUnsupported(RuntimeError):
    """Raised when media cannot safely enter the GPU decoder."""


@dataclass(frozen=True)
class VideoStreamMetadata:
    """Video metadata needed to make decoder-safety decisions."""

    codec: str
    formats: frozenset[str]
    width: int | None = None
    height: int | None = None

    @property
    def description(self) -> str:
        container = "/".join(sorted(self.formats)) or "unknown container"
        dimensions = (
            f", {self.width}x{self.height}"
            if self.width is not None and self.height is not None
            else ""
        )
        return f"{self.codec.upper()} in {container}{dimensions}"


def probe_gpu_video_input(media_path: Path) -> VideoStreamMetadata:
    """Inspect media and reject native decoder combinations known to be unsafe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height:format=format_name",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GpuDecodeUnsupported(
            "video metadata could not be inspected safely"
        ) from error

    if result.returncode != 0:
        raise GpuDecodeUnsupported("video metadata could not be inspected safely")

    metadata = _parse_ffprobe_output(result.stdout)
    if metadata.codec not in GPU_DECODER_CODECS:
        raise GpuDecodeUnsupported(
            f"{metadata.description} is not supported by the GPU video decoder; "
            "re-encode the video as H.264 or HEVC in MP4"
        )
    if metadata.codec == "vp9" and metadata.formats & UNSAFE_RANDOM_ACCESS_FORMATS:
        raise GpuDecodeUnsupported(
            "VP9 in Matroska/WebM is not supported by the GPU random-access "
            "decoder; re-encode the video as H.264 or HEVC in MP4"
        )
    if (
        metadata.width is not None
        and metadata.height is not None
        and (
            metadata.width > GPU_DECODER_MAX_WIDTH
            or metadata.height > GPU_DECODER_MAX_HEIGHT
        )
    ):
        raise GpuDecodeUnsupported(
            f"{metadata.description} exceeds the GPU decoder maximum "
            f"{GPU_DECODER_MAX_WIDTH}x{GPU_DECODER_MAX_HEIGHT}; re-encode the "
            "video at a supported resolution"
        )
    return metadata


def _parse_ffprobe_output(output: str) -> VideoStreamMetadata:
    try:
        payload = json.loads(output)
        stream = payload["streams"][0]
        codec = stream["codec_name"].strip().lower()
        format_name = payload["format"]["format_name"]
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise GpuDecodeUnsupported(
            "video metadata could not be inspected safely"
        ) from error

    if not codec or not isinstance(format_name, str):
        raise GpuDecodeUnsupported("video metadata could not be inspected safely")

    formats = frozenset(
        value.strip().lower() for value in format_name.split(",") if value.strip()
    )
    if not formats:
        raise GpuDecodeUnsupported("video metadata could not be inspected safely")
    return VideoStreamMetadata(
        codec=codec,
        formats=formats,
        width=_optional_positive_int(stream.get("width")),
        height=_optional_positive_int(stream.get("height")),
    )


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
