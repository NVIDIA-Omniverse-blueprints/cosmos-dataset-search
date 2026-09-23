# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CE1 GPU media safety inspection."""

import json
from types import SimpleNamespace

import pytest

from src.cosmos_embed_oss import media_probe


def _ffprobe_result(*, codec: str, format_name: str):
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "streams": [
                    {
                        "codec_name": codec,
                        "width": 1920,
                        "height": 1080,
                    }
                ],
                "format": {"format_name": format_name},
            }
        ),
    )


def test_probe_gpu_video_input_allows_h264_mp4(monkeypatch, tmp_path):
    monkeypatch.setattr(
        media_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: _ffprobe_result(
            codec="h264",
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
        ),
    )

    metadata = media_probe.probe_gpu_video_input(tmp_path / "video.bin")

    assert metadata.codec == "h264"
    assert "mp4" in metadata.formats
    assert metadata.description == "H264 in 3g2/3gp/m4a/mj2/mov/mp4, 1920x1080"


def test_probe_gpu_video_input_rejects_vp9_matroska_before_native_decode(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        media_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: _ffprobe_result(
            codec="vp9",
            format_name="matroska,webm",
        ),
    )

    with pytest.raises(media_probe.GpuDecodeUnsupported) as raised:
        media_probe.probe_gpu_video_input(tmp_path / "video.bin")

    assert "VP9 in Matroska/WebM" in str(raised.value)
    assert "H.264 or HEVC in MP4" in str(raised.value)


def test_probe_gpu_video_input_rejects_resolution_above_decoder_limit(
    monkeypatch,
    tmp_path,
):
    result = _ffprobe_result(
        codec="hevc",
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
    )
    payload = json.loads(result.stdout)
    payload["streams"][0].update(width=7680, height=4320)
    result.stdout = json.dumps(payload)
    monkeypatch.setattr(
        media_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(media_probe.GpuDecodeUnsupported) as raised:
        media_probe.probe_gpu_video_input(tmp_path / "video.bin")

    assert "HEVC" in str(raised.value)
    assert "7680x4320" in str(raised.value)
    assert "4096x2160" in str(raised.value)


def test_probe_gpu_video_input_rejects_unreadable_media(monkeypatch, tmp_path):
    monkeypatch.setattr(
        media_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    with pytest.raises(
        media_probe.GpuDecodeUnsupported,
        match="metadata could not be inspected safely",
    ):
        media_probe.probe_gpu_video_input(tmp_path / "video.bin")


@pytest.mark.parametrize("codec", sorted(media_probe.GPU_DECODER_CODECS))
def test_probe_preserves_native_codec_mappings(monkeypatch, tmp_path, codec):
    monkeypatch.setattr(
        media_probe.subprocess, "run",
        lambda *_args, **_kwargs: _ffprobe_result(codec=codec, format_name="avi"),
    )
    assert media_probe.probe_gpu_video_input(tmp_path / "video.bin").codec == codec


@pytest.mark.parametrize("codec", ["ffv1", "prores", "theora", "rawvideo", "unknown"])
def test_probe_rejects_unmapped_codec_before_native_decode(monkeypatch, tmp_path, codec):
    monkeypatch.setattr(
        media_probe.subprocess, "run",
        lambda *_args, **_kwargs: _ffprobe_result(codec=codec, format_name="matroska,webm"),
    )
    with pytest.raises(media_probe.GpuDecodeUnsupported, match="not supported.*GPU video decoder"):
        media_probe.probe_gpu_video_input(tmp_path / "video.bin")
