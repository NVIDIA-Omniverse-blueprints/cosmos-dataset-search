# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU video sampling and failure handling without a CUDA device in unit CI."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.cosmos_embed_oss.media_probe import GpuDecodeUnsupported
from src.cosmos_embed_oss import media_probe


@pytest.fixture
def gpu_decoder(monkeypatch, tmp_path):
    # Import the real pipeline against narrow native-library doubles. Loading
    # a separately named module keeps these doubles out of other tests.
    monkeypatch.setitem(sys.modules, "cvcuda", ModuleType("cvcuda"))
    nvidia = ModuleType("nvidia")
    nvidia.nvimgcodec = ModuleType("nvidia.nvimgcodec")
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.nvimgcodec", nvidia.nvimgcodec)
    monkeypatch.setitem(sys.modules, "av", None)
    spec = importlib.util.spec_from_file_location(
        "ce1_gpu_sampling_test", Path(__file__).parents[1] / "gpu_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = {"total": 17, "indices": [], "incomplete": False}

    class Decoder:
        def __len__(self):
            return calls["total"]

        def get_batch_frames_by_index(self, indices):
            calls["indices"].append(indices)
            selected = indices[:-1] if calls["incomplete"] else indices
            return [np.full((2, 3, 3), i, dtype=np.uint8) for i in selected]

    def new_decoder(path, **kwargs):
        calls["path"] = path
        calls["options"] = kwargs
        return Decoder()

    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", SimpleNamespace(
        SimpleDecoder=new_decoder, OutputColorType=SimpleNamespace(RGB="RGB")
    ))
    monkeypatch.setattr(module, "probe_gpu_video_input", lambda path: SimpleNamespace(
        description="H.264 in MP4"
    ))
    state = SimpleNamespace(
        cvcuda_stream=SimpleNamespace(), torch_stream=object()
    )

    class Stream:
        handle = 123

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sync(self):
            calls["synced"] = True

    state.cvcuda_stream = Stream()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(device=lambda *_: nullcontext(), stream=lambda *_: nullcontext()),
        stack=lambda frames: np.stack(frames),
    )
    processor = object.__new__(module.GpuMediaProcessor)
    processor._torch = torch
    processor._device = "cuda:0"
    processor._device_index = 0
    processor._frame_count = 8
    processor._stream_state = lambda: state
    processor._copy_rgb_frame = lambda frame, _label: frame
    processor._resize_batch = lambda rgb, _state: SimpleNamespace(
        unsqueeze=lambda _dim: np.expand_dims(np.transpose(rgb, (0, 3, 1, 2)), 0)
    )
    path = tmp_path / "video.mp4"
    path.write_bytes(b"synthetic native-decoder input")
    return processor, path, calls, state


@pytest.mark.parametrize("total,expected", [
    (1, [0, 0, 0, 0, 0, 0, 0, 0]),
    (2, [0, 0, 0, 0, 0, 0, 0, 1]),
    (7, [0, 0, 1, 2, 3, 4, 5, 6]),
    (8, [0, 1, 2, 3, 4, 5, 6, 7]),
    (17, [0, 2, 4, 6, 9, 11, 13, 16]),
])
def test_gpu_video_samples_unique_indices_and_restores_order(gpu_decoder, total, expected):
    processor, path, calls, _state = gpu_decoder
    calls["total"] = total
    result = processor.decode_video(path)

    assert result.shape == (1, 8, 3, 2, 3)
    assert result[0, :, 0, 0, 0].tolist() == expected
    assert calls["indices"] == [sorted(set(expected))]
    assert calls["path"] == str(path)
    assert calls["options"]["use_device_memory"] is True
    assert calls["options"]["need_scanned_stream_metadata"] is True
    assert calls["synced"] is True


@pytest.mark.parametrize("failure", ["empty", "incomplete"])
def test_gpu_video_failure_discards_decoder_without_cpu_fallback(gpu_decoder, failure):
    processor, path, calls, state = gpu_decoder
    if failure == "empty":
        calls["total"] = 0
    else:
        calls["incomplete"] = True

    with pytest.raises(GpuDecodeUnsupported, match="re-encode.*H.264 or HEVC in MP4"):
        processor.decode_video(path)

    assert state.video_decoder is None
    assert "synced" not in calls
    if failure == "empty":
        assert calls["indices"] == []

    calls["total"] = 8
    calls["incomplete"] = False
    assert processor.decode_video(path).shape == (1, 8, 3, 2, 3)


def test_unmapped_codec_never_reconfigures_cached_native_decoder(gpu_decoder, monkeypatch):
    processor, path, calls, state = gpu_decoder
    processor.decode_video(path)
    cached_decoder = state.video_decoder
    monkeypatch.setitem(
        processor.decode_video.__globals__, "probe_gpu_video_input",
        media_probe.probe_gpu_video_input,
    )
    monkeypatch.setattr(media_probe.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({
            "streams": [{"codec_name": "ffv1", "width": 320, "height": 192}],
            "format": {"format_name": "matroska,webm"},
        }),
    ))

    def forbidden_reconfigure(*_args):
        pytest.fail("unsupported codec entered native decoder reconfiguration")

    cached_decoder.reconfigure_decoder = forbidden_reconfigure
    previous_batches = len(calls["indices"])
    with pytest.raises(GpuDecodeUnsupported, match="FFV1.*not supported"):
        processor.decode_video(path)
    assert state.video_decoder is cached_decoder
    assert len(calls["indices"]) == previous_batches
