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

"""GPU media decoding and preprocessing for Cosmos-Embed1."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cvcuda
import nvidia.nvimgcodec as nvimgcodec

from src.cosmos_embed_oss.backends import _linspace_frame_indices
from src.cosmos_embed_oss.media_probe import (
    GPU_DECODER_MAX_HEIGHT,
    GPU_DECODER_MAX_WIDTH,
    GpuDecodeUnsupported,
    probe_gpu_video_input,
)


class GpuMediaProcessor:
    """Decode and resize Cosmos-Embed1 visual inputs on the GPU."""

    def __init__(
        self,
        *,
        torch: Any,
        device: str,
        dtype: Any,
        resolution: int,
        frame_count: int,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")

        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("GPU media processing requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU media processing requires CUDA")

        device_index = requested_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        self._torch = torch
        self._device = torch.device("cuda", device_index)
        self._device_index = device_index
        self._dtype = dtype
        self._resolution = resolution
        self._frame_count = frame_count
        self._tls = threading.local()

    def decode_frames(self, encoded_frames: Sequence[bytes]) -> Any:
        """Decode one clip of encoded images into a ``B,T,C,H,W`` tensor."""

        if len(encoded_frames) != self._frame_count:
            raise ValueError(
                f"expected {self._frame_count} video frames, got "
                f"{len(encoded_frames)}"
            )
        state = self._stream_state()

        if not hasattr(state, "image_decoder"):
            with self._torch.cuda.device(self._device):
                state.image_decoder = nvimgcodec.Decoder(
                    device_id=self._device_index,
                    max_num_cpu_threads=4,
                )

        code_streams = []
        for index, payload in enumerate(encoded_frames):
            try:
                code_streams.append(nvimgcodec.CodeStream(payload))
            except Exception as error:
                raise ValueError(
                    f"video frame {index} is not a supported image"
                ) from error

        with (
            self._torch.cuda.device(self._device),
            state.cvcuda_stream,
            self._torch.cuda.stream(state.torch_stream),
        ):
            decode_params = nvimgcodec.DecodeParams()
            decode_params.apply_exif_orientation = False
            decoded = state.image_decoder.read(
                code_streams,
                params=decode_params,
                cuda_stream=state.cvcuda_stream.handle,
            )
            if len(decoded) != len(code_streams):
                raise RuntimeError("nvImageCodec returned an incomplete frame batch")
            frames = [
                self._copy_rgb_frame(frame, f"video frame {index}")
                for index, frame in enumerate(decoded)
            ]
            output = self._resize_scale_reformat(frames, state)

        state.cvcuda_stream.sync()
        return output

    def decode_video(self, media_path: Path) -> Any:
        """Decode one video into a ``B,T,C,H,W`` tensor."""

        import PyNvVideoCodec as nvc

        metadata = probe_gpu_video_input(media_path)
        state = self._stream_state()
        try:
            with (
                self._torch.cuda.device(self._device),
                state.cvcuda_stream,
                self._torch.cuda.stream(state.torch_stream),
            ):
                decoder = self._video_decoder(media_path, state, nvc)
                total_frames = self._decoder_frame_count(decoder)
                frame_indices = _linspace_frame_indices(total_frames, self._frame_count)
                unique_indices = sorted(set(frame_indices))
                decoded = decoder.get_batch_frames_by_index(unique_indices)
                if len(decoded) != len(unique_indices):
                    raise ValueError(
                        "PyNvVideoCodec returned an incomplete frame batch"
                    )

                frames_by_index = {}
                for index, frame in zip(unique_indices, decoded, strict=True):
                    frames_by_index[index] = self._copy_rgb_frame(
                        frame,
                        f"video frame {index}",
                    )

                rgb = self._torch.stack(
                    [frames_by_index[index] for index in frame_indices]
                )
                output = self._resize_batch(rgb, state).unsqueeze(0)

            state.cvcuda_stream.sync()
            return output
        except GpuDecodeUnsupported as error:
            state.video_decoder = None
            raise GpuDecodeUnsupported(
                f"GPU video decoder could not decode {metadata.description}: "
                f"{error}; re-encode the video as H.264 or HEVC in MP4"
            ) from error
        except Exception as error:
            state.video_decoder = None
            raise GpuDecodeUnsupported(
                f"GPU video decoder could not decode {metadata.description}; "
                "re-encode the video as H.264 or HEVC in MP4"
            ) from error

    def _video_decoder(self, media_path: Path, state: Any, nvc: Any) -> Any:
        decoder = getattr(state, "video_decoder", None)
        if decoder is not None:
            try:
                decoder.reconfigure_decoder(str(media_path))
                return decoder
            except Exception:
                state.video_decoder = None

        try:
            state.video_decoder = nvc.SimpleDecoder(
                str(media_path),
                gpu_id=self._device_index,
                cuda_stream=state.cvcuda_stream.handle,
                use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
                need_scanned_stream_metadata=True,
                max_width=GPU_DECODER_MAX_WIDTH,
                max_height=GPU_DECODER_MAX_HEIGHT,
            )
        except Exception:
            state.video_decoder = None
            raise
        return state.video_decoder

    def _decoder_frame_count(self, decoder: Any) -> int:
        """Return the decoder frame count, rejecting unknown or empty media."""

        try:
            total_frames = len(decoder)
        except Exception as error:
            raise GpuDecodeUnsupported(
                "video frame count is unavailable from the GPU decoder"
            ) from error
        if total_frames is None or total_frames <= 0:
            raise GpuDecodeUnsupported("video reported no decodable frames")
        return int(total_frames)

    def _copy_rgb_frame(self, frame: Any, label: str) -> Any:
        if frame is None:
            raise ValueError(f"failed to decode {label}")
        tensor = self._torch.from_dlpack(frame).clone()
        if (
            tensor.ndim != 3
            or tensor.shape[-1] != 3
            or tensor.dtype != self._torch.uint8
        ):
            raise ValueError(f"{label} did not decode as HWC RGB uint8")
        return tensor

    def _resize_scale_reformat(self, frames: Sequence[Any], state: Any) -> Any:
        shapes = {tuple(frame.shape) for frame in frames}
        if len(shapes) == 1:
            nchw = self._resize_batch(self._torch.stack(list(frames)), state)
        else:
            nchw = self._torch.cat(
                [self._resize_batch(frame.unsqueeze(0), state) for frame in frames]
            )
        return nchw.unsqueeze(0)

    def _resize_batch(self, frames: Any, state: Any) -> Any:
        batch_size = frames.shape[0]
        tensor = cvcuda.as_tensor(frames.contiguous(), "NHWC")
        tensor = cvcuda.pillowresize(
            tensor,
            (batch_size, self._resolution, self._resolution, 3),
            cvcuda.Format.RGB8,
            cvcuda.Interp.LINEAR,
            stream=state.cvcuda_stream,
        )
        tensor = cvcuda.convertto(
            tensor,
            cvcuda.Type.F32,
            scale=1.0 / 255.0,
            stream=state.cvcuda_stream,
        )
        tensor = cvcuda.reformat(
            tensor,
            "NCHW",
            stream=state.cvcuda_stream,
        )
        output = self._torch.as_tensor(tensor.cuda(), device=self._device)
        if self._dtype is not None:
            output = output.to(dtype=self._dtype)
        return output

    def _stream_state(self) -> Any:
        state = self._tls
        if not hasattr(state, "cvcuda_stream"):
            with self._torch.cuda.device(self._device):
                state.cvcuda_stream = cvcuda.Stream()
                state.torch_stream = self._torch.cuda.ExternalStream(
                    state.cvcuda_stream.handle,
                    device=self._device,
                )
        return state
