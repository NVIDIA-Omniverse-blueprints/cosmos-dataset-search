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

"""Tests for the CVDS-owned Cosmos-Embed OSS backend boundary."""

from __future__ import annotations

import base64
import builtins
import importlib.util
import io
import math
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

from src import cosmos_embed_oss as ce1_defaults
from src.cosmos_embed_oss import backends
from src.cosmos_embed_oss import server as ce1_server
from src.haystack.components.video import cosmos_video_embedder

EMBEDDING_DIMENSION = ce1_defaults.DEFAULT_EMBEDDING_DIMENSION
MODEL_NAME = ce1_defaults.DEFAULT_MODEL_NAME


class _ContractTestBackend:
    name = "contract-test"

    def ready(self) -> None:
        return None

    def embed(
        self,
        inputs: list[str],
        *,
        request_type: str,
        input_kinds: list[str],
    ) -> list[list[float]]:
        del request_type, input_kinds
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in inputs]


@pytest.fixture(autouse=True)
def reset_backend_cache() -> Iterator[None]:
    backends.reset_embedding_backend_for_tests()
    yield
    backends.reset_embedding_backend_for_tests()


def test_default_backend_is_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "COSMOS_EMBED_BACKEND",
        "COSMOS_EMBED_MODEL_PATH",
        "COSMOS_EMBED_WEIGHTS_DIR",
        "COSMOS_EMBED_MODEL_VARIANT",
        "COSMOS_EMBED_ALLOW_HF_DOWNLOAD",
        "COSMOS_EMBED_HF_MODEL_ID",
        "COSMOS_EMBED_GPU_VIDEO",
    ):
        monkeypatch.delenv(name, raising=False)

    config = backends.backend_config_from_env()
    assert config.mode == "pytorch"
    assert config.allow_hf_download is False
    assert config.gpu_video is True
    assert backends.BackendConfig().gpu_video is True
    assert config.hf_model_id == backends.DEFAULT_HF_MODEL_ID
    assert backends.backend_name_for_config(config) == "pytorch-ce1-224p"
    with pytest.raises(backends.EmbeddingBackendError, match="MODEL_PATH.*WEIGHTS_DIR"):
        backends.get_embedding_backend()


def test_pytorch_backend_rejects_explicit_cpu_video_mode(tmp_path: Path) -> None:
    with pytest.raises(backends.EmbeddingBackendError) as error:
        backends.PyTorchCosmosEmbedBackend(
            backends.BackendConfig(model_path=str(tmp_path), gpu_video=False)
        )

    assert str(error.value) == (
        "CE1 requires GPU video decoding; "
        "COSMOS_EMBED_GPU_VIDEO=false is not supported"
    )


def test_pytorch_backend_rejects_cpu_video_mode_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.setenv("COSMOS_EMBED_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("COSMOS_EMBED_GPU_VIDEO", "false")

    assert backends.backend_config_from_env().gpu_video is False
    with pytest.raises(backends.EmbeddingBackendError) as error:
        backends.get_embedding_backend()

    assert str(error.value) == (
        "CE1 requires GPU video decoding; "
        "COSMOS_EMBED_GPU_VIDEO=false is not supported"
    )


def test_pytorch_backend_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.delenv("COSMOS_EMBED_MODEL_PATH", raising=False)
    monkeypatch.delenv("COSMOS_EMBED_WEIGHTS_DIR", raising=False)

    with pytest.raises(backends.EmbeddingBackendError, match="MODEL_PATH.*WEIGHTS_DIR"):
        backends.get_embedding_backend()


def test_pytorch_backend_rejects_missing_model_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-ce1-weights"

    with pytest.raises(backends.EmbeddingBackendError, match="existing directory"):
        backends.create_embedding_backend(
            backends.BackendConfig(mode="pytorch", model_path=str(missing_path))
        )


def test_pytorch_backend_forces_local_files_only_for_local_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(monkeypatch, [3.0, 4.0] + [0.0] * 254, capture=capture)

    backend = backends.create_embedding_backend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            local_files_only=False,
        )
    )

    backend.ready()

    calls = capture["from_pretrained_calls"]
    assert [call["kind"] for call in calls] == ["model", "processor"]
    assert all(call["args"] == (str(tmp_path),) for call in calls)
    assert all(call["kwargs"]["local_files_only"] is True for call in calls)
    assert all("token" not in call["kwargs"] for call in calls)


def test_pytorch_backend_allows_hf_model_only_when_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(monkeypatch, [3.0, 4.0] + [0.0] * 254, capture=capture)
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token")

    backend = backends.create_embedding_backend(
        backends.BackendConfig(
            mode="pytorch",
            allow_hf_download=True,
            hf_model_id="nvidia/Cosmos-Embed1-224p",
        )
    )

    backend.ready()

    calls = capture["from_pretrained_calls"]
    assert [call["kind"] for call in calls] == ["model", "processor"]
    assert all(call["args"] == ("nvidia/Cosmos-Embed1-224p",) for call in calls)
    assert all(call["kwargs"]["local_files_only"] is False for call in calls)
    assert all(call["kwargs"]["token"] is True for call in calls)


def test_pytorch_backend_allows_hf_model_from_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    cache_root = tmp_path / "hf-hub"
    cached_model = cache_root / "models--nvidia--Cosmos-Embed1-224p" / "snapshots"
    cached_model.mkdir(parents=True)
    install_fake_ce1_runtime(monkeypatch, [3.0, 4.0] + [0.0] * 254, capture=capture)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(backends, "_huggingface_cache_roots", lambda: [cache_root])

    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", allow_hf_download=True)
    )

    backend.ready()

    calls = capture["from_pretrained_calls"]
    assert all(call["args"] == ("nvidia/Cosmos-Embed1-224p",) for call in calls)
    assert all(call["kwargs"]["local_files_only"] is True for call in calls)
    assert all("token" not in call["kwargs"] for call in calls)


def test_pytorch_backend_rejects_hf_without_auth_or_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        backends,
        "_huggingface_cache_roots",
        lambda: [tmp_path / "empty-hf-hub"],
    )

    with pytest.raises(backends.EmbeddingBackendError, match="requires HF_TOKEN"):
        backends.create_embedding_backend(
            backends.BackendConfig(mode="pytorch", allow_hf_download=True)
        )


def test_pytorch_backend_requires_224p_for_256d_cvds_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(backends.EmbeddingBackendError, match="224p 256-d weights"):
        backends.create_embedding_backend(
            backends.BackendConfig(
                mode="pytorch",
                model_path=str(tmp_path),
                variant="336p",
            )
        )


def test_pytorch_backend_does_not_import_model_deps_until_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    imported: list[str] = []

    def tracked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "transformers", "PyNvVideoCodec"}:
            imported.append(name)
            raise AssertionError(f"{name} should not be imported yet")
        return original_import(name, *args, **kwargs)

    def unexpected_runtime_import(**_kwargs: Any) -> dict[str, Any]:
        pytest.fail("runtime dependencies must remain lazy until readiness")

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.setattr(
        backends.runtime_check,
        "import_pytorch_runtime_modules",
        unexpected_runtime_import,
    )

    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", model_path=str(tmp_path))
    )

    assert isinstance(backend, backends.PyTorchCosmosEmbedBackend)
    assert imported == []


def test_pytorch_backend_reports_missing_runtime_deps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_runtime_modules(*, module_names: tuple[str, ...]) -> dict[str, Any]:
        assert module_names == (
            "PyNvVideoCodec",
            *backends.runtime_check.REQUIRED_PYTORCH_MODULES,
        )
        assert "av" not in module_names
        raise backends.runtime_check.RuntimeDependencyError(
            "PyTorch CE1 backend requires runtime dependencies"
        )

    monkeypatch.setattr(
        backends.runtime_check,
        "import_pytorch_runtime_modules",
        missing_runtime_modules,
    )
    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", model_path=str(tmp_path))
    )

    with pytest.raises(
        backends.EmbeddingBackendError, match="requires runtime dependencies"
    ):
        backend.ready()


@pytest.mark.parametrize(
    "device,cuda_available",
    [(None, False), ("cuda", False), ("cuda:0", False), ("cpu", True), ("mps", True)],
)
def test_pytorch_readiness_rejects_non_cuda_runtime_before_model_load(
    device: str | None,
    cuda_available: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(
        monkeypatch,
        [1.0] + [0.0] * 255,
        capture=capture,
        cuda_available=cuda_available,
    )
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(model_path=str(tmp_path), device=device)
    )
    try:
        with pytest.raises(backends.EmbeddingBackendError, match="requires a CUDA GPU"):
            backend.ready()
    finally:
        backend.close()

    assert capture.get("model_load_count", 0) == 0
    assert capture.get("processor_load_count", 0) == 0
    assert capture.get("gpu_processor_init_count", 0) == 0


def test_pytorch_readiness_requires_pynvvideocodec_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(monkeypatch, [1.0] + [0.0] * 255, capture=capture)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(model_path=str(tmp_path))
    )
    try:
        with pytest.raises(backends.EmbeddingBackendError, match="PyNvVideoCodec"):
            backend.ready()
    finally:
        backend.close()

    assert capture.get("model_load_count", 0) == 0
    assert capture.get("processor_load_count", 0) == 0
    assert capture.get("gpu_processor_init_count", 0) == 0


@pytest.mark.parametrize("gpu_frames", [False, True])
def test_pytorch_gpu_runtime_is_ready_without_pyav(
    gpu_frames: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(monkeypatch, [1.0] + [0.0] * 255, capture=capture)
    # An installed PyAV on the test host must not hide a runtime dependency.
    assert sys.modules["av"] is None
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(model_path=str(tmp_path), gpu_frames=gpu_frames)
    )
    try:
        backend.ready()
        backend.ready()
    finally:
        backend.close()

    assert capture["model_load_count"] == 1
    assert capture["processor_load_count"] == 1
    assert capture["gpu_processor_init_count"] == 1
    assert capture["gpu_device"] == "cuda"
    assert "av" not in backends.runtime_check.REQUIRED_PYTORCH_MODULES


def test_pytorch_batch_move_preserves_integer_token_tensors() -> None:
    token_ids = _CapturingTensor(is_floating=False)
    pixel_values = _CapturingTensor(is_floating=True)

    moved = backends._move_batch_to_device(
        {"input_ids": token_ids, "pixel_values": pixel_values},
        device="cuda",
        dtype="bfloat16",
    )

    assert moved == {"input_ids": token_ids, "pixel_values": pixel_values}
    assert token_ids.calls == [(("cuda",), {})]
    assert pixel_values.calls == [(("cuda",), {"dtype": "bfloat16"})]


def test_cuda_dtype_matches_device_bfloat16_support() -> None:
    supported = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_bf16_supported=lambda: True),
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )
    unsupported = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_bf16_supported=lambda: False),
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )

    assert backends._default_torch_dtype(supported, "cuda") == "bfloat16"
    assert backends._default_torch_dtype(unsupported, "cuda:1") == "float16"
    assert backends._default_torch_dtype(supported, "cpu") == "float32"


@pytest.mark.parametrize("separate_bias", [False, True])
def test_sdpa_attention_matches_ce1_attention(separate_bias: bool) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 2
            self.scale = 0.5
            self.qkv = torch.nn.Linear(8, 24, bias=not separate_bias)
            if separate_bias:
                self.q_bias = torch.nn.Parameter(torch.randn(8))
                self.v_bias = torch.nn.Parameter(torch.randn(8))
            self.window_size = (2, 2)
            self.relative_position_bias_table = torch.nn.Parameter(
                torch.randn(12, self.num_heads)
            )
            self.register_buffer(
                "relative_position_index", torch.randint(0, 12, (5, 5))
            )
            self.attn_drop = torch.nn.Dropout(0.0)
            self.proj = torch.nn.Linear(8, 8)
            self.proj_drop = torch.nn.Dropout(0.0)

        def forward(self, value: Any, rel_pos_bias: Any | None = None) -> Any:
            batch_size, token_count, _ = value.shape
            qkv_bias = self.qkv.bias
            if separate_bias:
                qkv_bias = torch.cat(
                    (
                        self.q_bias,
                        torch.zeros_like(self.v_bias, requires_grad=False),
                        self.v_bias,
                    )
                )
            query, key, projected_value = (
                torch.nn.functional.linear(value, self.qkv.weight, qkv_bias)
                .reshape(batch_size, token_count, 3, self.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            scores = (query * self.scale) @ key.transpose(-2, -1)
            relative_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(token_count, token_count, -1)
            scores += relative_bias.permute(2, 0, 1).unsqueeze(0)
            if rel_pos_bias is not None:
                scores += rel_pos_bias
            output = scores.softmax(dim=-1) @ projected_value
            output = output.transpose(1, 2).reshape(batch_size, token_count, -1)
            return self.proj_drop(self.proj(output))

    attention = Attention().eval()
    model = types.SimpleNamespace(
        visual_encoder=types.SimpleNamespace(
            blocks=[types.SimpleNamespace(attn=attention)]
        )
    )
    values = torch.randn(2, 5, 8)
    shared_bias = torch.randn(1, 2, 5, 5)
    expected = attention(values, shared_bias)

    backends._enable_sdpa_attention(model, torch)
    actual = attention(values, shared_bias)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gpu_preparation_reuses_worker_for_singleton_requests(tmp_path: Path) -> None:
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            gpu_frames=True,
            preparation_workers=1,
        )
    )
    worker_threads: list[int] = []

    def prepare(_self: Any, _runtime: Any, value: str) -> str:
        worker_threads.append(threading.get_ident())
        return value

    backend._prepare_video_input = types.MethodType(prepare, backend)
    try:
        first = backend._prepare_video_batch(object(), ["first"])
        second = backend._prepare_video_batch(object(), ["second"])
    finally:
        backend.close()

    assert first == ["first"]
    assert second == ["second"]
    assert len(set(worker_threads)) == 1
    assert worker_threads[0] != threading.get_ident()


def test_pytorch_text_backend_normalizes_256d_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ce1_runtime(monkeypatch, [3.0, 4.0] + [0.0] * 254)
    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", model_path=str(tmp_path))
    )

    embeddings = backend.embed(
        ["left turn"],
        request_type="query",
        input_kinds=["text"],
    )

    assert len(embeddings) == 1
    assert len(embeddings[0]) == EMBEDDING_DIMENSION
    assert embeddings[0][0] == pytest.approx(0.6)
    assert embeddings[0][1] == pytest.approx(0.8)
    assert math.sqrt(sum(value * value for value in embeddings[0])) == pytest.approx(
        1.0
    )


def test_pytorch_text_backend_batches_model_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(
        monkeypatch,
        [3.0, 4.0] + [0.0] * 254,
        capture=capture,
    )
    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", model_path=str(tmp_path))
    )

    embeddings = backend.embed(
        ["left turn", "pedestrian crossing"],
        request_type="bulk_text",
        input_kinds=["text", "text"],
    )

    assert len(embeddings) == 2
    assert capture["text_batches"] == [["left turn", "pedestrian crossing"]]
    assert capture["text_model_call_count"] == 1


def test_backend_cache_initializes_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    def create_once(_config: backends.BackendConfig) -> Any:
        time.sleep(0.01)
        backend = types.SimpleNamespace(close=lambda: None)
        created.append(backend)
        return backend

    monkeypatch.setattr(backends, "create_embedding_backend", create_once)
    barrier = threading.Barrier(8)
    results: list[Any] = []

    def resolve_backend() -> None:
        barrier.wait()
        results.append(backends.get_embedding_backend())

    threads = [threading.Thread(target=resolve_backend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(created) == 1
    assert len(results) == 8
    assert all(result is created[0] for result in results)


def test_pytorch_runtime_initializes_once_across_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(
        monkeypatch,
        [3.0, 4.0] + [0.0] * 254,
        capture=capture,
    )
    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", model_path=str(tmp_path))
    )
    barrier = threading.Barrier(8)

    def mark_ready() -> None:
        barrier.wait()
        backend.ready()

    threads = [threading.Thread(target=mark_ready) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert capture["model_load_count"] == 1
    assert capture["processor_load_count"] == 1


def test_pytorch_backend_rejects_wrong_projection_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ce1_runtime(monkeypatch, [1.0] * (EMBEDDING_DIMENSION + 1))
    backend = backends.create_embedding_backend(
        backends.BackendConfig(mode="pytorch", model_path=str(tmp_path))
    )

    with pytest.raises(backends.EmbeddingBackendError, match="expected 256-d"):
        backend.embed(
            ["left turn"],
            request_type="query",
            input_kinds=["text"],
        )


def test_pytorch_backend_embeds_base64_video_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    media_tmp_dir = tmp_path / "media-tmp"

    def decode_video(media_path: Path) -> _FakeTensor:
        capture["base64_media_exists_during_decode"] = media_path.exists()
        capture["base64_media_bytes"] = media_path.read_bytes()
        capture["base64_media_suffix"] = media_path.suffix
        return _fake_video_tensor(count=8)

    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[3.0, 4.0] + [0.0] * 254,
        capture=capture,
        video_decoder=decode_video,
    )

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(media_tmp_dir),
        )
    )

    embeddings = backend.embed(
        [_video_base64_data_uri(b"fake-video")],
        request_type="query",
        input_kinds=["video"],
    )

    assert len(embeddings) == 1
    assert embeddings[0][0] == pytest.approx(0.6)
    assert embeddings[0][1] == pytest.approx(0.8)
    assert len(embeddings[0]) == EMBEDDING_DIMENSION
    assert capture["base64_media_exists_during_decode"] is True
    assert capture["base64_media_bytes"] == b"fake-video"
    assert capture["base64_media_suffix"] == ".mp4"
    assert capture["gpu_video_shapes"] == [(1, 8, 3, 2, 2)]
    assert "processed_video_shape" not in capture
    assert capture["video_model_called"] is True
    assert _directory_entries(media_tmp_dir) == []


def test_gpu_video_preparation_uses_materialized_file(tmp_path: Path) -> None:
    media_tmp_dir = tmp_path / "media-tmp"
    expected = object()
    decoded: dict[str, Any] = {}

    def decode_video(path: Path) -> object:
        decoded["payload"] = path.read_bytes()
        return expected

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(media_tmp_dir),
            gpu_video=True,
        )
    )
    runtime = types.SimpleNamespace(
        processor=types.SimpleNamespace(num_video_frames=8),
        gpu_media=types.SimpleNamespace(decode_video=decode_video),
    )
    try:
        result = backend._prepare_video_input(
            runtime,
            _video_base64_data_uri(b"fake-video"),
        )
    finally:
        backend.close()

    assert result is expected
    assert decoded["payload"] == b"fake-video"
    assert _directory_entries(media_tmp_dir) == []


def test_gpu_video_decode_failure_reports_client_error(tmp_path: Path) -> None:
    """A GPU-undecodable video must return a clear client error, not crash.

    GPU mode does not silently fall back to CPU decoding. The request is
    rejected with an actionable message instead.
    """

    def decode_video(_path: Path) -> object:
        raise RuntimeError("GPU video decoder could not open the media")

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
            gpu_video=True,
        )
    )
    runtime = types.SimpleNamespace(
        processor=types.SimpleNamespace(num_video_frames=8),
        gpu_media=types.SimpleNamespace(decode_video=decode_video),
    )
    try:
        with pytest.raises(backends.EmbeddingInputError) as error:
            backend._prepare_video_input(
                runtime,
                _video_base64_data_uri(b"fake-video"),
            )
    finally:
        backend.close()

    assert "NVDEC-supported codec" in str(error.value)


def test_gpu_video_safety_rejection_preserves_actionable_detail(tmp_path: Path) -> None:
    """Known native crash cases must return the preflight client error."""

    def decode_video(_path: Path) -> object:
        raise backends.GpuDecodeUnsupported(
            "VP9 in Matroska/WebM is not supported by the GPU random-access decoder"
        )

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
            gpu_video=True,
        )
    )
    runtime = types.SimpleNamespace(
        processor=types.SimpleNamespace(num_video_frames=8),
        gpu_media=types.SimpleNamespace(decode_video=decode_video),
    )
    try:
        with pytest.raises(backends.EmbeddingInputError) as error:
            backend._prepare_video_input(
                runtime,
                _video_base64_data_uri(b"fake-video"),
            )
    finally:
        backend.close()

    assert "VP9 in Matroska/WebM" in str(error.value)


def test_gpu_frame_decode_failure_reports_client_error(tmp_path: Path) -> None:
    """A GPU-undecodable frame image must return a clear client error."""

    def decode_frames(_payloads: Any) -> object:
        raise ValueError("unsupported image")

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
            gpu_frames=True,
        )
    )
    runtime = types.SimpleNamespace(
        processor=types.SimpleNamespace(num_video_frames=8),
        gpu_media=types.SimpleNamespace(decode_frames=decode_frames),
    )
    try:
        with pytest.raises(backends.EmbeddingInputError) as error:
            backend._prepare_video_input(runtime, _video_frames_base64_data_uri(1))
    finally:
        backend.close()

    assert "GPU decoder does not support" in str(error.value)


def test_gpu_decoder_frame_count_rejects_unusable_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame sampling must not trust an unknown or empty decoder length."""

    # This checks decoder metadata only; importing native GPU libraries is unnecessary.
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = []
    fake_nvidia.nvimgcodec = types.ModuleType("nvidia.nvimgcodec")
    monkeypatch.setitem(sys.modules, "cvcuda", types.ModuleType("cvcuda"))
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.nvimgcodec", fake_nvidia.nvimgcodec)
    spec = importlib.util.spec_from_file_location(
        "ce1_gpu_frame_count_test",
        Path(backends.__file__).with_name("gpu_pipeline.py"),
    )
    assert spec is not None and spec.loader is not None
    gpu_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_pipeline)
    processor = gpu_pipeline.GpuMediaProcessor.__new__(gpu_pipeline.GpuMediaProcessor)

    class _NoLength:
        pass

    class _Length:
        def __init__(self, value: int) -> None:
            self._value = value

        def __len__(self) -> int:
            return self._value

    assert processor._decoder_frame_count(_Length(120)) == 120
    # A single-frame video is valid and must still sample.
    assert processor._decoder_frame_count(_Length(1)) == 1

    with pytest.raises(gpu_pipeline.GpuDecodeUnsupported):
        processor._decoder_frame_count(_Length(0))
    with pytest.raises(gpu_pipeline.GpuDecodeUnsupported):
        processor._decoder_frame_count(_NoLength())


def test_pytorch_backend_downloads_presigned_video_batch_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    media_tmp_dir = tmp_path / "media-tmp"
    download_urls: list[str] = []
    downloaded_payloads: list[bytes] = []

    def media_downloader(
        url: str,
        destination_path: Path,
        timeout_seconds: float,
        max_bytes: int,
    ) -> None:
        download_urls.append(url)
        assert timeout_seconds == pytest.approx(9.0)
        assert max_bytes == backends.DEFAULT_MAX_MEDIA_BYTES
        destination_path.write_bytes(f"downloaded:{url}".encode("utf-8"))

    def decode_video(media_path: Path) -> _FakeTensor:
        downloaded_payloads.append(media_path.read_bytes())
        return _fake_video_tensor(count=8)

    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[5.0, 12.0] + [0.0] * 254,
        capture=capture,
        video_decoder=decode_video,
    )

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(media_tmp_dir),
            media_download_timeout_seconds=9.0,
        ),
        media_downloader=media_downloader,
    )

    embeddings = backend.embed(
        [
            "data:video/mp4;presigned_url,https://example.test/one.mp4?token=1",
            "data:video/mp4;presigned_url,https://example.test/two.mp4?token=2",
        ],
        request_type="bulk_video",
        input_kinds=["video", "video"],
    )

    assert len(embeddings) == 2
    assert embeddings[0][0] == pytest.approx(5.0 / 13.0)
    assert embeddings[0][1] == pytest.approx(12.0 / 13.0)
    assert download_urls == [
        "https://example.test/one.mp4?token=1",
        "https://example.test/two.mp4?token=2",
    ]
    assert downloaded_payloads == [
        b"downloaded:https://example.test/one.mp4?token=1",
        b"downloaded:https://example.test/two.mp4?token=2",
    ]
    assert capture["video_model_call_count"] == 1
    assert capture["video_model_batch_size"] == 2
    assert _directory_entries(media_tmp_dir) == []


def test_pytorch_backend_batches_pil_video_frames_in_one_model_call_without_pyav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[5.0, 12.0] + [0.0] * 254,
        capture=capture,
    )
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
            gpu_frames=False,
        )
    )

    embeddings = backend.embed(
        [_video_frames_base64_data_uri(10), _video_frames_base64_data_uri(20)],
        request_type="bulk_video",
        input_kinds=["video", "video"],
    )

    assert len(embeddings) == 2
    assert capture["processed_video_shapes"] == [
        (1, 8, 3, 2, 2),
        (1, 8, 3, 2, 2),
    ]
    assert capture["video_model_call_count"] == 1
    assert capture["video_model_batch_size"] == 2
    assert "gpu_video_shapes" not in capture
    assert "gpu_frame_payloads" not in capture
    assert sys.modules["av"] is None
    assert _directory_entries(tmp_path / "media-tmp") == []


def test_pytorch_backend_downloads_presigned_video_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    downloaded_urls: list[str] = []
    frame_bytes = _png_frame_bytes(42)
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[3.0, 4.0] + [0.0] * 254,
        capture=capture,
    )

    def media_downloader(
        url: str,
        destination_path: Path,
        _timeout_seconds: float,
        max_bytes: int,
    ) -> None:
        assert len(frame_bytes) <= max_bytes
        downloaded_urls.append(url)
        destination_path.write_bytes(frame_bytes)

    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
        ),
        media_downloader=media_downloader,
    )
    urls = [f"https://media.example.test/frame-{index}.png" for index in range(8)]

    embeddings = backend.embed(
        [_video_frames_presigned_data_uri(urls)],
        request_type="query",
        input_kinds=["video"],
    )

    assert len(embeddings) == 1
    assert downloaded_urls == urls
    assert capture["video_model_batch_size"] == 1
    assert _directory_entries(tmp_path / "media-tmp") == []


def test_video_frames_parser_requires_exactly_eight_frames() -> None:
    malformed = "data:video_frames/png;base64,{" + ",".join(["ZmFrZQ=="] * 7) + "}"

    with pytest.raises(backends.EmbeddingInputError, match="exactly 8"):
        backends.parse_video_frames_data_uri(malformed)


@pytest.mark.parametrize(
    "total_frames,expected",
    [
        (1, [0, 0, 0, 0, 0, 0, 0, 0]),
        (2, [0, 0, 0, 0, 0, 0, 0, 1]),
        (7, [0, 0, 1, 2, 3, 4, 5, 6]),
        (8, [0, 1, 2, 3, 4, 5, 6, 7]),
        (17, [0, 2, 4, 6, 9, 11, 13, 16]),
    ],
)
def test_ce1_frame_sampling_matches_numpy_linspace_flooring(
    total_frames: int,
    expected: list[int],
) -> None:
    indices = backends._linspace_frame_indices(total_frames, 8)

    assert indices == expected
    assert len(indices) == 8
    assert indices == sorted(indices)
    assert all(0 <= index < total_frames for index in indices)


def test_base64_video_respects_decoded_media_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[3.0, 4.0] + [0.0] * 254,
    )
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
            max_media_bytes=4,
        )
    )

    with pytest.raises(backends.EmbeddingInputError, match="maximum 4 bytes"):
        backend.embed(
            [_video_base64_data_uri(b"five!")],
            request_type="query",
            input_kinds=["video"],
        )

    assert _directory_entries(tmp_path / "media-tmp") == []


@pytest.mark.parametrize(
    "url, message",
    [
        ("http://example.com/video.mp4", "must use https"),
        ("https://user:secret@example.com/video.mp4", "credentials"),
        ("https://127.0.0.1/video.mp4", "private or reserved"),
        ("https://169.254.169.254/latest/meta-data", "private or reserved"),
    ],
)
def test_presigned_url_validation_rejects_unsafe_targets(
    url: str,
    message: str,
) -> None:
    with pytest.raises(backends.EmbeddingInputError, match=message):
        backends._validate_presigned_download_url(url)


def test_presigned_url_validation_rejects_invalid_port_and_redirect_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(backends.EmbeddingInputError, match="invalid port"):
        backends._validate_presigned_download_url(
            "https://example.com:not-a-port/video.mp4"
        )

    request = types.SimpleNamespace(full_url="https://example.com/video.mp4")
    handler = backends._ValidatingRedirectHandler()
    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://127.0.0.1/internal",
        )


def test_presigned_url_validation_checks_allowlist_and_all_dns_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ALLOWED_HOSTS_ENV,
        "*.example.com",
    )
    monkeypatch.setattr(
        backends.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ],
    )

    backends._validate_presigned_download_url("https://assets.example.com/video.mp4")
    with pytest.raises(backends.EmbeddingInputError, match="allowlist"):
        backends._validate_presigned_download_url(
            "https://assets.example.net/video.mp4"
        )

    monkeypatch.setattr(
        backends.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("10.0.0.2", 443)),
        ],
    )
    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        backends._validate_presigned_download_url(
            "https://assets.example.com/video.mp4"
        )


def test_operator_endpoint_override_rewrites_local_emulator_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ENDPOINT_ENV,
        "http://localstack:4566",
    )

    rewritten = backends._presigned_video_download_url(
        "http://localhost:4566/bucket/video.mp4?signature=abc"
    )

    assert rewritten == ("http://localstack:4566/bucket/video.mp4?signature=abc")
    backends._validate_presigned_download_url(rewritten)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8680/video.mp4",
        "http://host.docker.internal:8680/video.mp4",
    ],
)
def test_operator_endpoint_override_preserves_local_server_on_other_port(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ENDPOINT_ENV,
        "http://localstack:4566",
    )
    assert backends._presigned_video_download_url(url) == url


def test_presigned_url_validation_allows_exact_configured_http_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV,
        "http://host.docker.internal:8680,http://cds-frames-http:8680/",
    )

    backends._validate_presigned_download_url(
        "http://host.docker.internal:8680/video.mp4"
    )
    backends._validate_presigned_download_url("http://cds-frames-http:8680/video.mp4")
    with pytest.raises(backends.EmbeddingInputError, match="exact HTTP origin"):
        backends._validate_presigned_download_url(
            "http://host.docker.internal:8681/video.mp4"
        )


def test_presigned_url_validation_rejects_invalid_http_origin_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV,
        "http://host.docker.internal:8680/not-an-origin",
    )

    with pytest.raises(backends.EmbeddingBackendError, match="exact http"):
        backends._validate_presigned_download_url(
            "http://host.docker.internal:8680/video.mp4"
        )


@pytest.mark.parametrize(
    "url",
    [
        (
            "https://raw.githubusercontent.com/nvidia-cosmos/cosmos-transfer1/"
            "main/assets/example1_input_video.mp4"
        ),
        "https://bucket.s3.amazonaws.com/path/video.mp4?signature=abc",
        "https://storage.googleapis.com/example/video.mp4",
    ],
)
def test_operator_endpoint_override_does_not_rewrite_public_https(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ENDPOINT_ENV,
        "http://localstack:4566",
    )

    assert backends._presigned_video_download_url(url) == url


def test_remote_copy_stops_at_configured_byte_limit() -> None:
    response = io.BytesIO(b"12345")
    output = io.BytesIO()

    with pytest.raises(backends.EmbeddingInputError, match="maximum 4 bytes"):
        backends._copy_response_with_limit(response, output, max_bytes=4)

    assert output.getvalue() == b""


def test_presigned_download_rejects_truncated_body_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShortResponse(io.BytesIO):
        status = 200
        headers = {"Content-Length": "10"}

    opener = types.SimpleNamespace(
        open=lambda *_args, **_kwargs: ShortResponse(b"short")
    )
    monkeypatch.setattr(
        backends,
        "_validate_presigned_download_url",
        lambda _url: None,
    )
    monkeypatch.setattr(
        backends.urllib.request,
        "build_opener",
        lambda *_handlers: opener,
    )
    media_tmp_dir = tmp_path / "media-tmp"

    with pytest.raises(backends.MediaDownloadError, match="Content-Length"):
        with backends.materialize_video_data_uri(
            "data:video/mp4;presigned_url,https://example.test/video.mp4?sig=1",
            tmp_dir=media_tmp_dir,
            download_timeout_seconds=5.0,
            max_media_bytes=1024,
        ):
            pass

    assert _directory_entries(media_tmp_dir) == []


def test_pytorch_backend_cleans_temp_files_when_video_decode_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_tmp_dir = tmp_path / "media-tmp"
    captured_path: Path | None = None

    def failing_gpu_decoder(media_path: Path) -> _FakeTensor:
        nonlocal captured_path
        captured_path = media_path
        assert media_path.exists()
        raise RuntimeError("decoder failed")

    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[3.0, 4.0] + [0.0] * 254,
        video_decoder=failing_gpu_decoder,
    )
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(media_tmp_dir),
        )
    )

    with pytest.raises(backends.EmbeddingInputError, match="NVDEC-supported codec"):
        backend.embed(
            [_video_base64_data_uri(b"bad-video")],
            request_type="query",
            input_kinds=["video"],
        )

    assert captured_path is not None
    assert not captured_path.exists()
    assert _directory_entries(media_tmp_dir) == []


def test_pytorch_backend_rejects_invalid_base64_video_without_temp_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_tmp_dir = tmp_path / "media-tmp"
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[3.0, 4.0] + [0.0] * 254,
    )
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(media_tmp_dir),
        )
    )

    with pytest.raises(backends.EmbeddingBackendError, match="invalid base64"):
        backend.embed(
            ["data:video/mp4;base64,not-valid-base64%%"],
            request_type="query",
            input_kinds=["video"],
        )

    assert _directory_entries(media_tmp_dir) == []


def test_build_embeddings_response_uses_backend_without_changing_contract() -> None:
    request = ce1_server.EmbeddingsRequest(
        encoding_format="float",
        inputs=["left turn", "pedestrian crossing"],
        model=MODEL_NAME,
        request_type="query",
    )
    backend = _ContractTestBackend()

    body = ce1_server.build_embeddings_response(request, backend=backend)

    assert body["object"] == "list"
    assert body["model"] == MODEL_NAME
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert all(len(item["embedding"]) == EMBEDDING_DIMENSION for item in body["data"])
    assert body["usage"]["num_videos"] == 0


def test_build_embeddings_response_uses_pytorch_video_backend_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[8.0, 15.0] + [0.0] * 254,
    )
    backend = backends.PyTorchCosmosEmbedBackend(
        backends.BackendConfig(
            mode="pytorch",
            model_path=str(tmp_path),
            media_tmp_dir=str(tmp_path / "media-tmp"),
        )
    )
    request = ce1_server.EmbeddingsRequest(
        encoding_format="float",
        inputs=[_video_base64_data_uri(b"fake-video")],
        model=MODEL_NAME,
        request_type="query",
    )

    body = ce1_server.build_embeddings_response(request, backend=backend)

    assert body["object"] == "list"
    assert body["model"] == MODEL_NAME
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["embedding"][0] == pytest.approx(8.0 / 17.0)
    assert body["data"][0]["embedding"][1] == pytest.approx(15.0 / 17.0)
    assert len(body["data"][0]["embedding"]) == EMBEDDING_DIMENSION
    assert body["usage"]["num_videos"] == 1


def test_existing_cvds_client_can_call_pytorch_presigned_video_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_tmp_dir = tmp_path / "media-tmp"
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[7.0, 24.0] + [0.0] * 254,
    )
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.setenv("COSMOS_EMBED_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("COSMOS_EMBED_TMP_DIR", str(media_tmp_dir))
    monkeypatch.setattr(
        backends,
        "download_presigned_video",
        lambda _url, path, _timeout, _max_bytes: path.write_bytes(b"downloaded-video"),
    )
    ce1_http_server = ce1_server.run_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=ce1_http_server.serve_forever, daemon=True)
    thread.start()
    host, port = ce1_http_server.server_address

    try:
        client = cosmos_video_embedder.CosmosEmbedClient(f"http://{host}:{port}")
        embeddings = client.embed_videos(
            [
                "data:video/mp4;presigned_url,https://example.test/one.mp4",
                "data:video/mp4;presigned_url,https://example.test/two.mp4",
            ]
        )
    finally:
        ce1_http_server.shutdown()
        ce1_http_server.server_close()
        thread.join(timeout=5)

    assert len(embeddings) == 2
    assert embeddings[0][0] == pytest.approx(7.0 / 25.0)
    assert embeddings[0][1] == pytest.approx(24.0 / 25.0)
    assert all(len(embedding) == EMBEDDING_DIMENSION for embedding in embeddings)
    assert _directory_entries(media_tmp_dir) == []


def test_existing_cvds_client_can_call_pytorch_base64_video_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_tmp_dir = tmp_path / "media-tmp"
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[9.0, 40.0] + [0.0] * 254,
    )
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.setenv("COSMOS_EMBED_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("COSMOS_EMBED_TMP_DIR", str(media_tmp_dir))
    ce1_http_server = ce1_server.run_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=ce1_http_server.serve_forever, daemon=True)
    thread.start()
    host, port = ce1_http_server.server_address

    try:
        client = cosmos_video_embedder.CosmosEmbedClient(f"http://{host}:{port}")
        embeddings = client.embed_videos([_video_base64_data_uri(b"fake-video")])
    finally:
        ce1_http_server.shutdown()
        ce1_http_server.server_close()
        thread.join(timeout=5)

    assert len(embeddings) == 1
    assert embeddings[0][0] == pytest.approx(9.0 / 41.0)
    assert embeddings[0][1] == pytest.approx(40.0 / 41.0)
    assert len(embeddings[0]) == EMBEDDING_DIMENSION
    assert _directory_entries(media_tmp_dir) == []


def test_pytorch_http_invalid_video_returns_non_2xx_without_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ce1_runtime(
        monkeypatch,
        [0.0, 1.0] + [0.0] * 254,
        video_projected_embedding=[3.0, 4.0] + [0.0] * 254,
    )
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.setenv("COSMOS_EMBED_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("COSMOS_EMBED_TMP_DIR", str(tmp_path / "media-tmp"))
    ce1_http_server = ce1_server.run_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=ce1_http_server.serve_forever, daemon=True)
    thread.start()
    host, port = ce1_http_server.server_address

    try:
        response = requests.post(
            f"http://{host}:{port}/v1/embeddings",
            json={
                "input": "data:video/mp4;base64,not-valid-base64%%",
                "request_type": "query",
                "encoding_format": "float",
                "model": MODEL_NAME,
            },
            timeout=5,
        )
    finally:
        ce1_http_server.shutdown()
        ce1_http_server.server_close()
        thread.join(timeout=5)

    assert response.status_code == 422
    body = response.json()
    assert "data" not in body
    assert body["error"]["type"] == "invalid_request_error"


def test_ready_endpoint_reports_pytorch_config_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.delenv("COSMOS_EMBED_MODEL_PATH", raising=False)
    monkeypatch.delenv("COSMOS_EMBED_WEIGHTS_DIR", raising=False)
    server = ce1_server.run_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        response = requests.get(
            f"http://{host}:{port}/v1/health/ready",
            timeout=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "service_unavailable"


def test_liveness_ignores_invalid_backend_config_and_metadata_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.delenv("COSMOS_EMBED_MODEL_PATH", raising=False)
    monkeypatch.delenv("COSMOS_EMBED_WEIGHTS_DIR", raising=False)
    server = ce1_server.run_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        live = requests.get(f"http://{host}:{port}/v1/health/live", timeout=5)
        metadata = requests.get(f"http://{host}:{port}/v1/metadata", timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert live.status_code == 200
    assert metadata.status_code == 503


def test_ready_endpoint_reports_hf_auth_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSMOS_EMBED_BACKEND", "pytorch")
    monkeypatch.setenv("COSMOS_EMBED_ALLOW_HF_DOWNLOAD", "true")
    monkeypatch.delenv("COSMOS_EMBED_MODEL_PATH", raising=False)
    monkeypatch.delenv("COSMOS_EMBED_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        backends,
        "_huggingface_cache_roots",
        lambda: [tmp_path / "empty-hf-hub"],
    )
    server = ce1_server.run_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        response = requests.get(
            f"http://{host}:{port}/v1/health/ready",
            timeout=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    body = response.json()
    assert response.status_code == 503
    assert body["error"]["type"] == "service_unavailable"
    assert "requires HF_TOKEN" in body["error"]["detail"]


def test_normalize_embedding_rejects_zero_vector() -> None:
    with pytest.raises(backends.EmbeddingBackendError, match="zero or invalid"):
        backends.normalize_embedding([0.0] * EMBEDDING_DIMENSION)


def test_ce1_video_frame_count_defaults_when_processor_leaves_it_unset() -> None:
    processor = types.SimpleNamespace(num_video_frames=None)

    assert backends._target_num_video_frames(processor) == 8


def install_fake_ce1_runtime(
    monkeypatch: pytest.MonkeyPatch,
    projected_embedding: list[float],
    *,
    video_projected_embedding: list[float] | None = None,
    capture: dict[str, Any] | None = None,
    target_num_frames: int = 8,
    cuda_available: bool = True,
    video_decoder: Callable[[Path], Any] | None = None,
) -> None:
    capture = capture if capture is not None else {}
    # These tests exercise inference contracts with fake model objects. The
    # real pre-load integrity gate has separate positive and negative tests.
    monkeypatch.setattr(
        backends, "_verified_model_reference", lambda config, reference: reference,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: True,
    )
    fake_torch.from_numpy = lambda value: _FakeTensor(value)
    fake_torch.cat = lambda tensors, dim=0: _FakeTensor(
        _concatenate_fake_tensors(tensors, dim=dim)
    )
    fake_torch.no_grad = lambda: _NoGrad()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.57.6"
    fake_transformers.AutoModel = _FakeAutoModel(
        text_projection=projected_embedding,
        video_projection=video_projected_embedding or projected_embedding,
        capture=capture,
    )
    fake_transformers.AutoProcessor = _FakeAutoProcessor(
        capture=capture,
        target_num_frames=target_num_frames,
    )

    def create_gpu_media_processor(
        _backend: Any,
        *,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> _FakeGpuMediaProcessor:
        assert torch is fake_torch
        assert device.startswith("cuda")
        capture["gpu_processor_init_count"] = (
            capture.get("gpu_processor_init_count", 0) + 1
        )
        capture["gpu_device"] = device
        capture["gpu_dtype"] = dtype
        return _FakeGpuMediaProcessor(capture=capture, video_decoder=video_decoder)

    monkeypatch.setattr(
        backends.PyTorchCosmosEmbedBackend,
        "_create_gpu_media_processor",
        create_gpu_media_processor,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "PyNvVideoCodec", types.ModuleType("PyNvVideoCodec")
    )
    monkeypatch.setitem(sys.modules, "av", None)
    monkeypatch.setitem(sys.modules, "timm", types.ModuleType("timm"))
    monkeypatch.setitem(sys.modules, "einops", types.ModuleType("einops"))


class _FakeGpuMediaProcessor:
    """GPU interface double; never imports or initializes a native decoder."""

    def __init__(
        self,
        *,
        capture: dict[str, Any],
        video_decoder: Callable[[Path], Any] | None,
    ) -> None:
        self._capture = capture
        self._video_decoder = video_decoder

    def decode_video(self, path: Path) -> _FakeTensor:
        assert path.is_file()
        tensor = (
            self._video_decoder(path)
            if self._video_decoder is not None
            else _fake_video_tensor(count=8)
        )
        self._capture.setdefault("gpu_video_shapes", []).append(tensor._values.shape)
        return tensor

    def decode_frames(self, payloads: list[bytes]) -> _FakeTensor:
        assert len(payloads) == 8
        self._capture.setdefault("gpu_frame_payloads", []).append(list(payloads))
        return _fake_video_tensor(count=8)


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False


def _directory_entries(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.iterdir())


def _fake_video_tensor(count: int) -> _FakeTensor:
    import numpy as np

    frames = [
        np.full((2, 2, 3), fill_value=index, dtype=np.uint8) for index in range(count)
    ]
    return _FakeTensor(
        np.transpose(np.expand_dims(np.stack(frames), 0), (0, 1, 4, 2, 3))
    )


def _video_base64_data_uri(payload: bytes) -> str:
    encoded_payload = base64.b64encode(payload).decode("ascii")
    return f"data:video/mp4;base64,{encoded_payload}"


def _png_frame_bytes(fill_value: int) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (2, 2), color=(fill_value, fill_value, fill_value))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _video_frames_base64_data_uri(fill_value: int) -> str:
    payloads = [
        base64.b64encode(_png_frame_bytes(fill_value + index)).decode("ascii")
        for index in range(8)
    ]
    return f"data:video_frames/png;base64,{{{','.join(payloads)}}}"


def _video_frames_presigned_data_uri(urls: list[str]) -> str:
    return f"data:video_frames/png;presigned_url,{{{','.join(urls)}}}"


class _FakeAutoModel:
    def __init__(
        self,
        *,
        text_projection: list[float],
        video_projection: list[float],
        capture: dict[str, Any],
    ) -> None:
        self._text_projection = text_projection
        self._video_projection = video_projection
        self._capture = capture

    def from_pretrained(self, *args: Any, **kwargs: Any) -> "_FakeModel":
        self._capture["model_load_count"] = self._capture.get("model_load_count", 0) + 1
        self._capture.setdefault("from_pretrained_calls", []).append(
            {"kind": "model", "args": args, "kwargs": kwargs}
        )
        return _FakeModel(
            text_projection=self._text_projection,
            video_projection=self._video_projection,
            capture=self._capture,
        )


class _FakeAutoProcessor:
    def __init__(
        self,
        *,
        capture: dict[str, Any],
        target_num_frames: int,
    ) -> None:
        self._capture = capture
        self._target_num_frames = target_num_frames

    def from_pretrained(self, *args: Any, **kwargs: Any) -> "_FakeProcessor":
        self._capture["processor_load_count"] = (
            self._capture.get("processor_load_count", 0) + 1
        )
        self._capture.setdefault("from_pretrained_calls", []).append(
            {"kind": "processor", "args": args, "kwargs": kwargs}
        )
        return _FakeProcessor(
            capture=self._capture,
            target_num_frames=self._target_num_frames,
        )


class _FakeProcessor:
    def __init__(
        self,
        *,
        capture: dict[str, Any],
        target_num_frames: int,
    ) -> None:
        self._capture = capture
        self.num_video_frames = target_num_frames

    def __call__(
        self,
        *,
        text: list[str] | None = None,
        videos: Any | None = None,
        return_tensors: str,
    ) -> dict[str, Any]:
        assert return_tensors == "pt"
        if text is not None:
            assert text
            self._capture.setdefault("text_batches", []).append(list(text))
            return {"input_ids": _FakeTensor([1] * len(text))}
        assert videos is not None
        self._capture["processed_video_shape"] = videos.shape
        self._capture.setdefault("processed_video_shapes", []).append(videos.shape)
        return {"videos": _FakeTensor(videos)}


class _FakeModel:
    def __init__(
        self,
        *,
        text_projection: list[float],
        video_projection: list[float],
        capture: dict[str, Any],
    ) -> None:
        self._text_projection = text_projection
        self._video_projection = video_projection
        self._capture = capture
        self.eval_called = False

    def to(self, *_args: Any, **_kwargs: Any) -> "_FakeModel":
        return self

    def eval(self) -> None:
        self.eval_called = True

    def get_text_embeddings(self, **batch: Any) -> Any:
        self._capture["text_model_call_count"] = (
            self._capture.get("text_model_call_count", 0) + 1
        )
        batch_size = len(batch["input_ids"]._values)
        return types.SimpleNamespace(
            text_proj=[self._text_projection for _ in range(batch_size)]
        )

    def get_video_embeddings(self, *, videos: Any) -> Any:
        self._capture["video_model_called"] = True
        self._capture["video_model_call_count"] = (
            self._capture.get("video_model_call_count", 0) + 1
        )
        self._capture["video_model_input"] = videos
        batch_size = videos._values.shape[0]
        self._capture["video_model_batch_size"] = batch_size
        return types.SimpleNamespace(
            visual_proj=[self._video_projection for _ in range(batch_size)]
        )


class _FakeTensor:
    def __init__(self, values: Any) -> None:
        self._values = values

    def to(self, *_args: Any, **_kwargs: Any) -> "_FakeTensor":
        return self


class _CapturingTensor:
    def __init__(self, *, is_floating: bool) -> None:
        self._is_floating = is_floating
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def is_floating_point(self) -> bool:
        return self._is_floating

    def to(self, *args: Any, **kwargs: Any) -> "_CapturingTensor":
        self.calls.append((args, kwargs))
        return self


def _concatenate_fake_tensors(tensors: list[_FakeTensor], *, dim: int) -> Any:
    import numpy as np

    return np.concatenate([tensor._values for tensor in tensors], axis=dim)
