# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small trusted fixtures exercise the same verification used for real weights."""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from src.cosmos_embed_oss import backends, model_integrity
from src.cosmos_embed_oss.tests.test_backends import install_fake_ce1_runtime


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    files = {
        "config.json": b'{"auto_map":{"AutoModel":"modeling_embed1.CosmosEmbed1"}}',
        "modeling_embed1.py": b"# trusted model code\n",
        "preprocessing_embed1.py": b"# trusted processor code\n",
        "model.safetensors.index.json": b"{}",
        "model.safetensors": b"test-only model weights",
    }
    for name, value in files.items():
        (root / name).write_bytes(value)
    manifest = tmp_path / "application-owned-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": model_integrity.MODEL_ID,
                "revision": model_integrity.MODEL_REVISION,
                "sha256": {
                    name: hashlib.sha256(value).hexdigest()
                    for name, value in files.items()
                },
            }
        )
    )
    monkeypatch.setattr(model_integrity, "MANIFEST_PATH", manifest)
    return root


def test_valid_snapshot_and_cached_blob_symlink(snapshot: Path, tmp_path: Path) -> None:
    assert model_integrity.verify_model_snapshot(snapshot) == snapshot
    weights = snapshot / "model.safetensors"
    blob = tmp_path / "blob"
    weights.rename(blob)
    weights.symlink_to(blob)
    assert model_integrity.verify_model_snapshot(snapshot) == snapshot


@pytest.mark.parametrize(
    "name",
    [
        "modeling_embed1.py",
        "preprocessing_embed1.py",
        "config.json",
        "model.safetensors.index.json",
        "model.safetensors",
    ],
)
def test_tampered_files_fail_closed(snapshot: Path, name: str) -> None:
    (snapshot / name).write_text("tampered")
    with pytest.raises(model_integrity.ModelIntegrityError, match="checksum mismatch"):
        model_integrity.verify_model_snapshot(snapshot)


def test_missing_file_fails_closed(snapshot: Path) -> None:
    (snapshot / "modeling_embed1.py").unlink()
    with pytest.raises(
        model_integrity.ModelIntegrityError, match="missing required file"
    ):
        model_integrity.verify_model_snapshot(snapshot)


@pytest.mark.parametrize(
    "name",
    [
        "injected.py",
        "tokenizer.json",
        "pytorch_model.bin",
        "modeling_embed1.pyc",
        "subdir/injected.py",
    ],
)
def test_extra_loadable_files_rejected(snapshot: Path, name: str) -> None:
    path = snapshot / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("untrusted")
    with pytest.raises(
        model_integrity.ModelIntegrityError, match="unapproved model file"
    ):
        model_integrity.verify_model_snapshot(snapshot)


def test_manifest_on_model_volume_is_not_trusted(snapshot: Path) -> None:
    (snapshot / "model_manifest.json").write_text('{"sha256":{}}')
    with pytest.raises(
        model_integrity.ModelIntegrityError, match="unapproved model file"
    ):
        model_integrity.verify_model_snapshot(snapshot)


def test_reject_directory_symlinks(snapshot: Path, tmp_path: Path) -> None:
    (snapshot / "external").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(model_integrity.ModelIntegrityError, match="directory link"):
        model_integrity.verify_model_snapshot(snapshot)


@pytest.mark.parametrize("revision", ["main", "", "a" * 40])
def test_online_revision_must_be_release_approved(revision: str) -> None:
    with pytest.raises(backends.EmbeddingBackendError, match="approved model"):
        backends.create_embedding_backend(
            backends.BackendConfig(allow_hf_download=True, hf_revision=revision)
        )


def test_alternate_model_repo_rejected() -> None:
    with pytest.raises(backends.EmbeddingBackendError, match="approved model"):
        backends.create_embedding_backend(
            backends.BackendConfig(allow_hf_download=True, hf_model_id="other/model")
        )


def test_online_load_pins_then_verifies_one_local_snapshot(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=download),
    )
    config = backends.BackendConfig(allow_hf_download=True)
    ref = backends._ResolvedPytorchModel(config.hf_model_id, "huggingface", False, True)
    resolved = backends._verified_model_reference(config, ref)
    assert calls == [
        {
            "repo_id": model_integrity.MODEL_ID,
            "revision": model_integrity.MODEL_REVISION,
            "local_files_only": False,
            "token": True,
        }
    ]
    assert resolved.reference == str(snapshot)
    assert backends._from_pretrained_kwargs(config, resolved) == {
        "trust_remote_code": True,
        "local_files_only": True,
    }


@pytest.mark.parametrize("tamper", [False, True])
def test_integrity_checked_before_either_transformers_loader(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch, tamper: bool
) -> None:
    real_verifier = backends._verified_model_reference
    capture = {}
    install_fake_ce1_runtime(monkeypatch, [1.0] + [0.0] * 255, capture=capture)
    monkeypatch.setattr(backends, "_verified_model_reference", real_verifier)
    backend = backends.create_embedding_backend(
        backends.BackendConfig(model_path=str(snapshot))
    )
    if tamper:
        (snapshot / "modeling_embed1.py").write_text(
            "raise Exception('must never execute')"
        )
        with pytest.raises(
            backends.EmbeddingBackendError, match="integrity verification failed"
        ):
            backend.ready()
        assert not capture.get("from_pretrained_calls")
    else:
        backend.ready()
        assert [item["kind"] for item in capture["from_pretrained_calls"]] == [
            "model",
            "processor",
        ]


def test_release_manifest_covers_model_code_config_tokenizer_and_weights() -> None:
    manifest = json.loads(model_integrity.MANIFEST_PATH.read_text())
    assert manifest["revision"] == model_integrity.MODEL_REVISION
    required = set(manifest["sha256"]) - set(manifest["optional_files"])
    assert {
        "configuration_embed1.py",
        "modeling_embed1.py",
        "modeling_qformer.py",
        "modeling_vit.py",
        "modeling_utils.py",
        "modeling_outputs.py",
        "preprocessing_embed1.py",
        "config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "vocab.txt",
    } <= required
    assert len([name for name in required if name.endswith(".safetensors")]) == 5
