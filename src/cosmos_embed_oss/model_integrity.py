# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the release-approved CE1 snapshot before executing its custom code.

The trust anchor ships with application code, never with the mounted model.
Updating a model requires reviewing its revision and this manifest together.
Model storage and the HF module cache must remain operator-controlled: hashing
at startup cannot protect against a host administrator modifying files later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "nvidia/Cosmos-Embed1-224p"
MODEL_REVISION = "787e0b996f5260a71ad474a283c90539a2e12986"
MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")


class ModelIntegrityError(RuntimeError):
    """The requested snapshot is not the release-approved model."""


def validate_model_source(model_id: str, revision: str) -> None:
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        raise ModelIntegrityError(
            f"CE1 requires the approved model {MODEL_ID}@{MODEL_REVISION}; "
            "review and update the application checksum manifest for a new revision"
        )


def verify_model_snapshot(model_dir: str | Path) -> Path:
    """Fail closed on missing, modified or additional loadable model files."""
    root = Path(model_dir).resolve()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_model_source(manifest["model_id"], manifest["revision"])
    files = manifest["sha256"]
    if not root.is_dir():
        raise ModelIntegrityError("CE1 snapshot directory is missing")

    # HF snapshots may use file symlinks to their content-addressed blob store.
    # Hash the target bytes, not the symlink path. Do not follow directory links.
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith((".cache/", ".git/")):
            continue
        if path.is_symlink() and path.is_dir():
            raise ModelIntegrityError(
                f"CE1 snapshot contains a directory link: {relative}"
            )
        if (
            path.suffix
            in {
                ".py",
                ".pyc",
                ".json",
                ".txt",
                ".safetensors",
                ".bin",
                ".pt",
                ".pth",
                ".so",
            }
            and relative not in files
        ):
            raise ModelIntegrityError(
                f"CE1 snapshot contains an unapproved model file: {relative}"
            )

    for relative, expected in files.items():
        path = root / relative
        # Optional upstream conversion/example code is checked when present,
        # but is not needed by an inference-only snapshot.
        if relative in manifest.get("optional_files", []) and not path.exists():
            continue
        if not path.is_file():
            raise ModelIntegrityError(
                f"CE1 snapshot is missing required file: {relative}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ModelIntegrityError(f"CE1 snapshot checksum mismatch: {relative}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    args = parser.parse_args()
    try:
        verify_model_snapshot(args.model_dir)
    except (ModelIntegrityError, OSError) as error:
        parser.exit(1, f"CE1 model verification failed: {error}\n")
    print(f"Verified {MODEL_ID}@{MODEL_REVISION}")


if __name__ == "__main__":
    main()
