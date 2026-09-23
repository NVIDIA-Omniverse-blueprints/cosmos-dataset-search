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

"""Runtime dependency checks for the CVDS-owned Cosmos-Embed OSS service."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

REQUIRED_PYTORCH_MODULES = (
    "torch",
    "transformers",
    "numpy",
    "PIL",
    "timm",
    "einops",
)


class RuntimeDependencyError(RuntimeError):
    """Raised when the PyTorch CE1 runtime is not importable."""


def import_pytorch_runtime_modules(
    *,
    module_names: Iterable[str] = REQUIRED_PYTORCH_MODULES,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Import required CE1 PyTorch runtime modules or raise a clear error."""

    modules: dict[str, Any] = {}
    missing: list[str] = []
    for module_name in module_names:
        try:
            modules[module_name] = import_module(module_name)
        except Exception as error:  # pragma: no cover - exact import failures vary.
            missing.append(f"{module_name} ({type(error).__name__}: {error})")

    if missing:
        required = ", ".join(module_names)
        missing_detail = "; ".join(missing)
        raise RuntimeDependencyError(
            "PyTorch CE1 backend requires runtime dependencies "
            f"{required}; missing or unusable: {missing_detail}"
        )
    return modules


def pytorch_runtime_status(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Return import/version status for the PyTorch CE1 runtime image."""

    modules = import_pytorch_runtime_modules(import_module=import_module)
    versions = {
        module_name: str(getattr(module, "__version__", "present"))
        for module_name, module in modules.items()
    }
    torch = modules["torch"]
    cuda = getattr(torch, "cuda", None)
    cuda_available = False
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        cuda_available = bool(cuda.is_available())
    return {
        "status": "ok",
        "required_modules": versions,
        "cuda_available": cuda_available,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by Docker build and runtime smoke checks."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON status")
    args = parser.parse_args(argv)

    try:
        status = pytorch_runtime_status()
    except RuntimeDependencyError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        _print_human_status(status)
    return 0


def _print_human_status(status: Mapping[str, Any]) -> None:
    print(f"status={status['status']}")
    modules = status.get("required_modules", {})
    if isinstance(modules, Mapping):
        for module_name in sorted(modules):
            print(f"{module_name}={modules[module_name]}")
    print(f"cuda_available={status.get('cuda_available', False)}")


if __name__ == "__main__":
    raise SystemExit(main())
