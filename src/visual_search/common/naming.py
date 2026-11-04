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

"""Safe collection / table name helpers for vector backends."""

from __future__ import annotations

import re


def create_safe_name(name: str) -> str:
    """
    Create a backend-safe collection or table name.

    Replaces non-alphanumeric characters with underscores and prepends ``a``
    if the name would otherwise start with a digit.
    """
    if not name:
        raise ValueError("Input name cannot be empty.")
    safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    if not re.match(r"^[a-zA-Z_]", safe_name):
        safe_name = f"a{safe_name}"
    if len(safe_name) > 255:
        safe_name = safe_name[:255]
    return safe_name
