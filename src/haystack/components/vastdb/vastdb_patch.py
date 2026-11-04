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

"""Make VastDB SDK ``select()`` work on tables that contain a vector column."""

from __future__ import annotations

import logging

import pyarrow as pa

logger = logging.getLogger(__name__)


def is_vector_field(field: pa.Field) -> bool:
    t = str(field.type)
    if "fixed_size_list" in t:
        return True
    return "list<" in t and "float" in t


def _install() -> None:
    try:
        import vastdb._internal as _internal
    except ImportError:
        logger.debug("vastdb not installed; skipping select() patch")
        return

    if getattr(_internal.build_query_data_request, "_cds_vastdb_patched", False):
        return

    _original = _internal.build_query_data_request

    def _patched(schema=pa.schema([]), predicate=None, field_names=None):
        supported = [f for f in schema if not is_vector_field(f)]
        unsupported = {f.name for f in schema if is_vector_field(f)}
        if unsupported:
            logger.debug(
                "Excluding vector columns from VastDB SDK query: %s", unsupported
            )
        names = field_names
        if names is not None:
            names = [n for n in names if n not in unsupported]
        return _original(pa.schema(supported), predicate, names)

    _patched._cds_vastdb_patched = True  # type: ignore[attr-defined]
    _internal.build_query_data_request = _patched


_install()
