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

"""Unit tests for filter utilities of milvus document store in Haystack."""

from src.haystack.components.milvus.filter_utils import (
    AndOperation,
    EqOperation,
    GteOperation,
    GtOperation,
    InOperation,
    LteOperation,
    LtOperation,
    NeOperation,
    NinOperation,
    NotOperation,
    OrOperation,
)


def test_filters() -> None:
    assert (
        InOperation("dummy_field", ["1.0"]).convert_to_milvus()
        == '(dummy_field in ["1.0"])'
    )
    assert (
        NinOperation("dummy_field", ["1.0"]).convert_to_milvus()
        == '(dummy_field not in ["1.0"])'
    )
    assert (
        AndOperation(
            [EqOperation("dummy_field1", "1.0"), EqOperation("dummy_field2", "2.0")]
        ).convert_to_milvus()
        == '((dummy_field1 == "1.0") and (dummy_field2 == "2.0"))'
    )
    assert (
        AndOperation(
            [
                GteOperation("dummy_field1", "1.0"),
                LteOperation("dummy_field2", "2.0"),
                LtOperation("dummy_field3", "3.0"),
            ]
        ).convert_to_milvus()
        == '((dummy_field1 >= "1.0") and (dummy_field2 <= "2.0") and (dummy_field3 < "3.0"))'
    )
    assert (
        NotOperation(
            [GtOperation("dummy_field1", "1.0"), NeOperation("dummy_field2", "2.0")]
        ).convert_to_milvus()
        == '((dummy_field1 <= "1.0") or (dummy_field2 == "2.0"))'
    )
    assert (
        OrOperation(
            [GtOperation("dummy_field1", "1.0"), NeOperation("dummy_field2", "2.0")]
        ).convert_to_milvus()
        == '((dummy_field1 > "1.0") or (dummy_field2 != "2.0"))'
    )
