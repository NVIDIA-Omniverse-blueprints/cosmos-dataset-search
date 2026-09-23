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

"""Unit tests for Haystack joiners."""

import pytest

from src.haystack.components.joiners import Concatenate, Flatten


@pytest.mark.parametrize(
    "type_alias", ["typing.List[float]", "float", "haystack.Document"]
)
def test_serialization_concatenate(type_alias: str) -> None:
    component = Concatenate(type_alias=type_alias)
    serialized_dict = component.to_dict()
    new_component = Concatenate.from_dict(serialized_dict)
    assert new_component.to_dict() == serialized_dict


@pytest.mark.parametrize("type_alias", ["typing.List[float]", "typing.Sequence[float]"])
def test_serialization_flatten(type_alias: str) -> None:
    component = Flatten(type_alias=type_alias)
    serialized_dict = component.to_dict()
    new_component = Flatten.from_dict(serialized_dict)
    assert new_component.to_dict() == serialized_dict
