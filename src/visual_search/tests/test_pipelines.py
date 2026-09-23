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

"""Unit tests for pipelines."""

from typing import Dict, List, Tuple

from haystack import Pipeline, component

from src.visual_search.common.pipelines import extract_subgraph


@component
class ComponentA:
    @component.output_types(output=List[float])
    def run(self, input: List[float]) -> Dict[str, List[float]]:
        return {"output": input * 2}


@component
class ComponentB:
    @component.output_types(tuple=Tuple[float, ...])
    def run(self, lst: List[float]) -> Dict[str, Tuple[float, ...]]:
        return {"tuple": tuple(lst)}


@component
class ComponentC:
    @component.output_types(a=int, b=float)
    def run(self, lst: List[float]) -> Dict[str, int | float]:
        return {"a": len(lst), "b": sum(lst)}


@component
class ComponentD:
    @component.output_types(output=int)
    def run(self, input: int | float) -> Dict[str, int | float]:
        return {"input": 3 * input}


def test_extract_subgraph() -> None:
    """Test subgraph extraction."""

    p = Pipeline()
    p.add_component("component_a", ComponentA())
    p.add_component("component_b", ComponentB())
    p.add_component("component_c", ComponentC())
    p.add_component("component_d", ComponentD())
    p.add_component("component_d1", ComponentD())

    p.connect("component_a.output", "component_b.lst")
    p.connect("component_a.output", "component_c.lst")
    p.connect("component_c.a", "component_d.input")
    p.connect("component_c.b", "component_d1.input")

    old_output = p.run({"component_a": {"input": [1.0, 2.0]}})
    assert old_output == {
        "component_b": {"tuple": (1.0, 2.0, 1.0, 2.0)},
        "component_d": {"input": 12},
        "component_d1": {"input": 18.0},
    }
    new_p = extract_subgraph(p, ["component_c"])
    new_output = new_p.run({"component_a": {"input": [1.0, 2.0]}})
    assert new_output == {"component_c": {"a": 4, "b": 6.0}}
