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

from typing import Any, Dict, List, Tuple

from src.visual_search.common.models import is_field_included


def test_is_field_included():
    test_cases: List[Tuple[Dict[str, Any], bool]] = [
        ({"field": "session_id", "operator": "in", "value": ["a", "b"]}, True),
        ({"field": "patch_id", "operator": "==", "value": "1"}, False),
        # allowed: session_id == ? and other = ?
        (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "session_id", "operator": "in", "value": ["a", "b"]},
                    {"field": "patch_id", "operator": "==", "value": "1"},
                ],
            },
            True,
        ),
        # not allowed: other_1 == ? and other_2 = ?
        (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "camera", "operator": "in", "value": ["a", "b"]},
                    {"field": "patch_id", "operator": "==", "value": "1"},
                ],
            },
            False,
        ),
        # not allowed: session_id == ? or other = ?
        (
            {
                "operator": "OR",
                "conditions": [
                    {"field": "session_id", "operator": "in", "value": ["a", "b"]},
                    {
                        "field": "camera_id",
                        "operator": "in",
                        "value": ["camera_front_wide_120fov"],
                    },
                ],
            },
            False,
        ),
        # allowed: session_id == ? or (session_id == ? and other = ?)
        (
            {
                "operator": "OR",
                "conditions": [
                    {"field": "session_id", "operator": "==", "value": "a"},
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "session_id", "operator": "==", "value": "b"},
                            {
                                "field": "camera_id",
                                "operator": "in",
                                "value": ["camera_front_wide_120fov"],
                            },
                        ],
                    },
                ],
            },
            True,
        ),
    ]
    for test_filter, result in test_cases:
        assert is_field_included(test_filter, "session_id") == result
