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

"""Serializer utility for haystack components."""


import inspect
from typing import Any, Dict, Final

from haystack.core.serialization import default_from_dict, default_to_dict

TYPE_KEY: Final = "type"
INIT_PARAMETERS_KEY: Final = "init_parameters"


class SerializerMixin:
    """Serializer mixin that infers serialization parameters from `__init__` signature."""

    def to_dict(self) -> Dict[str, Any]:
        """Default `to_dict` method."""
        init_method = self.__class__.__init__
        signature = inspect.signature(init_method)
        param_names = [
            param.name
            for param in signature.parameters.values()
            if param.name not in ("self", "args", "kwargs")
        ]
        param_dict = dict()
        for param in param_names:
            if not hasattr(self, param):
                raise AttributeError(
                    f"`{param}` attribute is missing from `self` in `{self.__class__.__name__}`. "
                    "Please assign to `self` for serializer to work."
                )
            param_dict[param] = getattr(self, param)
        return default_to_dict(self, **param_dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Default `from_dict` method."""
        return default_from_dict(cls, data)
