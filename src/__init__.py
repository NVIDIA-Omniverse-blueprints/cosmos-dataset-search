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

"""
Cosmos Video Dataset Search - Main package.

This is the root package for all CVDS services and utilities.

Package Structure:
- visual_search/     Main visual search service  
- triton/           Triton model repository and inference
- models/           Shared model utilities and configurations
- haystack/         Haystack integration components  
- wrappers/         Service wrappers
"""

__version__ = "0.6.0" 