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

"""Uvicorn Python wrapper."""

import sys

import uvicorn

if __name__ == "__main__":
    print("Wrapper received arguments:", sys.argv)
    file = sys.argv.pop(1)
    function = sys.argv.pop(1)
    stem, extension = file.split(".")
    assert extension == "py", f"File should be a Python (.py) file! Got {extension}."
    module_import_path = stem.replace("/", ".")
    app_name = module_import_path + ":" + function
    print(f"Starting uvicorn app `{app_name}`")
    sys.argv.insert(1, app_name)
    uvicorn.main() 