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

import os

from cryptography.fernet import Fernet


def generate_secret_key():
    # Convert to base64 for easier storage/transmission
    encryption_key = Fernet.generate_key().decode()
    return encryption_key


def set_env_variable(key_name="SECRET_ENCRYPTION_KEY"):
    secret_key = generate_secret_key()

    # Set environment variable for current process
    os.environ[key_name] = secret_key

    # Print instructions for permanent setup
    print(f"export {key_name}='{secret_key}'")
    return secret_key


if __name__ == "__main__":
    set_env_variable()
