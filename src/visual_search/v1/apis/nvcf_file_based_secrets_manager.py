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

import json
import os
from typing import Dict

from src.visual_search.common.exceptions import SecretsNotFoundError

class NVCFFileBasedSecretsManager:
    def __init__(self, nvfc_secrets_path: str = None):
        # Use SECRETS_FILE_PATH as the default NVCF Location
        self.nvfc_secrets_path = os.environ.get("NGC_SECRETS_FILE_PATH", "/var/secrets/secrets.json")

    def acquire_key(self) -> Dict[str, str] | None:
        # Check environment variables first
        access_key = os.environ.get("CVDS_S3_ACCESS_KEY")
        secret_key = os.environ.get("CVDS_S3_SECRET_KEY")
        aws_region = os.environ.get("AWS_REGION")
        if access_key and secret_key:
            return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key, "aws_region": aws_region}

        # Read from the NVCF Location
        try:
            with open(self.nvfc_secrets_path) as f:
                secrets = json.load(f)
                access_key = secrets.get("CVDS_S3_ACCESS_KEY")
                secret_key = secrets.get("CVDS_S3_SECRET_KEY")
                aws_region = secrets.get("AWS_REGION")
                if access_key and secret_key:
                    return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key, "aws_region": aws_region}
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return None

    def get_secrets(self) -> Dict[str, str]:
        secret_value = self.acquire_key()
        if secret_value is not None:
            return secret_value
        else:
            raise SecretsNotFoundError("NVCF Secrets not found!")
