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
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import jwt
import requests


def get_api_key() -> Optional[str]:
    """Gets API key."""
    return os.environ.get("HTTP_AUTH_API_KEY", None)


@lru_cache(maxsize=1)
def request_auth_token(
    api_key: str,
    token_endpoint: str = "https://maglev-prod-sjc4.nvda.ai/authn/v1/token",
) -> str:
    """Gets auth token."""
    response = requests.get(
        token_endpoint, headers={"Authorization": f"ApiKey {api_key}"}
    )
    token = response.json()["token"]
    return token


def is_token_expired(token):
    try:
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        exp_timestamp = decoded_token["exp"]
        exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        return datetime.now(timezone.utc) >= exp_datetime
    except jwt.ExpiredSignatureError:
        return True
    except Exception as e:
        print(f"Error decoding token: {e}")
        return True


def get_auth_token(
    api_key: str,
) -> str:
    token_endpoint = os.environ.get(
        "MAGLEV_TOKEN_ENDPOINT", "https://maglev-prod-sjc4.nvda.ai/authn/v1/token"
    )
    token = request_auth_token(api_key, token_endpoint)
    if is_token_expired(token):
        # Clear the cache and fetch a new token
        request_auth_token.cache_clear()
        token = request_auth_token(api_key, token_endpoint)
    return token
