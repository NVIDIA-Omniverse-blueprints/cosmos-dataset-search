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

"""Vast S3 (S3-compatible) client helpers for CDS media storage."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_vast_s3_client(
    *,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = None,
    ssl_verify: Optional[bool] = None,
) -> Any:
    """
    Build a boto3 S3 client pointed at Vast S3 (path-style, SigV4).

    Environment defaults (used when kwargs are omitted):
      VAST_S3_ENDPOINT or AWS_ENDPOINT_URL
      VAST_S3_ACCESS_KEY / VAST_ACCESS_KEY / AWS_ACCESS_KEY_ID
      VAST_S3_SECRET_KEY / VAST_SECRET_KEY / AWS_SECRET_ACCESS_KEY
      VAST_S3_REGION / AWS_DEFAULT_REGION
      VAST_S3_SSL_VERIFY (true/false)
    """
    import boto3
    from botocore.client import Config

    endpoint = endpoint or os.getenv("VAST_S3_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL")
    if not endpoint:
        raise ValueError(
            "Vast S3 endpoint is required (VAST_S3_ENDPOINT or AWS_ENDPOINT_URL)"
        )

    access_key = (
        access_key
        or os.getenv("VAST_S3_ACCESS_KEY")
        or os.getenv("VAST_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
    )
    secret_key = (
        secret_key
        or os.getenv("VAST_S3_SECRET_KEY")
        or os.getenv("VAST_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    if not access_key or not secret_key:
        raise ValueError("Vast S3 access/secret keys are required")

    region = region or os.getenv("VAST_S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

    if ssl_verify is None:
        ssl_verify = os.getenv("VAST_S3_SSL_VERIFY", "true").lower() in (
            "1",
            "true",
            "yes",
        )

    logger.info("Connecting to Vast S3 at %s (path-style)", endpoint)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        verify=ssl_verify,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    """Create ``bucket`` on Vast S3 if it does not already exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        logger.info("Creating Vast S3 bucket %s", bucket)
        client.create_bucket(Bucket=bucket)
