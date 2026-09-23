#!/usr/bin/env python3
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

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import List

import boto3
import requests

# ------------------------------------------------------------------
# CONFIG – edit these three lines or override with CLI arguments
# ------------------------------------------------------------------
DEFAULT_COSMOS_EMBED_URI = "http://localhost:9000"
DEFAULT_BUCKET = "cosmos-test-bucket"
S3_ENDPOINT = "http://localhost:4566"  # LocalStack default
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
# ------------------------------------------------------------------


def upload_and_presign(
    s3_client,
    bucket: str,
    local_file: Path,
    expiry_s: int = 3 * 60 * 60,
    url_host: str = None,
) -> str:
    key = f"cosmos-tests/{local_file.stem}-{os.getpid()}{local_file.suffix}"
    s3_client.upload_file(str(local_file), bucket, key)
    url = s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_s,
    )
    # If user asked for a different host, rewrite the URL
    if url_host:
        from urllib.parse import urlparse, urlunparse

        parts = list(urlparse(url))
        # keep port if present
        if ":" in parts[1]:
            _, port = parts[1].split(":", 1)
            parts[1] = f"{url_host}:{port}"
        else:
            parts[1] = url_host
        url = urlunparse(parts)
    return url


def build_video_inputs(urls: List[str]) -> List[str]:
    """Wrap each presigned URL in the 'data:video/<fmt>;presigned_url,<URL>' envelope."""
    inputs = []
    for url in urls:
        inputs.append(f"data:video/mp4;presigned_url,{url}")
    return inputs


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Cosmos-Embed bulk_video presigned-URL test"
    )
    p.add_argument(
        "--cosmos-embed-uri",
        default=DEFAULT_COSMOS_EMBED_URI,
        help="Base URI of the CE1 OSS service (default: %(default)s)",
    )
    p.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="S3 bucket to upload to (default: %(default)s)",
    )
    p.add_argument(
        "--endpoint-url",
        default=S3_ENDPOINT,
        help="Custom S3 endpoint (LocalStack/MinIO). If unset," " boto3 uses real AWS",
    )
    # Optional: replace the hostname in the generated presigned URLs so
    # they are reachable from inside the CE1 OSS service container.
    p.add_argument(
        "--url-host",
        help="Override hostname part of presigned URLs "
        "(e.g. 'localstack' or 'host.docker.internal').",
    )
    p.add_argument(
        "video_files", nargs="+", help="Local video files (.mp4, .mov …) to upload (≥2)"
    )
    args = p.parse_args(argv)

    if len(args.video_files) < 2:
        sys.exit("Need at least two input files for a real bulk test.")

    # ---------- S3 SETUP ----------
    s3 = boto3.client("s3", endpoint_url=args.endpoint_url, region_name=AWS_REGION)

    # Ensure bucket exists (idempotent)
    try:
        s3.head_bucket(Bucket=args.bucket)
    except s3.exceptions.ClientError as exc:
        # Bucket is absent – create it with region-aware logic
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=args.bucket)
        else:
            s3.create_bucket(
                Bucket=args.bucket,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )

    # ---------- UPLOAD & PRESIGN ----------
    presigned_urls = []
    for vid in args.video_files:
        url = upload_and_presign(s3, args.bucket, Path(vid), url_host=args.url_host)
        presigned_urls.append(url)
        print(f"Uploaded {vid} -> {url.split('?')[0]}")

    video_inputs = build_video_inputs(presigned_urls)
    print("\nPrepared bulk_video payload:")
    for v in video_inputs:
        print("  ", v[:120] + ("…" if len(v) > 120 else ""))

    # ---------- CALL CE1 OSS service ----------
    payload = {
        "input": video_inputs,
        "request_type": "bulk_video",
        "encoding_format": "float",
        "model": "nvidia/cosmos-embed1",
    }
    url = f"{args.cosmos_embed_uri.rstrip('/')}/v1/embeddings"
    print(f"\nPOST {url}")
    resp = requests.post(url, json=payload, timeout=600)
    print("Status:", resp.status_code)

    try:
        data = resp.json()
    except ValueError:
        print("Non-JSON response:\n", resp.text)
        sys.exit(1)

    if resp.ok:
        print(f"Success – received {len(data['data'])} embeddings")
    else:
        print("Error response:")
        print(json.dumps(data, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
