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
import base64
import json
import os
import sys

import requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    p.add_argument("--collection-id", required=True)
    p.add_argument("--video", required=True, help="Path to a small video file (<= ~50 MiB base64)")
    p.add_argument("--mime", default="video/mp4")
    args = p.parse_args()

    with open(args.video, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"{args.base_url}/v1/collections/{args.collection_id}/documents"
    payload = [
        {
            "content": b64,            # raw base64 (no data:... prefix)
            "mime_type": args.mime,    # use "video/mp4" (or your actual mime)
            # "metadata": {"source": "demo"}  # optional
        }
    ]

    r = requests.post(url, headers={"Content-Type": "application/json"},
                      data=json.dumps(payload), timeout=600)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()
