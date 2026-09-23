#!/bin/bash
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

# Pass a local video file as the first argument.
video_b64=$(base64 < "${1:?Usage: test_cosmos_embed_service.sh VIDEO_PATH}" | tr -d '\n')

curl -X 'POST' \
  'http://0.0.0.0:9000/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  --data @- <<EOF
{
  "input": [
    "data:video/mp4;base64,${video_b64}"
  ],
  "encoding_format": "float",
  "model": "nvidia/cosmos-embed1",
  "request_type": "query"
}
EOF
