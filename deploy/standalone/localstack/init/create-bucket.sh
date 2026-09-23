#!/usr/bin/env bash
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

set -euo pipefail

# Ensure the S3 bucket used by Milvus exists in LocalStack.

BUCKET_NAME="${BUCKET_NAME:-cosmos-test-bucket}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"

aws_cmd() {
  if command -v awslocal >/dev/null 2>&1; then
    awslocal "$@"
  else
    aws --endpoint-url "${ENDPOINT_URL}" "$@"
  fi
}

echo "[localstack:init] ensuring bucket '${BUCKET_NAME}' in region '${REGION}'"

if aws_cmd s3api head-bucket --bucket "${BUCKET_NAME}" 2>/dev/null; then
  echo "[localstack:init] bucket '${BUCKET_NAME}' already exists"
else
  if [ "${REGION}" = "us-east-1" ]; then
    aws_cmd s3api create-bucket --bucket "${BUCKET_NAME}"
  else
    aws_cmd s3api create-bucket --bucket "${BUCKET_NAME}" --create-bucket-configuration LocationConstraint="${REGION}"
  fi
  echo "[localstack:init] created bucket '${BUCKET_NAME}'"
fi

echo "[localstack:init] bucket '${BUCKET_NAME}' is ready"
