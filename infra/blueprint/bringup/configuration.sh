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

## Cosmos Video Search (CVS)

echo "=============================================="
echo "  CDS Deployment Configuration Validation"
echo "=============================================="
echo ""
echo "You must create secrets for nvcr.io and docker in order to pull the docker images."
echo "Get your NGC API key here: https://org.ngc.nvidia.com/setup/api-key"
echo "Get your docker hub personal access token here: https://app.docker.com/settings/personal-access-tokens"
echo ""
echo "Checking environment variables..."
echo ""

function env_var_check() {
  local var_name="$1"
  local show_mode="$2"
  local optional="$3"

  if [ -z "${!var_name}" ]; then
    if [ "$optional" = "optional" ]; then
      echo "Warning: $var_name is not set (optional)"
      return 0
    else
      echo "Error: $var_name is not set."
      exit 1
    fi
  fi

  if [ "$show_mode" = "show" ]; then
    echo "$var_name=${!var_name}"
  else
    # Show first few characters for security
    local value="${!var_name}"
    echo "$var_name=${value:0:10}...<masked>"
  fi
}

# Check required environment variables:
echo "Checking required credentials..."
env_var_check NGC_API_KEY
env_var_check DOCKER_PAT
env_var_check AWS_ACCESS_KEY_ID
env_var_check AWS_SECRET_ACCESS_KEY

# AWS_SESSION_TOKEN is optional - only required for temporary credentials
echo ""
echo "Checking AWS credential type..."
if [[ "$AWS_ACCESS_KEY_ID" == ASIA* ]]; then
  echo "Detected temporary AWS credentials (ASIA prefix)"
  env_var_check AWS_SESSION_TOKEN
  echo "Using temporary session credentials"
elif [[ "$AWS_ACCESS_KEY_ID" == AKIA* ]]; then
  echo "Detected permanent AWS credentials (AKIA prefix)"
  if [ -n "$AWS_SESSION_TOKEN" ]; then
    echo "Warning: AWS_SESSION_TOKEN is set but appears to be permanent credentials"
    echo "This is unusual. If using permanent credentials, AWS_SESSION_TOKEN should be empty."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  else
    echo "Using permanent AWS credentials (AWS_SESSION_TOKEN not required)"
  fi
else
  echo "Warning: Cannot determine AWS credential type from AWS_ACCESS_KEY_ID"
  echo "Expected AKIA* (permanent) or ASIA* (temporary)"
  env_var_check AWS_SESSION_TOKEN "" optional
fi

echo ""
echo "Checking deployment configuration..."
# If you want to print values, add "show" as the second argument:
env_var_check DOCKER_USER show
env_var_check CLUSTER_NAME show
env_var_check AWS_REGION show
env_var_check S3_BUCKET_NAME show

echo ""
echo "Validating deployment environment..."

# Check for existing AWS config directory
if [ ! -d "$HOME/.aws/" ]; then
  echo "No .aws directory found in \$HOME (good - prevents credential conflicts)";
else
  echo "Error: $HOME/.aws config folder exists"
  echo "  This can cause credential conflicts with environment variables."
  echo "  Please DELETE or rename $HOME/.aws before proceeding."
  echo ""
  echo "  You can backup and remove it with:"
  echo "    mv ~/.aws ~/.aws.backup"
  exit 1
fi

# Check if the length of CLUSTER_NAME is greater than 20
if [ ${#CLUSTER_NAME} -gt 20 ]; then
  echo "Error: CLUSTER_NAME length cannot exceed 20 characters."
  echo "  Current: '$CLUSTER_NAME' (${#CLUSTER_NAME} characters)"
  echo "  Please use a shorter name."
  exit 1
else
  echo "CLUSTER_NAME length OK (${#CLUSTER_NAME} characters)"
fi

# Validate cluster name format
if [[ ! "$CLUSTER_NAME" =~ ^[a-zA-Z0-9-]+$ ]]; then
  echo "Error: CLUSTER_NAME can only contain letters, numbers, and hyphens"
  echo "  Current: '$CLUSTER_NAME'"
  exit 1
else
  echo "CLUSTER_NAME format OK"
fi

# Validate S3 bucket name format
if [[ ! "$S3_BUCKET_NAME" =~ ^[a-z0-9][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "Error: S3_BUCKET_NAME must:"
  echo "  - Start and end with lowercase letter or number"
  echo "  - Contain only lowercase letters, numbers, and hyphens"
  echo "  Current: '$S3_BUCKET_NAME'"
  exit 1
else
  echo "S3_BUCKET_NAME format OK"
fi

echo ""
echo "=============================================="
echo "  Configuration validation complete!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Cluster: $CLUSTER_NAME"
echo "  Region: $AWS_REGION"
echo "  S3 Bucket: $S3_BUCKET_NAME"
echo "  Credential Type: $(if [[ "$AWS_ACCESS_KEY_ID" == AKIA* ]]; then echo "Permanent"; else echo "Temporary"; fi)"
echo ""
echo "Ready to proceed with deployment."
echo ""
