#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

## Cosmos Dataset Search (CDS)

set -e

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment for CDS CLI..."
    source "$VENV_PATH"
fi

### Are the environment variables set?
if [ -z "$CUSTOM_AWS_ACCESS_KEY_ID" ] || [ -z "$CUSTOM_AWS_REGION" ] || [ -z "$CUSTOM_AWS_SECRET_ACCESS_KEY" ] || [ -z "$CUSTOM_S3_BUCKET_NAME" ]; then
  echo "Please set the environment variables CUSTOM_AWS_ACCESS_KEY_ID, CUSTOM_AWS_REGION, CUSTOM_AWS_SECRET_ACCESS_KEY, CUSTOM_S3_BUCKET_NAME, CUSTOM_S3_PROFILE"
  exit 1
fi

PROFILE_NAME="${CUSTOM_S3_BUCKET_NAME}-profile"

AWS_CREDENTIALS_FILE="${HOME}/.aws/credentials"
AWS_CONFIG_FILE="${HOME}/.aws/config"
PROFILE_EXISTS=false

if grep -q "\[${PROFILE_NAME}\]" "$AWS_CREDENTIALS_FILE" 2>/dev/null && \
   grep -q "\[profile ${PROFILE_NAME}\]" "$AWS_CONFIG_FILE" 2>/dev/null; then
    PROFILE_EXISTS=true
fi

if [ "$PROFILE_EXISTS" = true ]; then
    echo "AWS profile '${PROFILE_NAME}' already exists in credentials and config. Skipping creation."
else
    echo "Creating AWS profile '${PROFILE_NAME}' with the storage secrets"
    aws configure set aws_access_key_id "${CUSTOM_AWS_ACCESS_KEY_ID}" --profile "${PROFILE_NAME}"
    aws configure set aws_secret_access_key "${CUSTOM_AWS_SECRET_ACCESS_KEY}" --profile "${PROFILE_NAME}"
    aws configure set region "${CUSTOM_AWS_REGION}" --profile "${PROFILE_NAME}"
    if [[ -n "${CUSTOM_ENDPOINT_URL:-}" ]]; then
        aws configure set endpoint_url "${CUSTOM_ENDPOINT_URL}" --profile "${PROFILE_NAME}"
    fi
fi

# Determine execution context (CI vs local)
if [[ -n "${CUSTOM_API_ENDPOINT:-}" ]]; then
  KUBECTL_CMD="${CUSTOM_KUBECTL_CMD:-oc}"
elif [[ -n "${GITLAB_CI:-}" ]]; then
  echo "Running in GitLab CI environment..."
  KUBECTL_CMD="kubectl"
else
  echo "Running in local environment..."
  KUBECTL_CMD="docker exec cds-deployment kubectl"
fi

# Get ingress hostname
if [[ -n "${CUSTOM_API_ENDPOINT:-}" ]]; then
  VS_API="$CUSTOM_API_ENDPOINT"
else
  VS_API=$($KUBECTL_CMD get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
fi
echo "Deployment URL: https://$VS_API/cosmos-dataset-search"

# Create Kubernetes secret with AWS credentials.
# The visual-search app reads secrets from the "default" namespace,
# so when running on OpenShift (CUSTOM_API_ENDPOINT set) we target "default".
SECRETS_NAME="${CUSTOM_S3_BUCKET_NAME}-secrets-videos"
if [[ -n "${CUSTOM_API_ENDPOINT:-}" ]] && [[ -z "${CUSTOM_SECRET_NAMESPACE:-}" ]]; then
  CUSTOM_SECRET_NAMESPACE="default"
fi
NS_FLAG=""
if [[ -n "${CUSTOM_SECRET_NAMESPACE:-}" ]]; then
  NS_FLAG="-n $CUSTOM_SECRET_NAMESPACE"
fi

# Check if secret already exists
if $KUBECTL_CMD get secret $SECRETS_NAME $NS_FLAG &>/dev/null; then
    echo "Kubernetes secret '$SECRETS_NAME' already exists. Deleting and recreating..."
    $KUBECTL_CMD delete secret $SECRETS_NAME $NS_FLAG
fi

echo "Creating Kubernetes secret with AWS credentials..."
ENDPOINT_FLAG=""
if [[ -n "${CUSTOM_ENDPOINT_URL:-}" ]]; then
    ENDPOINT_FLAG="--from-literal=endpoint_url=$CUSTOM_ENDPOINT_URL"
fi
$KUBECTL_CMD create secret generic $SECRETS_NAME $NS_FLAG \
  --from-literal=aws_access_key_id=$CUSTOM_AWS_ACCESS_KEY_ID \
  --from-literal=aws_secret_access_key=$CUSTOM_AWS_SECRET_ACCESS_KEY \
  --from-literal=aws_region=$CUSTOM_AWS_REGION \
  $ENDPOINT_FLAG

### Now we need to create a collection for the data.
### The tag file tells the UI how to download images and videos.
echo "{ \"tags\": {
  \"storage-template\": \"s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/{{filename}}\",
  \"storage-secrets\": \"$SECRETS_NAME\"
} }" > custom_video_bucket_meta.json
COLLECTION=$(cds collections create --pipeline cosmos_video_search_milvus --config-yaml custom_video_bucket_meta.json --name "video")
COLLECTION_ID=$(echo "$COLLECTION" | awk -F'"' '/"id":/{print $4}')
echo $COLLECTION_ID

### Now we need to ingest the data
### Use --limit <N> to process a subset of the data.
echo "This will take a while... By increasing the number of visual-search pods, it can be greatly accelerated."
INGEST_CMD="cds ingest files s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER --collection-id $COLLECTION_ID --s3-profile $PROFILE_NAME --extensions mp4 --num-workers 3 --limit 5"
echo "$INGEST_CMD"
eval "$INGEST_CMD"
  