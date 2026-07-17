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

## Cosmos Video Search (CVS)

# Resolve the directory of this script:
# Push into the script directory so that all relative paths work:
# Ensure we pop back to the original directory on exit or error:
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null || exit
trap 'popd > /dev/null' EXIT


### CI-specific: Check K8s deployment status and configure CLI
if [[ -n "${CUSTOM_API_ENDPOINT:-}" ]]; then
  VS_API="$CUSTOM_API_ENDPOINT"
  echo "Using API endpoint from CUSTOM_API_ENDPOINT: https://$VS_API/api"
elif [[ -n "${GITLAB_CI:-}" ]]; then
  echo "Running in GitLab CI - checking K8s deployment..."
  source ./configuration.sh

  ### Check status of pods
  ./check_pods_status.sh

  ### Poll the UI endpoint
  VS_API=$(kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
  while true; do
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$VS_API")
    if [[ "$status" != "000" ]]; then
      break
    fi
    sleep 10
  done
else
  echo "Skipping K8s deployment checks (not in CI environment)."
  # For local use, set VS_API to the ingress hostname of the CDS deployment
  VS_API=$(docker exec cds-deployment bash -c "kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'")
  echo "Setting this in ~/.config/cds/config as the api_endpoint: https://$VS_API/api"
fi

### Install CDS CLI from packaged source
source ./install_cds_cli.sh

### Activate virtual environment (created by install_cds_cli.sh)
PACKAGED_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ -f "${PACKAGED_ROOT}/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${PACKAGED_ROOT}/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${PACKAGED_ROOT}/.venv"
    exit 1
fi

# Configure the CDS CLI
echo "Configuring CDS CLI..."

# Create config directory if it doesn't exist
mkdir -p ~/.config/cds

# Write config (overwrites existing to avoid duplicates)
cat << EOF > ~/.config/cds/config
[default]
api_endpoint = https://$VS_API/api
EOF

### Validate the server is up and running by querying that the cosmos_video_search_milvus pipeline is available.
echo "Pipelines:"
cds pipelines list | grep 'cosmos_video_search_milvus'

echo CVDS blueprint running at $CLUSTER_NAME. Installation complete.
