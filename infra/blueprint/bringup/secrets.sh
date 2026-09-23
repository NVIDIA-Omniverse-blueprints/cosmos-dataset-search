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

# Ensure NGC_API_KEY is set
if [ -z "$NGC_API_KEY" ]; then
  echo "Error: NGC_API_KEY is not set"
  exit 1
fi

# Ensure DOCKER_PAT key is set
if [ -z "$DOCKER_PAT" ] ; then
  echo "Error: DOCKER_PAT is not set"
  exit 1
fi

# Ensure DOCKER_USER key is set
if [ -z "$DOCKER_USER" ] ; then
  echo "Error: DOCKER_USER is not set"
  exit 1
fi

# CI-hosted CDS images use NVCV_NGC_KEY; other deployments retain NGC_API_KEY.
NVCR_PULL_KEY="${NVCV_NGC_KEY:-$NGC_API_KEY}"
NVCR_AUTH=$(echo -n "\$oauthtoken:$NVCR_PULL_KEY" | base64 -w 0)

# Compute the docker auth value
DOCKER_AUTH=$(echo -n "$DOCKER_USER:$DOCKER_PAT" | base64 -w 0)

# Keep pull credentials outside the checkout and CI artifacts.
umask 077
export REGISTRY_CONFIG_PATH
REGISTRY_CONFIG_PATH=$(mktemp)
trap 'rm -f -- "$REGISTRY_CONFIG_PATH"' EXIT
cat <<EOF > "$REGISTRY_CONFIG_PATH"
{
  "auths": {
    "nvcr.io": {
      "auth": "$NVCR_AUTH",
      "username": "\$oauthtoken",
      "password": "$NVCR_PULL_KEY",
      "email": "user@example.com"
    },
    "https://index.docker.io/v1/": {
      "auth": "$DOCKER_AUTH"
    }
  }
}
EOF

# Delete the existing secret (if any)
kubectl delete secret nvcr-io --ignore-not-found

# Create the Kubernetes secret
kubectl create secret generic nvcr-io \
  --from-file=".dockerconfigjson=$REGISTRY_CONFIG_PATH" \
  --type=kubernetes.io/dockerconfigjson

kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "nvcr-io"}]}'

# Clean up the temporary file
rm -f -- "$REGISTRY_CONFIG_PATH"

### Create cosmos-embed specific secrets (using NGC_API_KEY)
echo "Creating cosmos-embed secrets with NGC_API_KEY..."

# Create ngc-api secret (for runtime NGC API access)
kubectl create secret generic ngc-api \
  --from-literal=NGC_API_KEY="$NGC_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

# Create ngc-secret (for image pulling)
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Cosmos-embed secrets created successfully"

# Secrets key
# First check if the secret already exists
if kubectl get secret secret-encryption-key > /dev/null 2>&1; then
    echo "Secret already exists, keeping existing SECRETS_ENCRYPTION_KEY"
else
    # Only generate new key if secret doesn't exist and env var isn't set
    if [ -z "$SECRETS_ENCRYPTION_KEY" ]; then
        echo "SECRETS_ENCRYPTION_KEY is not set."
	echo "Generating new SECRETS_ENCRYPTION_KEY using command: 'python3 generate_secret_key.py'"
        export SECRETS_ENCRYPTION_KEY=$(python3 generate_secret_key.py | grep "export" | cut -d"'" -f2)
    fi

    echo "Creating new secret with SECRETS_ENCRYPTION_KEY"
    kubectl create secret generic secret-encryption-key \
            --from-literal=SECRETS_ENCRYPTION_KEY="$SECRETS_ENCRYPTION_KEY"
fi

# check for NVCF secrets file
if [ -f "/var/secrets/secrets.json" ]; then
    echo "NVCF secrets file found."
    if grep -q "$NGC_API_KEY_NAME" "/var/secrets/secrets.json"; then
        echo "NGC_API_KEY_NAME found in NVCF secrets file, exporting path"
        export NGC_SECRETS_FILE_PATH="/var/secrets/secrets.json"
    else
        echo "NGC_API_KEY_NAME not found in NVCF secrets file, exiting"
        exit 1
    fi
else
    echo "NVCF secrets file not found, exiting"
    exit 0
fi
