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

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null || exit
trap 'popd > /dev/null' EXIT

AWS_BACKUP_DIR="$HOME/.aws_backup_k8s_up"
if [ -d "$HOME/.aws" ]; then
    rm -rf "$AWS_BACKUP_DIR" 2>/dev/null || true
    mv "$HOME/.aws" "$AWS_BACKUP_DIR"
fi
trap 'if [ -d "$AWS_BACKUP_DIR" ]; then rm -rf "$HOME/.aws" 2>/dev/null || true; mv "$AWS_BACKUP_DIR" "$HOME/.aws"; fi; popd > /dev/null' EXIT

source ./configuration.sh

./secrets.sh

# The visual-search Helm release below renders and manages its Secret RBAC.
# Do not pass Helm template files directly to kubectl.

COSMOS_EMBED_SERVICE_MODE="${COSMOS_EMBED_SERVICE_MODE:-oss}"
COSMOS_EMBED_HELM_ARGS=()
EXPECT_CE1_OSS_RUNTIME=true

if [ "${COSMOS_EMBED_SERVICE_MODE}" != "oss" ]; then
  echo "Error: COSMOS_EMBED_SERVICE_MODE must be 'oss' for this release; got '${COSMOS_EMBED_SERVICE_MODE}'" >&2
  exit 1
fi
if [ -z "${COSMOS_EMBED_OSS_MODEL_PVC:-}" ]; then
  echo "Error: COSMOS_EMBED_OSS_MODEL_PVC must name a pre-provisioned PVC" >&2
  echo "CI must provide offline CE1 model weights through that PVC; runtime Hugging Face downloads are disabled by default." >&2
  exit 1
fi

COSMOS_EMBED_OSS_IMAGE_REPOSITORY="${COSMOS_EMBED_OSS_IMAGE_REPOSITORY:-nvcr.io/nvidia/blueprint/cosmos-embed1-oss}"
COSMOS_EMBED_OSS_IMAGE_TAG="${COSMOS_EMBED_OSS_IMAGE_TAG:-1.2.0}"
COSMOS_EMBED_HELM_ARGS+=(
  --set-string "image.repository=${COSMOS_EMBED_OSS_IMAGE_REPOSITORY}"
  --set-string "image.tag=${COSMOS_EMBED_OSS_IMAGE_TAG}"
  --set-string "extraVolumes.ce1-oss-model.persistentVolumeClaim.claimName=${COSMOS_EMBED_OSS_MODEL_PVC}"
  --set-string "envVars.COSMOS_EMBED_ALLOW_HF_DOWNLOAD=${COSMOS_EMBED_ALLOW_HF_DOWNLOAD:-false}"
)
echo "Deploying CE1 OSS PyTorch image ${COSMOS_EMBED_OSS_IMAGE_REPOSITORY}:${COSMOS_EMBED_OSS_IMAGE_TAG} with model PVC ${COSMOS_EMBED_OSS_MODEL_PVC}"

helm upgrade --install cosmos-embed ./cosmos-embed \
  "${COSMOS_EMBED_HELM_ARGS[@]}" \
  --timeout 45m

# Use CI_COMMIT_SHORT_SHA only in CI environment, otherwise use fixed version
if [ -n "${GITLAB_CI:-}" ] && [ "${IS_RELEASE:-false}" = "false" ]; then
  VISUAL_SEARCH_IMAGE_REPOSITORY="${VISUAL_SEARCH_IMAGE_REPOSITORY:-nvcr.io/nvidia/blueprint/cosmos-dataset-search}"
  VISUAL_SEARCH_IMAGE_TAG="${VISUAL_SEARCH_IMAGE_TAG:-${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short=8 HEAD)}}"
  echo "Running in GitLab CI environment, using visual-search image: ${VISUAL_SEARCH_IMAGE_REPOSITORY}:${VISUAL_SEARCH_IMAGE_TAG}"
  helm upgrade --install visual-search visual-search \
    --values values.yaml \
    --set-string visualSearch.image.repository="${VISUAL_SEARCH_IMAGE_REPOSITORY}" \
    --set-string visualSearch.image.tag="${VISUAL_SEARCH_IMAGE_TAG}"
else
  echo "Running in production environment, using fixed image tag from values.yaml (1.2.0)"
  helm upgrade --install visual-search visual-search \
    --values values.yaml
fi

helm repo add milvus https://zilliztech.github.io/milvus-helm

envsubst < milvus-values.yaml | helm upgrade --install milvus milvus/milvus \
  --version 4.2.58 \
  --values - \
  --set image.all.repository=milvusdb/milvus \
  --set image.all.tag=v2.4.4

# Waiting for milvus-querynode so we can replace it, maybe not necessary.
until kubectl get deployment milvus-querynode >/dev/null 2>&1; do sleep 5; done

kubectl patch deployment milvus-querynode -p '{
  "spec": {
    "replicas": 1,
    "strategy": {
      "type": "RollingUpdate",
      "rollingUpdate": { "maxSurge": 0, "maxUnavailable": 1 }
    },
    "template": {
      "spec": {
        "nodeSelector": { "role": "milvus-query", "memory-type": "high" },
        "tolerations": [
          { "key": "dedicated", "operator": "Equal", "value": "querynode", "effect": "NoExecute" }
        ],
        "containers": [{
          "name": "querynode",
          "image": "milvusdb/milvus:v2.4.4",
          "env": [
            { "name": "GOGC", "value": "10" },
            { "name": "GOMEMLIMIT", "value": "110GiB" },
            { "name": "KNOWHERE_ENABLE_GPU", "value": "false" },
            { "name": "MILVUS_ENABLE_GPU", "value": "false" },
            { "name": "KNOWHERE_GPU_MEM_POOL_SIZE", "value": "0" }
          ],
          "resources": {
            "limits": {},
            "requests": {}
          }
        }]
      }
    }
  }
}'

kubectl rollout status deployment/milvus-querynode --timeout=7200s

openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout privateKey.key -out certificate.crt -subj "/C=US/ST=Texas/L=Austin/O=NVIDIA/OU=CVDS/CN=self-signed-tls"
kubectl create secret tls visual-search-tls --key privateKey.key --cert certificate.crt --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.6.4/deploy/static/provider/cloud/deploy.yaml
kubectl wait --namespace ingress-nginx --for=condition=Available --timeout=180s deployment/ingress-nginx-controller

# Probe the admission webhook before creating ingress
echo "Probing ingress-nginx admission webhook readiness..."
echo "Checking webhook pods:"
kubectl get pods -n ingress-nginx -l app.kubernetes.io/component=controller
echo ""
echo "Checking webhook service:"
kubectl get svc -n ingress-nginx ingress-nginx-controller-admission
echo ""
echo "Checking webhook endpoints:"
kubectl get endpoints -n ingress-nginx ingress-nginx-controller-admission
echo ""

# Wait for admission webhook to be fully ready
echo "Waiting for admission webhook to be fully operational..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

# Additional buffer for webhook initialization
echo "Giving webhook service 15 seconds to stabilize..."
sleep 15

# Create ingress with retry logic
echo "Creating ingress resource..."
MAX_RETRIES=3
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if kubectl apply -f ingress/ingress.yaml 2>&1 | tee /tmp/ingress_apply.log; then
    echo "Ingress created successfully"
    break
  else
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Ingress creation failed. Error details:"
    cat /tmp/ingress_apply.log
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo "Retrying in 15s... (attempt $RETRY_COUNT/$MAX_RETRIES)"
      echo "Current webhook status:"
      kubectl get validatingwebhookconfigurations ingress-nginx-admission -o yaml | grep -A 5 "webhooks:"
      sleep 15
    else
      echo "Failed to create ingress after $MAX_RETRIES attempts"
      echo "Final diagnostics:"
      kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller --tail=50
      exit 1
    fi
  fi
done

./check_ingress_hostname.sh

echo "Performing final health check for all services..."
K8S_HEALTH_CHECK_TIMEOUT_SECONDS="${K8S_HEALTH_CHECK_TIMEOUT_SECONDS:-3600}"
if [[ ! "${K8S_HEALTH_CHECK_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: K8S_HEALTH_CHECK_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
HEALTH_CHECK_DEADLINE=$((SECONDS + K8S_HEALTH_CHECK_TIMEOUT_SECONDS))

dump_cosmos_embed_diagnostics() {
  local pod_name
  pod_name="$(kubectl get pods \
    -l app.kubernetes.io/name=cosmos-embed \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [ -z "${pod_name}" ]; then
    echo "No Cosmos-Embed pod exists for diagnostics."
    return
  fi
  echo "Cosmos-Embed pod diagnostics for ${pod_name}:"
  kubectl describe pod "${pod_name}" || true
  echo "Cosmos-Embed current logs:"
  kubectl logs "${pod_name}" --tail=500 || true
  echo "Cosmos-Embed previous logs, when available:"
  kubectl logs "${pod_name}" --previous --tail=500 || true
}

echo "Will monitor pod status every 2 minutes for up to ${K8S_HEALTH_CHECK_TIMEOUT_SECONDS} seconds..."

while true; do
  echo "==================== $(date) ===================="
  echo "Current pod status:"
  kubectl get pods
  echo ""

  PODS_TABLE="$(kubectl get pods --no-headers 2>/dev/null | sed '/^No resources found/d')"

  FAILED_PODS=$(
    printf "%s\n" "$PODS_TABLE" | awk '
      $3 ~ /^(Failed|Error|CrashLoopBackOff|ImagePullBackOff|ErrImagePull|CreateContainerConfigError|CreateContainerError|RunContainerError|ContainerCannotRun|StartError|StartContainerError)$/ {c++}
      END{print c+0}'
  )

  PENDING_PODS=$(
    printf "%s\n" "$PODS_TABLE" | awk '
      $3=="Pending" || $3=="ContainerCreating" {c++}
      END{print c+0}'
  )

  NOT_READY=$(
    printf "%s\n" "$PODS_TABLE" | awk '
      $3=="Running" {
        n=split($2,a,"/");
        if (n!=2 || a[1]!=a[2]) c++
      }
      END{print c+0}'
  )

  if [ "$FAILED_PODS" -gt 0 ]; then
    echo "ERROR: Found $FAILED_PODS failed pods."
    kubectl get pods | awk 'NR==1 || $3 ~ /^(Failed|Error|CrashLoopBackOff|ImagePullBackOff|ErrImagePull|CreateContainerConfigError|CreateContainerError|RunContainerError|ContainerCannotRun|StartError|StartContainerError)$/'
    dump_cosmos_embed_diagnostics
    exit 1
  fi

  if [ "${EXPECT_CE1_OSS_RUNTIME}" = "true" ]; then
    COSMOS_EMBED_RESTARTS="$(kubectl get pods \
      -l app.kubernetes.io/name=cosmos-embed \
      -o jsonpath='{range .items[*].status.containerStatuses[*]}{.restartCount}{"\n"}{end}' \
      2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
    if [ "${COSMOS_EMBED_RESTARTS}" -gt 0 ]; then
      echo "ERROR: CE1 OSS restarted ${COSMOS_EMBED_RESTARTS} time(s) before becoming ready."
      dump_cosmos_embed_diagnostics
      exit 1
    fi
  fi

  if [ "$PENDING_PODS" -eq 0 ] && [ "$NOT_READY" -eq 0 ]; then
    echo "SUCCESS: All services are Ready!"
    echo "Final pod status:"
    kubectl get pods
    break
  fi

  if [ "${SECONDS}" -ge "${HEALTH_CHECK_DEADLINE}" ]; then
    echo "ERROR: Services did not become ready within ${K8S_HEALTH_CHECK_TIMEOUT_SECONDS} seconds."
    dump_cosmos_embed_diagnostics
    exit 1
  fi

  echo "Status summary: pending=$PENDING_PODS, not_ready=$NOT_READY, failed=$FAILED_PODS"
  echo "Waiting 2 minutes before next check..."
  echo ""
  sleep 120
done

echo "Deployment completed successfully."
