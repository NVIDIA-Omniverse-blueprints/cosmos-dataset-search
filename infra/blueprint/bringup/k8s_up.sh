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

helm upgrade --install cosmos-embed ./triton-cosmos-embed \
  --values cosmos-embed-override.yaml \
  --timeout 45m

# Use CI_COMMIT_SHORT_SHA only in CI environment, otherwise use fixed version
if [ -n "${GITLAB_CI:-}" ] && [ "${IS_RELEASE:-false}" = "false" ]; then
  CI_COMMIT_SHORT_SHA=$(git rev-parse --short=8 HEAD) # Short SHA for CI environment
  echo "Running in GitLab CI environment, using image tag: ${CI_COMMIT_SHORT_SHA}"
  helm upgrade --install visual-search visual-search \
    --values values.yaml \
    --set-string visualSearch.image.tag="${CI_COMMIT_SHORT_SHA}"
else
  echo "Running in production environment, using fixed image tag from values.yaml (0.6.0)"
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

INGRESS_HOSTNAME=$(kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
if [[ -n "$INGRESS_HOSTNAME" ]]; then
    echo "Installing visual-search-react-ui with ingress hostname: $INGRESS_HOSTNAME"
    echo "Current directory: $(pwd)"
    echo "Checking for UI values file: $(ls -la visual-search-react-ui/values.yaml 2>/dev/null || echo 'NOT FOUND')"
    helm install visual-search-react-ui ./visual-search-react-ui \
      --values values.yaml \
      --values visual-search-react-ui/values.yaml \
      --set global.ingress.host="$INGRESS_HOSTNAME"
else
    echo "ERROR: Could not determine ingress hostname after waiting"
    echo "The UI deployment requires a valid ingress hostname to set CVDS_UI_URL correctly"
    echo "Please check the ingress status and retry deployment"
    kubectl get ingress simple-ingress -o wide
    exit 1
fi

echo "Performing final health check for all services..."
echo "Will monitor pod status every 2 minutes until all services are ready (no timeout)..."

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
    exit 1
  fi

  if [ "$PENDING_PODS" -eq 0 ] && [ "$NOT_READY" -eq 0 ]; then
    echo "SUCCESS: All services are Ready!"
    echo "Final pod status:"
    kubectl get pods
    break
  fi

  echo "Status summary: pending=$PENDING_PODS, not_ready=$NOT_READY, failed=$FAILED_PODS"
  echo "Waiting 2 minutes before next check..."
  echo ""
  sleep 120
done

echo "Deployment completed successfully."
