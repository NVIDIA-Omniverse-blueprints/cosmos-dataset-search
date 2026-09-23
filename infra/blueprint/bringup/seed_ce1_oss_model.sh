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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_HF_ID="${COSMOS_EMBED_OSS_MODEL_HF_ID:-nvidia/Cosmos-Embed1-224p}"
MODEL_HF_REVISION="${COSMOS_EMBED_OSS_MODEL_HF_REVISION:?COSMOS_EMBED_OSS_MODEL_HF_REVISION must pin an exact Hugging Face commit}"
MODEL_SHA256_FILE="${COSMOS_EMBED_OSS_MODEL_SHA256_FILE:-${SCRIPT_DIR}/cosmos-embed1-224p.sha256}"
MODEL_PVC="${COSMOS_EMBED_OSS_MODEL_PVC:-ce1-oss-model}"
MODEL_PVC_SIZE="${COSMOS_EMBED_OSS_MODEL_PVC_SIZE:-10Gi}"
MODEL_STORAGE_CLASS="${COSMOS_EMBED_OSS_MODEL_STORAGE_CLASS:-high-perf-gp3}"
MODEL_NAMESPACE="${COSMOS_EMBED_OSS_MODEL_NAMESPACE:-default}"
SEEDER_IMAGE="${COSMOS_EMBED_OSS_MODEL_SEEDER_IMAGE:-python@sha256:8c97ebedc32fd60935cdf5992e935753e2a0f98231830028050e1e04bd3c13c2}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-test_artifacts}"

if [[ ! "${MODEL_HF_ID}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "Error: COSMOS_EMBED_OSS_MODEL_HF_ID must be an owner/model Hugging Face ID" >&2
  exit 1
fi

if [[ ! "${MODEL_HF_REVISION}" =~ ^[[:xdigit:]]{40}$ ]]; then
  echo "Error: COSMOS_EMBED_OSS_MODEL_HF_REVISION must be a 40-character commit SHA" >&2
  exit 1
fi

if [[ ! -r "${MODEL_SHA256_FILE}" ]]; then
  echo "Error: CE1 weight checksum manifest is not readable: ${MODEL_SHA256_FILE}" >&2
  exit 1
fi

for value in "${MODEL_PVC}" "${MODEL_NAMESPACE}" "${MODEL_STORAGE_CLASS}"; do
  if [[ ! "${value}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Error: Kubernetes names must be lower-case DNS labels; got '${value}'" >&2
    exit 1
  fi
done

if [[ ! "${MODEL_PVC_SIZE}" =~ ^[1-9][0-9]*(Gi|Ti)$ ]]; then
  echo "Error: COSMOS_EMBED_OSS_MODEL_PVC_SIZE must use Gi or Ti units" >&2
  exit 1
fi

for command in kubectl python3 sha256sum tar; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Error: required command '${command}' is unavailable" >&2
    exit 1
  fi
done

WORK_DIR="$(mktemp -d)"
MODEL_DIR="${WORK_DIR}/cosmos-embed1"
POD_SUFFIX="${CI_COMMIT_SHORT_SHA:-manual}"
POD_SUFFIX="$(printf '%s' "${POD_SUFFIX}" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
SEEDER_POD="ce1-oss-model-seeder-${POD_SUFFIX:-manual}"

cleanup() {
  kubectl delete pod "${SEEDER_POD}" --namespace "${MODEL_NAMESPACE}" \
    --ignore-not-found --wait=false >/dev/null 2>&1 || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

mkdir -p "${ARTIFACTS_DIR}" "${MODEL_DIR}"

echo "Downloading CE1 OSS model ${MODEL_HF_ID}@${MODEL_HF_REVISION}"
python3 - "${MODEL_HF_ID}" "${MODEL_HF_REVISION}" "${MODEL_DIR}" <<'PY'
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

model_id, revision, model_dir = sys.argv[1:]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
resolved_revision = HfApi(token=token).model_info(
    repo_id=model_id,
    revision=revision,
).sha
if resolved_revision != revision:
    raise SystemExit(
        f"Hugging Face resolved {model_id}@{revision} to {resolved_revision}"
    )

snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=model_dir,
    token=token,
)
shutil.rmtree(os.path.join(model_dir, ".cache"), ignore_errors=True)
PY

python3 - "${MODEL_DIR}" <<'PY'
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
required_files = {
    "config.json",
    "model.safetensors.index.json",
    "modeling_embed1.py",
    "preprocessing_embed1.py",
}
found_files = {path.name for path in model_dir.iterdir() if path.is_file()}
missing = sorted(required_files - found_files)
if missing:
    raise SystemExit(f"model snapshot is missing required files: {', '.join(missing)}")
if not any(model_dir.glob("*.safetensors")):
    raise SystemExit("model snapshot contains no root-level safetensor shards")
PY

echo "Verifying CE1 OSS weight shards"
(
  cd "${MODEL_DIR}"
  sha256sum --check --strict "${MODEL_SHA256_FILE}"
)

kubectl apply --namespace "${MODEL_NAMESPACE}" -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${MODEL_PVC}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ${MODEL_STORAGE_CLASS}
  resources:
    requests:
      storage: ${MODEL_PVC_SIZE}
EOF

kubectl delete pod "${SEEDER_POD}" --namespace "${MODEL_NAMESPACE}" \
  --ignore-not-found --wait=true
kubectl apply --namespace "${MODEL_NAMESPACE}" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${SEEDER_POD}
  labels:
    app.kubernetes.io/name: ce1-oss-model-seeder
spec:
  restartPolicy: Never
  imagePullSecrets:
    - name: nvcr-io
  nodeSelector:
    role: cvs-gpu
  tolerations:
    - key: exclusive
      operator: Equal
      value: cvs-gpu
      effect: NoSchedule
  containers:
    - name: seeder
      image: "${SEEDER_IMAGE}"
      imagePullPolicy: IfNotPresent
      command: ["sh", "-c", "trap : TERM INT; sleep 3600 & wait"]
      securityContext:
        runAsUser: 0
        runAsGroup: 0
      resources:
        requests:
          cpu: 100m
          memory: 128Mi
        limits:
          cpu: "1"
          memory: 1Gi
      volumeMounts:
        - name: model
          mountPath: /models/cosmos-embed1
  volumes:
    - name: model
      persistentVolumeClaim:
        claimName: ${MODEL_PVC}
EOF

kubectl wait pod "${SEEDER_POD}" --namespace "${MODEL_NAMESPACE}" \
  --for=condition=Ready --timeout=10m

echo "Hydrating PVC ${MODEL_NAMESPACE}/${MODEL_PVC} with verified CE1 weights"
tar -C "${MODEL_DIR}" -cf - . | \
  kubectl exec --stdin "${SEEDER_POD}" --namespace "${MODEL_NAMESPACE}" -- \
    sh -ceu 'find /models/cosmos-embed1 -mindepth 1 -delete; tar -xf - -C /models/cosmos-embed1; sync'

kubectl exec "${SEEDER_POD}" --namespace "${MODEL_NAMESPACE}" -- sh -ceu '
  test -s /models/cosmos-embed1/config.json
  test -s /models/cosmos-embed1/model.safetensors.index.json
  test -s /models/cosmos-embed1/modeling_embed1.py
  test -s /models/cosmos-embed1/preprocessing_embed1.py
  find /models/cosmos-embed1 -maxdepth 1 -type f -name "*.safetensors" | grep -q .
'

read -r MODEL_SHA256_MANIFEST_DIGEST _ < <(sha256sum "${MODEL_SHA256_FILE}")
cat > "${ARTIFACTS_DIR}/ce1-oss-model-provenance.txt" <<EOF
model_hf_id=${MODEL_HF_ID}
model_hf_revision=${MODEL_HF_REVISION}
model_weight_sha256_manifest=${MODEL_SHA256_MANIFEST_DIGEST}
model_pvc=${MODEL_PVC}
model_namespace=${MODEL_NAMESPACE}
model_storage_class=${MODEL_STORAGE_CLASS}
model_pvc_size=${MODEL_PVC_SIZE}
EOF

kubectl get pvc "${MODEL_PVC}" --namespace "${MODEL_NAMESPACE}" -o wide
echo "CE1 OSS model PVC is ready"
