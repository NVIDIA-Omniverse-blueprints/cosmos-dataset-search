#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

LOG_PREFIX="[download-ce1-model]"
STANDALONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${STANDALONE_DIR}/../.." && pwd)"
ENV_FILE="${STANDALONE_DIR}/.env"

fail() {
  printf '%s[ERROR] %s\n' "$LOG_PREFIX" "$1" >&2
  exit 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  fail "${ENV_FILE} is missing. Copy deploy/standalone/.env.example to deploy/standalone/.env and configure it first."
fi

set -a
source "$ENV_FILE"
set +a

: "${COSMOS_EMBED_MODEL_HOST_PATH:=${HOME}/.cache/cosmos-embed1}"

if [[ -z "${COSMOS_EMBED_HF_MODEL_ID:-}" ]]; then
  fail "COSMOS_EMBED_HF_MODEL_ID is empty. Set it in deploy/standalone/.env."
fi
if [[ -z "${COSMOS_EMBED_HF_REVISION:-}" ]]; then
  fail "COSMOS_EMBED_HF_REVISION is empty. Set it in deploy/standalone/.env."
fi

model_files_are_complete() {
  local required_file
  local required_files=(
    config.json
    configuration_embed1.py
    model.safetensors.index.json
    modeling_embed1.py
    preprocessing_embed1.py
    processor_config.json
    vocab.txt
    model-00001-of-00005.safetensors
    model-00002-of-00005.safetensors
    model-00003-of-00005.safetensors
    model-00004-of-00005.safetensors
    model-00005-of-00005.safetensors
  )

  [[ -d "$COSMOS_EMBED_MODEL_HOST_PATH" ]] || return 1
  for required_file in "${required_files[@]}"; do
    [[ -s "${COSMOS_EMBED_MODEL_HOST_PATH}/${required_file}" ]] || return 1
  done
  grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"cosmos-embed1"' \
    "${COSMOS_EMBED_MODEL_HOST_PATH}/config.json" || return 1
  grep -Eq '"AutoModel"[[:space:]]*:[[:space:]]*"modeling_embed1\.CosmosEmbed1"' \
    "${COSMOS_EMBED_MODEL_HOST_PATH}/config.json" || return 1
}

if model_files_are_complete; then
  chmod -R a+rX "$COSMOS_EMBED_MODEL_HOST_PATH"
  printf '%s[INFO] Complete CE1 OSS model snapshot already exists at %s; skipping download.\n' \
    "$LOG_PREFIX" \
    "$COSMOS_EMBED_MODEL_HOST_PATH"
  exit 0
fi

if [[ -x "${REPO_ROOT}/.venv/bin/hf" ]]; then
  HF_CLI="${REPO_ROOT}/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  HF_CLI="$(command -v hf)"
else
  fail "The Hugging Face CLI is not installed. Run 'make install' and retry."
fi

mkdir -p "$COSMOS_EMBED_MODEL_HOST_PATH"
printf '%s[INFO] Downloading %s at revision %s to %s\n' \
  "$LOG_PREFIX" \
  "$COSMOS_EMBED_HF_MODEL_ID" \
  "$COSMOS_EMBED_HF_REVISION" \
  "$COSMOS_EMBED_MODEL_HOST_PATH"
"$HF_CLI" download "$COSMOS_EMBED_HF_MODEL_ID" \
  --revision "$COSMOS_EMBED_HF_REVISION" \
  --local-dir "$COSMOS_EMBED_MODEL_HOST_PATH"
chmod -R a+rX "$COSMOS_EMBED_MODEL_HOST_PATH"
printf '%s[INFO] CE1 OSS model files are ready.\n' "$LOG_PREFIX"
