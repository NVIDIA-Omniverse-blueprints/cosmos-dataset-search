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

LOG_PREFIX="[validate-env]"
ROOT_DIR="$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="${ROOT_DIR}/.env"
ENV_TEMPLATE="${ROOT_DIR}/.env.example"

log_info() {
  local message="$1"
  printf '%s[INFO] %s\n' "$LOG_PREFIX" "$message"
}

log_warn() {
  local message="$1"
  printf '%s[WARN] %s\n' "$LOG_PREFIX" "$message"
}

log_error() {
  local message="$1"
  printf '%s[ERROR] %s\n' "$LOG_PREFIX" "$message" >&2
}

fail() {
  local message="$1"
  log_error "$message"
  exit 1
}

mask_secret() {
  local value="$1"
  local length
  length=${#value}

  if (( length <= 8 )); then
    printf '********'
    return
  fi

  local prefix suffix
  prefix=${value:0:4}
  suffix=${value:length-4:4}
  printf '%s****%s' "$prefix" "$suffix"
}

check_required() {
  local var_name="$1"
  local value
  value=${!var_name-}

  if [[ -z "${value:-}" ]]; then
    fail "${var_name} is not set. Populate it in your .env file before continuing."
  fi
}

check_not_placeholder() {
  local var_name="$1"
  local placeholder="$2"
  local value
  value=${!var_name-}

  if [[ "$value" == "$placeholder" ]]; then
    fail "${var_name} is still set to the placeholder value (${placeholder}). Update it in your .env file."
  fi
}

check_readable_directory() {
  local setting_name="$1"
  local value="$2"

  if [[ ! -d "$value" ]]; then
    fail "${setting_name} points to '${value}', but that directory does not exist. Create it or update the value."
  fi

  if [[ ! -r "$value" || ! -x "$value" ]]; then
    fail "${setting_name} points to '${value}', but it is not readable and traversable. Run 'chmod -R a+rX ${value}' or update the value."
  fi
}

ce1_model_download_command() {
  printf "make download-ce1-model"
}

check_cosmos_embed_model_files() {
  local model_dir="$1"
  local required_file
  local missing_files=()
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

  for required_file in "${required_files[@]}"; do
    if [[ ! -s "${model_dir}/${required_file}" ]]; then
      missing_files+=("$required_file")
    fi
  done
  if (( ${#missing_files[@]} > 0 )); then
    fail "CE1 OSS model directory is incomplete; missing or empty: ${missing_files[*]}. Re-download it with: $(ce1_model_download_command)"
  fi

  # The validation image is intentionally minimal, so check the pinned config's
  # required schema markers without adding a second JSON runtime.
  if ! grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"cosmos-embed1"' \
    "${model_dir}/config.json"; then
    fail "CE1 OSS config.json does not declare model_type 'cosmos-embed1'. Re-download it with: $(ce1_model_download_command)"
  fi
  if ! grep -Eq '"AutoModel"[[:space:]]*:[[:space:]]*"modeling_embed1\.CosmosEmbed1"' \
    "${model_dir}/config.json"; then
    fail "CE1 OSS config.json does not declare the Cosmos-Embed AutoModel mapping. Re-download it with: $(ce1_model_download_command)"
  fi
}

check_integer() {
  local var_name="$1"
  local min_value="$2"
  local value
  value=${!var_name-}

  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    fail "${var_name} must be an integer. Current value: '${value}'."
  fi

  if (( value < min_value )); then
    fail "${var_name} must be greater than or equal to ${min_value}. Current value: ${value}."
  fi
}

check_log_level() {
  local var_name="$1"
  local allowed_values="$2"
  local value
  value=${!var_name-}

  if [[ -n "$value" ]]; then
    local match=false
    local level
    for level in $allowed_values; do
      if [[ "$value" == "$level" ]]; then
        match=true
        break
      fi
    done

    if [[ "$match" == false ]]; then
      fail "${var_name} must be one of ${allowed_values}. Current value: '${value}'."
    fi
  fi
}

check_cuda_devices() {
  local var_name="$1"
  local value
  value=${!var_name-}

  if [[ -z "$value" ]]; then
    fail "${var_name} is empty. Set it to the GPU device IDs that Milvus can use (e.g., 0 or 0,1)."
  fi

  if ! [[ "$value" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    fail "${var_name} must be a comma-separated list of GPU indices (e.g., '0' or '0,1'). Current value: '${value}'."
  fi
}

check_uri() {
  local var_name="$1"
  local value
  value=${!var_name-}

  if ! [[ "$value" =~ ^https?:// ]]; then
    fail "${var_name} must start with http:// or https://. Current value: '${value}'."
  fi
}

check_boolean() {
  local var_name="$1"
  local value
  value=${!var_name-}

  case "${value,,}" in
    1|true|yes|y|on|0|false|no|n|off)
      return
      ;;
    *)
      fail "${var_name} must be a boolean. Current value: '${value}'."
      ;;
  esac
}

check_cosmos_embed_backend() {
  local value="${COSMOS_EMBED_BACKEND,,}"

  case "$value" in
    ce1-pytorch|pytorch|pytorch-ce1)
      return
      ;;
    *)
      fail "COSMOS_EMBED_BACKEND must select the PyTorch backend. Current value: '${COSMOS_EMBED_BACKEND}'."
      ;;
  esac
}

hf_model_cache_exists() {
  local model_id="$1"
  local cache_name="models--${model_id//\//--}"
  local roots=()
  local root

  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    roots+=("$HF_HUB_CACHE")
  fi
  if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    roots+=("$HUGGINGFACE_HUB_CACHE")
  fi
  if [[ -n "${COSMOS_EMBED_HF_CACHE_VALIDATION_PATH:-}" ]]; then
    roots+=("$COSMOS_EMBED_HF_CACHE_VALIDATION_PATH")
    roots+=("$COSMOS_EMBED_HF_CACHE_VALIDATION_PATH/hub")
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    roots+=("$HF_HOME/hub")
  fi
  roots+=("$HOME/.cache/huggingface/hub")

  for root in "${roots[@]}"; do
    if [[ -d "$root/$cache_name/snapshots" ]]; then
      return 0
    fi
  done

  return 1
}

check_cosmos_embed_oss_config() {
  check_cosmos_embed_backend
  check_boolean COSMOS_EMBED_ALLOW_HF_DOWNLOAD
  check_required COSMOS_EMBED_HF_REVISION

  if [[ "$COSMOS_EMBED_MODEL_VARIANT" != "224p" ]]; then
    fail "COSMOS_EMBED_MODEL_VARIANT must be 224p for the CVDS 256-d CE1 contract."
  fi

  if [[ -n "${COSMOS_EMBED_MODEL_PATH:-}" || -n "${COSMOS_EMBED_WEIGHTS_DIR:-}" ]]; then
    if [[ -n "${COSMOS_EMBED_MODEL_HOST_PATH:-}" ]]; then
      check_not_placeholder COSMOS_EMBED_MODEL_HOST_PATH "/path/to/cosmos-embed1"
      if [[ -n "${COSMOS_EMBED_MODEL_VALIDATION_PATH:-}" ]]; then
        local model_validation_path="$COSMOS_EMBED_MODEL_VALIDATION_PATH"
        check_readable_directory \
          "COSMOS_EMBED_MODEL_HOST_PATH validation mount" \
          "$model_validation_path"
      else
        local model_validation_path="$COSMOS_EMBED_MODEL_HOST_PATH"
        if [[ ! -d "$model_validation_path" ]]; then
          fail "COSMOS_EMBED_MODEL_HOST_PATH points to '${model_validation_path}', but the CE1 OSS model directory does not exist. Download the configured snapshot with: $(ce1_model_download_command)"
        fi
        check_readable_directory \
          "COSMOS_EMBED_MODEL_HOST_PATH" \
          "$model_validation_path"
      fi
      check_cosmos_embed_model_files "$model_validation_path"
    else
      fail "COSMOS_EMBED_MODEL_HOST_PATH is not set. Set it to the host directory containing the CE1 weights mounted at ${COSMOS_EMBED_MODEL_PATH}."
    fi
    log_info "CE1 OSS model files are complete and valid: ${COSMOS_EMBED_MODEL_HOST_PATH}"
    return
  fi

  if [[ "${COSMOS_EMBED_ALLOW_HF_DOWNLOAD,,}" != "true" ]]; then
    fail "CE1 OSS PyTorch requires COSMOS_EMBED_MODEL_PATH or COSMOS_EMBED_WEIGHTS_DIR. Set COSMOS_EMBED_ALLOW_HF_DOWNLOAD=true only for explicit Hugging Face model loading."
  fi

  if [[ -z "${COSMOS_EMBED_HF_MODEL_ID:-}" ]]; then
    fail "COSMOS_EMBED_HF_MODEL_ID must not be empty when COSMOS_EMBED_ALLOW_HF_DOWNLOAD=true."
  fi

  if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    log_info "CE1 OSS Hugging Face model loading has an auth token configured."
    return
  fi

  if hf_model_cache_exists "$COSMOS_EMBED_HF_MODEL_ID"; then
    log_info "CE1 OSS Hugging Face model loading will use an existing local cache."
    return
  fi

  fail "COSMOS_EMBED_ALLOW_HF_DOWNLOAD=true requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN, or cached Hugging Face files for ${COSMOS_EMBED_HF_MODEL_ID}."
}

check_docker_login() {
  local registry="$1"
  local docker_config="${HOME}/.docker/config.json"
  
  if [[ ! -f "$docker_config" ]]; then
    fail "Docker config not found at ${docker_config}. Run 'docker login ${registry}' first."
  fi
  
  # Check if registry is in docker config
  if ! grep -q "\"${registry}\"" "$docker_config" 2>/dev/null; then
    fail "Not logged in to ${registry}. Run 'docker login ${registry}' with your NGC API key."
  fi
  
  log_info "Docker authentication for ${registry} is configured."
}

log_info "Starting environment validation..."

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${ENV_TEMPLATE}" ]]; then
    log_info "'.env' not found. Copying from template..."
    cp "${ENV_TEMPLATE}" "${ENV_FILE}"
    log_warn "Created a fresh .env from template. Please review and update placeholder values before re-running."
    exit 1
  else
    fail "'.env' file is missing and template '.env.example' not found at ${ROOT_DIR}."
  fi
fi

set -a
source "${ENV_FILE}"
set +a

# Compose defaults only apply when variables are unset. A local .env can contain
# empty overrides, so normalize optional knobs here before validation.
: "${COSMOS_EMBED_SERVICE_MODE:=oss}"
: "${COSMOS_EMBED_BACKEND:=pytorch}"
: "${COSMOS_EMBED_MODEL_VARIANT:=224p}"
: "${COSMOS_EMBED_ALLOW_HF_DOWNLOAD:=false}"
: "${COSMOS_EMBED_HF_MODEL_ID:=nvidia/Cosmos-Embed1-224p}"
: "${COSMOS_EMBED_LOCAL_FILES_ONLY:=true}"
: "${COSMOS_EMBED_MAX_MEDIA_BYTES:=100663296}"
: "${COSMOS_EMBED_MEDIA_DOWNLOAD_TIMEOUT_SECONDS:=300}"

# Required values
check_required DATA_DIR
check_required AWS_ACCESS_KEY_ID
check_required AWS_SECRET_ACCESS_KEY
check_required COSMOS_EMBED_URI
check_required ALLOWED_PIPELINES
if [[ "$COSMOS_EMBED_SERVICE_MODE" != "oss" ]]; then
  fail "COSMOS_EMBED_SERVICE_MODE must be 'oss' for this release. Current value: '${COSMOS_EMBED_SERVICE_MODE}'."
fi
log_info "Cosmos Embed service mode: ${COSMOS_EMBED_SERVICE_MODE}"

check_required NVIDIA_API_KEY

# Validate core values
check_not_placeholder DATA_DIR "/path/to/your/data"

check_not_placeholder NVIDIA_API_KEY "<your_api_key>"
log_info "NVIDIA_API_KEY detected: $(mask_secret "$NVIDIA_API_KEY")"

check_cosmos_embed_oss_config
check_boolean COSMOS_EMBED_LOCAL_FILES_ONLY
check_integer COSMOS_EMBED_MAX_MEDIA_BYTES 1
check_integer COSMOS_EMBED_MEDIA_DOWNLOAD_TIMEOUT_SECONDS 1
log_info "Data directory: ${DATA_DIR}"
if [[ -n "${DATA_DIR_VALIDATION_PATH:-}" ]]; then
  check_readable_directory "DATA_DIR validation mount" "$DATA_DIR_VALIDATION_PATH"
else
  check_readable_directory "DATA_DIR" "$DATA_DIR"
fi
log_info "Data directory is readable."

# C-Radio, Cosmos Reason, and caption embedding remain NIM services when only
# Cosmos Embed switches to OSS, so the standalone stack always needs NGC auth.
check_docker_login "nvcr.io"

# Numeric validations
check_integer GUNICORN_PORT 1
check_integer GUNICORN_WORKERS 1
check_integer GUNICORN_TIMEOUT 30
check_integer MAX_DOCS_PER_UPLOAD 1
check_integer COSMOS_EMBED_GPUS 0

check_cuda_devices MILVUS_CUDA_VISIBLE_DEVICES

# URI validations
check_uri COSMOS_EMBED_URI

if [[ -n "${COSMOS_EMBED_PRESIGNED_URL_ENDPOINT_URL-}" ]]; then
  check_uri COSMOS_EMBED_PRESIGNED_URL_ENDPOINT_URL
fi

if [[ -n "${AWS_ENDPOINT_URL-}" ]]; then
  check_uri AWS_ENDPOINT_URL
fi

# Log level validations
check_log_level VISUAL_SEARCH_LOG_LEVEL "DEBUG INFO WARNING ERROR CRITICAL"

# AWS defaults (acceptable for LocalStack)
if [[ "$AWS_ACCESS_KEY_ID" == "test" && "$AWS_SECRET_ACCESS_KEY" == "test" ]]; then
  log_info "Using AWS test credentials (expected for LocalStack)."
fi

log_info "Environment validation completed successfully."
