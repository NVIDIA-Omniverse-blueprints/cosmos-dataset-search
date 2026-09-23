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

# PyTorch-capable CE1 OSS service image using the CVDS Python base runtime.
ARG PYTHON_BASE_IMAGE=python-base:latest
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

ENV COSMOS_EMBED_DIM=256 \
    COSMOS_EMBED_BACKEND=pytorch \
    COSMOS_EMBED_ALLOW_HF_DOWNLOAD=false \
    COSMOS_EMBED_GPU_FRAMES=true \
    COSMOS_EMBED_GPU_VIDEO=true \
    COSMOS_EMBED_HF_MODEL_ID=nvidia/Cosmos-Embed1-224p \
    COSMOS_EMBED_LOCAL_FILES_ONLY=true \
    COSMOS_EMBED_MAX_MEDIA_BYTES=100663296 \
    COSMOS_EMBED_MAX_BATCH_SIZE=64 \
    COSMOS_EMBED_MODEL=nvidia/cosmos-embed1 \
    COSMOS_EMBED_MODEL_PATH=/models/cosmos-embed1 \
    COSMOS_EMBED_MODEL_VARIANT=224p \
    COSMOS_EMBED_PREPARATION_WORKERS=8 \
    COSMOS_EMBED_SDPA=true \
    COSMOS_EMBED_SERVICE_VERSION=1.2.0 \
    COSMOS_EMBED_TMP_DIR=/tmp/ram \
    COSMOS_EMBED_TORCH_COMPILE=true \
    COSMOS_EMBED_MEDIA_DOWNLOAD_TIMEOUT_SECONDS=300 \
    HF_HOME=/home/appuser/.cache/huggingface \
    NIM_HTTP_API_PORT=8000 \
    NIM_HTTP_HOST=0.0.0.0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER root

RUN mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser/.cache/huggingface

USER appuser

# CE1 uses NVDEC video decoding; do not ship the optional benchmark PyAV wheel.
RUN uv sync --frozen --no-cache --extra gpu-media --index-strategy unsafe-best-match \
    && python -c "import importlib.util; assert importlib.util.find_spec('av') is None"

COPY --chown=appuser:appuser src/__init__.py /app/src/__init__.py
COPY --chown=appuser:appuser src/cosmos_embed_oss /app/src/cosmos_embed_oss
COPY --chown=appuser:appuser LICENSE /app/LICENSE

RUN python -m src.cosmos_embed_oss.runtime_check --json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import os, urllib.request; port=os.environ.get('NIM_HTTP_API_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health/ready', timeout=2).read()"

CMD ["python", "-m", "src.cosmos_embed_oss.server"]
