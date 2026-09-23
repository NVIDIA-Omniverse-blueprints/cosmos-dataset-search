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

# 
# Python Base Dockerfile

FROM nvcr.io/nvidia/base/ubuntu:22.04_20240212


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies and apply available distribution updates to the
# whole installed base, including security fixes beyond a single media library.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libmagic1 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Pin the validated tool release, including its patched quinn-proto dependency.
RUN pip install uv==0.12.13

# Create user first
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Create /app directory owned by appuser  
RUN mkdir -p /app && chown appuser:appuser /app

# Set working directory
WORKDIR /app

# Switch to appuser
USER appuser

# Now everything runs as appuser in appuser-owned directory

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./

# Install dependencies WITHOUT client extras (no Ray, Fire, Rich, python-magic)
# Backend services don't need the CDS client CLI dependencies
RUN echo "Installing Python dependencies (excluding client extras)..." && \
    uv sync --frozen --no-cache --index-strategy unsafe-best-match

# Activate the uv virtual environment for all Python commands
ENV PATH="/app/.venv/bin:$PATH"

RUN python --version && python -c "import sys; print(sys.executable)"
