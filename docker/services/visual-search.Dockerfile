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

# Visual Search FastAPI service using python-base
ARG PYTHON_BASE_IMAGE=python-base:latest
FROM ${PYTHON_BASE_IMAGE}

# Switch to appuser before copying
USER appuser

# CDS forwards video to CE1. Reject a base that brings its unused decoder back.
RUN python -c "import importlib.util; assert importlib.util.find_spec('av') is None, 'PyAV belongs in the CE1 runtime, not CDS'"

# Copy application code
COPY --chown=appuser:appuser src/triton /app/src/triton
COPY --chown=appuser:appuser src/models /app/src/models
COPY --chown=appuser:appuser src/haystack /app/src/haystack
COPY --chown=appuser:appuser src/wrappers /app/src/wrappers
COPY --chown=appuser:appuser src/visual_search /app/src/visual_search
COPY --chown=appuser:appuser ./LICENSE /app/LICENSE
COPY --chown=appuser:appuser ./OSS_SOURCES.txt /app/OSS_SOURCES.txt

# Expose port (default 8000, configurable via GUNICORN_PORT)
EXPOSE 8000

# Health check - use GUNICORN_PORT environment variable or default to 8000
# Increased timeout to 30s to handle slow Milvus collection metadata operations
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${GUNICORN_PORT:-8000}/health || exit 1

# Set Python path and run the application using gunicorn wrapper
ENV PYTHONPATH="/app"
CMD ["python", "/app/src/wrappers/gunicorn_wrapper.py", "src/visual_search/main.py", "app"]
