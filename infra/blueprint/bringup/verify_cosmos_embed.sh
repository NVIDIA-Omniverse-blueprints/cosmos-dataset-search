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

echo "=== Verifying Cosmos-Embed Deployment ==="

# Check if cosmos-embed pod is running
echo "1. Checking pod status..."
kubectl get pods -l app.kubernetes.io/name=cosmos-embed

# Wait for pod to be ready if not already
echo "2. Waiting for cosmos-embed to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=cosmos-embed --timeout=600s

# Get service information
echo "3. Getting service information..."
COSMOS_SERVICE=$(kubectl get service -l app.kubernetes.io/name=cosmos-embed -o jsonpath='{.items[0].metadata.name}')
echo "Service name: $COSMOS_SERVICE"

# Test health endpoint using port-forward
echo "4. Testing health endpoints..."
kubectl port-forward service/$COSMOS_SERVICE 8090:8000 &
PF_PID=$!
sleep 5

# Health check
echo "Testing health endpoints..."
curl -s http://localhost:8090/v1/health/ready || echo "Health check failed"
curl -s http://localhost:8090/v1/health/live || echo "Liveness check failed"

# Test text embedding
echo "5. Testing text embedding..."
curl -X POST http://localhost:8090/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "input": ["The quick brown fox jumps over the lazy dog"],
    "request_type": "query",
    "encoding_format": "float",
    "model": "nvidia/cosmos-embed1"
  }' | jq '.data[0].embedding | length' || echo "Text embedding test failed"

# Clean up port-forward
kill $PF_PID 2>/dev/null

echo "=== Verification Complete ==="
echo "Cosmos-embed service: $COSMOS_SERVICE"
echo "Access via: kubectl port-forward service/$COSMOS_SERVICE 8090:8000"

# Show service details
echo "=== Service Details ==="
kubectl describe service $COSMOS_SERVICE