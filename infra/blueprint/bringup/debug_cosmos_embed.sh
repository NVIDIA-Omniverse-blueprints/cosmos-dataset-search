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

echo "=== COSMOS-EMBED DEBUGGING SCRIPT ==="
echo "Timestamp: $(date)"
echo

echo "=== POD STATUS ==="
kubectl get pods -l app.kubernetes.io/name=cosmos-embed -o wide
echo

echo "=== POD DETAILS ==="
POD_NAME=$(kubectl get pods -l app.kubernetes.io/name=cosmos-embed -o jsonpath='{.items[0].metadata.name}')
if [ -n "$POD_NAME" ]; then
    echo "Describing pod: $POD_NAME"
    kubectl describe pod $POD_NAME
    echo
    echo "=== POD EVENTS ==="
    kubectl get events --field-selector involvedObject.name=$POD_NAME --sort-by='.lastTimestamp'
else
    echo "No cosmos-embed pod found"
fi
echo

echo "=== NODE RESOURCES ==="
kubectl get nodes -o wide
echo
kubectl describe nodes | grep -A 10 -B 5 "nvidia.com/gpu"
echo

echo "=== GPU NODE LABELS ==="
kubectl get nodes --show-labels | grep -E "(cvs-gpu|nvidia)"
echo

echo "=== STORAGE CLASSES ==="
kubectl get storageclass
echo

echo "=== PVC STATUS ==="
kubectl get pvc
echo

echo "=== HELM RELEASE STATUS ==="
helm list
echo
helm status cosmos-embed
echo

echo "=== SERVICE STATUS ==="
kubectl get service -l app.kubernetes.io/name=cosmos-embed
echo

echo "=== RESOURCE QUOTAS ==="
kubectl describe resourcequotas 2>/dev/null || echo "No resource quotas found"
echo

echo "=== END DEBUG SCRIPT ==="