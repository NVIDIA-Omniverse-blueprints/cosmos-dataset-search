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

## Cosmos Video Search (CVS)

set -eo pipefail

# Resolve the directory of this script:
# Push into the script directory so that all relative paths work:
# Ensure we pop back to the original directory on exit or error:
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null || exit
trap 'popd > /dev/null' EXIT

source ./configuration.sh

### Make your cluster
./make_cluster.sh -y

### Install a high performance default storage class
kubectl apply -f high-performance-storageclass.yaml
eksctl utils associate-iam-oidc-provider \
    --region $AWS_REGION \
    --cluster $CLUSTER_NAME \
    --approve
eksctl create iamserviceaccount \
    --name ebs-csi-controller-sa \
    --namespace kube-system \
    --cluster $CLUSTER_NAME \
    --region $AWS_REGION \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
    --approve \
    --override-existing-serviceaccounts
EBS_ROLE_ARN="$(eksctl get iamserviceaccount --cluster $CLUSTER_NAME --name ebs-csi-controller-sa --namespace kube-system --output json | awk -F'"' '/roleARN/{print $4}')"
eksctl create addon \
    --name aws-ebs-csi-driver \
    --cluster $CLUSTER_NAME \
    --region $AWS_REGION \
    --service-account-role-arn $EBS_ROLE_ARN \
    --force
### Check status of EBS CSI controller status
./check_ebs_csi_driver.sh
eksctl utils migrate-to-pod-identity --cluster $CLUSTER_NAME --approve

### Deploy NVIDIA-daemonset
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
# Patch the daemonset to install on our gpu nodes:
kubectl patch ds nvidia-device-plugin-daemonset -n kube-system --patch '{
  "spec": {
    "template": {
      "spec": {
        "tolerations": [
          {
            "key": "exclusive",
            "operator": "Equal",
            "value": "cvs-gpu",
            "effect": "NoSchedule"
          }
        ]
      }
    }
  }
}'
