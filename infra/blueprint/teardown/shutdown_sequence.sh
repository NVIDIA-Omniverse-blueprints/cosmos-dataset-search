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

# Resolve the directory of this script:
# Push into the script directory so that all relative paths work:
# Ensure we pop back to the original directory on exit or error:
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null || exit
trap 'popd > /dev/null' EXIT

### Check required environment variables
if [ -z "$CLUSTER_NAME" ] || [ -z "$AWS_REGION" ] || [ -z "$S3_BUCKET_NAME" ]; then
  echo "Error: Required environment variables not set"
  echo "Please ensure CLUSTER_NAME, AWS_REGION, and S3_BUCKET_NAME are set"
  exit 1
fi

# If running in GitLab CI, check the environment variables are set
if [[ -n "${GITLAB_CI:-}" ]]; then
  echo "Running in GitLab CI environment..."
  if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: Required environment variables not set"
    echo "Please ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set"
    exit 1
  fi
fi

# Skip if -y
if [ "$1" == "-y" ]; then
  echo "Skipping confirmation prompts..."
else
  read -p "Do you want to delete all cluster resources including instances, persistent volumes and S3 bucket? Press enter to continue."
fi

###################### Graceful Application Shutdown ######################
aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"

echo "Gracefully shutting down applications..."

# Delete ingress first to prevent orphaned load balancers
kubectl delete ingress simple-ingress || true

# Delete Helm releases (cleans up services/load balancers properly)
echo "Uninstalling Helm releases..."
helm uninstall visual-search || true
helm uninstall cosmos-embed || true
helm uninstall milvus || true

# Force delete any remaining pods to speed up node drainage
echo "Force deleting remaining pods..."
kubectl delete pods --all --grace-period=0 --force || true

# Delete PVCs while cluster is still running (faster than post-cluster cleanup)
echo "Deleting PVCs while cluster is available..."
kubectl delete pvc --all --timeout=60s || true

# Wait for load balancers to clean up (prevents orphaned ELBs)
echo "Waiting for load balancer cleanup..."
sleep 30


###################### VPC CNI Delete ######################
aws eks delete-addon \
  --cluster-name "$CLUSTER_NAME" \
  --addon-name vpc-cni \
  --region "$AWS_REGION"

###################### Delete Cluster ######################
echo "Deleting EKS cluster $CLUSTER_NAME in region $AWS_REGION..."
eksctl delete cluster \
  --region="$AWS_REGION" \
  --name="$CLUSTER_NAME" \
  --force \
  --wait=true

###################### Delete PVC ######################
echo "Deleting Dynamic Persistent Volumes in cluster $CLUSTER_NAME in region $AWS_REGION..."

# Define the tag key and value
TAG_KEY="KubernetesCluster"
TAG_VALUE="$CLUSTER_NAME"

# Find volumes that match the tag key and value
VOLUME_IDS=$(aws ec2 describe-volumes \
    --query "Volumes[?Tags[?Key=='$TAG_KEY' && Value=='$TAG_VALUE']].VolumeId" \
    --output text)

# Check if any volumes were found
if [ -z "$VOLUME_IDS" ]; then
    echo "No volumes found with tag $TAG_KEY:$TAG_VALUE."
else
    # Loop through the volume IDs and delete each one
    for VOLUME_ID in $VOLUME_IDS; do
        echo "Deleting volume: $VOLUME_ID"
        aws ec2 delete-volume --volume-id "$VOLUME_ID"
        if [ $? -eq 0 ]; then
            echo "Deleted volume: $VOLUME_ID successfully."
        else
            echo "Failed to delete volume: $VOLUME_ID"
        fi
    done
fi

###################### Delete S3 ######################
echo "Deleting S3 bucket: $S3_BUCKET_NAME in region $AWS_REGION..."
aws s3 rb "s3://$S3_BUCKET_NAME" --force > /dev/null 2>&1
echo "Checking if the bucket was deleted..."
IS_BUCKET_EXISTS=$(aws s3 ls "s3://$S3_BUCKET_NAME" 2>&1)

if echo "$IS_BUCKET_EXISTS" | grep -q "NoSuchBucket"; then
  echo "Bucket '$S3_BUCKET_NAME' is deleted."
elif echo "$IS_BUCKET_EXISTS" | grep -q "ExpiredToken"; then
  echo "Bucket check failed. Please try again."
else
  echo "Bucket '$S3_BUCKET_NAME' still exists. This is unexpected."
fi

###################### Delete VPC ######################
source ./delete_vpc.sh
delete_vpc

###################### Delete EIP ######################
EIP=$(aws ec2 describe-addresses \
  --filters "Name=tag:Name,Values=eksctl-${CLUSTER_NAME}/NATIP" \
  --query 'Addresses[*].AllocationId' \
  --output text --region "$AWS_REGION")
NAT=$(aws ec2 describe-nat-gateways \
  --region "$AWS_REGION" \
  --query 'NatGateways[?contains(NatGatewayAddresses[].AllocationId, `${EIP}`)].NatGatewayId' \
  --output text)

if [ -n "$NAT" ]; then
  aws ec2 delete-nat-gateway \
    --nat-gateway-id "${NAT}" \
    --region "$AWS_REGION"
fi

aws ec2 release-address \
  --allocation-id "${EIP}" \
  --region "$AWS_REGION" \

###################### Delete Cluster A Second Time
###################### VNC-CNI Bug######################
echo "Deleting cluster again..."
eksctl delete cluster \
  --region="$AWS_REGION" \
  --name="$CLUSTER_NAME" \
  --force \
  --wait=true
