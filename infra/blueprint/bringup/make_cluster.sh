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

set -e

check_vpc_capacity() {
  local vpc_count
  local vpc_quota
  local vpc_quota_integer

  vpc_count=$(aws ec2 describe-vpcs \
    --region "$AWS_REGION" \
    --query 'length(Vpcs)' \
    --output text)

  if ! vpc_quota=$(aws service-quotas get-service-quota \
    --service-code vpc \
    --quota-code L-F678F1CE \
    --region "$AWS_REGION" \
    --query 'Quota.Value' \
    --output text 2>/dev/null); then
    echo "Warning: unable to read the VPC quota; continuing with cluster creation"
    return 0
  fi

  vpc_quota_integer="${vpc_quota%%.*}"
  echo "VPC capacity in $AWS_REGION: ${vpc_count}/${vpc_quota_integer}"
  if [ "$vpc_count" -ge "$vpc_quota_integer" ]; then
    echo "Error: no VPC capacity is available in $AWS_REGION."
    echo "Delete an unused pipeline-managed environment or raise AWS quota L-F678F1CE, then retry this pipeline."
    return 1
  fi
}

# Ensure AWS_REGION is set
if [ -z "$AWS_REGION" ]; then
  echo "Error: AWS_REGION is not set"
  exit 1
fi

if [ -z "$CLUSTER_NAME" ]; then
  echo "Error: CLUSTER_NAME is not set"
  exit 1
fi

if [ "$1" == '-y' ]; then
  echo "Creating EKS cluster $CLUSTER_NAME in $AWS_REGION"
else
  echo "Creating EKS cluster $CLUSTER_NAME in $AWS_REGION, continue?"
  read -p "Press enter to continue"
fi

envsubst < cvs.yaml > cvs-rendered.yaml

# Check if cluster already exists or if there are orphaned CloudFormation stacks
EKSCTL_CREATE_TIMEOUT="${EKSCTL_CREATE_TIMEOUT:-35m}"
echo "Checking for existing cluster or orphaned CloudFormation stacks..."
if eksctl get cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Cluster $CLUSTER_NAME already exists, skipping creation"
else
  # Check for orphaned CloudFormation stacks
  CLUSTER_STACK_NAME="eksctl-${CLUSTER_NAME}-cluster"
  if aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$CLUSTER_STACK_NAME" >/dev/null 2>&1; then
    echo "Found orphaned CloudFormation stack for cluster $CLUSTER_NAME, cleaning up..."
    eksctl delete cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --wait || true
    if aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$CLUSTER_STACK_NAME" >/dev/null 2>&1; then
      echo "EKS cleanup left $CLUSTER_STACK_NAME behind; deleting it directly..."
      aws cloudformation update-termination-protection \
        --no-enable-termination-protection \
        --region "$AWS_REGION" \
        --stack-name "$CLUSTER_STACK_NAME" >/dev/null 2>&1 || true
      aws cloudformation delete-stack \
        --region "$AWS_REGION" \
        --stack-name "$CLUSTER_STACK_NAME"
      timeout "$EKSCTL_CREATE_TIMEOUT" \
        aws cloudformation wait stack-delete-complete \
        --region "$AWS_REGION" \
        --stack-name "$CLUSTER_STACK_NAME"
    fi
    echo "Cleanup completed, proceeding with cluster creation..."
  fi

  check_vpc_capacity
  echo "Using eksctl operation timeout: $EKSCTL_CREATE_TIMEOUT"
  eksctl create cluster -f cvs-rendered.yaml --timeout "$EKSCTL_CREATE_TIMEOUT"
fi

# Function to check cluster status
check_cluster() {
  eksctl get cluster --name $CLUSTER_NAME --region $AWS_REGION | grep "ACTIVE"
}

# Poll until cluster is ready
echo "Waiting for cluster to become active..."
while ! check_cluster; do
  sleep 30
  echo "Still waiting..."
done

# Update kubeconfig to point to the new cluster
aws eks update-kubeconfig --name $CLUSTER_NAME --region $AWS_REGION
