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

source ./configuration.sh

if [ ! -d "$HOME/.aws/" ]; then
  echo "No .aws directory found in \$HOME, looking good!!!";
else
  echo "DELETE $HOME/.aws config folder before proceeding"
  exit 1
fi

aws eks update-kubeconfig --name $CLUSTER_NAME --region $AWS_REGION

### Now we need to create an s3 bucket and upload tiny-imagenet to it.
eksctl utils associate-iam-oidc-provider --region=$AWS_REGION --cluster=$CLUSTER_NAME

SERVICE_ACCOUNT_NAME="s3-access-sa"
NAMESPACE="default"
POLICY_NAME="EKS-S3AccessPolicy-$CLUSTER_NAME"

# Attempt to create the S3 bucket and handle if it already exists
echo "Creating the S3 bucket..."

# If bucket is already created, aws s3api will set an error code which could make the script stop exit early.
# If bucket is already created, we should still go ahead with rest of the script.
set +e
if [ "$AWS_REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket $S3_BUCKET_NAME --region $AWS_REGION --output json
else
  aws s3api create-bucket --bucket $S3_BUCKET_NAME --region $AWS_REGION \
    --create-bucket-configuration LocationConstraint=$AWS_REGION --output json
fi
EXIT_STATUS=$?
set -e

if [ $EXIT_STATUS -eq 0 ]; then
  echo "S3 bucket created successfully."
elif [ $EXIT_STATUS -eq 254 ]; then
  # Error code 255 generally means that there was an error in the AWS CLI command.
  # We need to check the error message to determine if it's because the bucket already exists.
  if aws s3api head-bucket --bucket $S3_BUCKET_NAME 2>/dev/null; then
    echo "Bucket '$S3_BUCKET_NAME' already exists and is accessible. Proceeding..."
  else
    echo "Bucket '$S3_BUCKET_NAME' already exists but is not accessible by you. Exiting."
    exit 1
  fi
else
  echo "An unexpected error occurred with exit code $EXIT_STATUS. Exiting."
  exit $EXIT_STATUS
fi

# Check if OIDC provider exists
OIDC_PROVIDER_URL=$(aws eks describe-cluster \
  --region $AWS_REGION \
  --name $CLUSTER_NAME \
  --query "cluster.identity.oidc.issuer" \
  --output text)

OIDC_ID=$(echo $OIDC_PROVIDER_URL | awk -F'/' '{print $NF}')
EXISTING_OIDC_PROVIDER=$(aws iam list-open-id-connect-providers \
  --query "OpenIDConnectProviderList[?ends_with(Arn, '/id/$OIDC_ID')]" \
  --output text)

if [ -n "$EXISTING_OIDC_PROVIDER" ]; then
  echo "OIDC provider already exists: $EXISTING_OIDC_PROVIDER"
else
  echo "No existing OIDC provider found. Associating a new OIDC provider..."
  eksctl utils associate-iam-oidc-provider \
    --region $AWS_REGION \
    --cluster $CLUSTER_NAME \
    --approve
fi

# Define the S3 access policy
POLICY_DOCUMENT=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$S3_BUCKET_NAME",
        "arn:aws:s3:::$S3_BUCKET_NAME/*"
      ]
    }
  ]
}
EOF
)

# Create the IAM policy and extract its ARN
echo "Creating IAM policy for S3 access..."
EXISTING_POLICY_ARN=$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='$POLICY_NAME'].Arn" --output text)
if [ "$EXISTING_POLICY_ARN" != "None" ] && [ -n "$EXISTING_POLICY_ARN" ]; then
  echo "Detaching entities attached to the policy: $EXISTING_POLICY_ARN"

  # Detach the policy from roles
  ROLE_ATTACHMENTS=$(aws iam list-entities-for-policy --policy-arn "$EXISTING_POLICY_ARN" --query "PolicyRoles[].RoleName" --output text)
  for ROLE in $ROLE_ATTACHMENTS; do
    echo "Detaching policy from role: $ROLE"
    aws iam detach-role-policy --role-name "$ROLE" --policy-arn "$EXISTING_POLICY_ARN"
  done
  echo "Deleting existing policy: $EXISTING_POLICY_ARN"
  echo "If this fails - you have a preexisting policy that this script has just detached roles from."
  aws iam delete-policy --policy-arn "$EXISTING_POLICY_ARN"
  echo "Success! Your policy should be correct."
fi

POLICY_ARN=$(aws iam create-policy \
  --policy-name "$POLICY_NAME" \
  --policy-document "$POLICY_DOCUMENT" \
  --query 'Policy.Arn' \
  --output text)

if [ -z "$POLICY_ARN" ]; then
  echo "Failed to create IAM policy. Exiting."
  exit 1
fi

echo "IAM policy created with ARN: $POLICY_ARN"

# Extract the IAM role name from the output
ROLE_NAME=$(aws iam list-roles --query "Roles[?contains(RoleName, 'eksctl-$CLUSTER_NAME-cluster-ServiceRole')].RoleName" --output text)
if [ -z "$ROLE_NAME" ]; then
  echo "Failed to capture the IAM role name. Exiting."
  exit 1
fi

echo "IAM role associated with the service account: $ROLE_NAME"
eksctl delete iamserviceaccount --name $SERVICE_ACCOUNT_NAME --namespace $NAMESPACE --cluster $CLUSTER_NAME --wait
# Create the serviceaccount
eksctl create iamserviceaccount \
  --name $SERVICE_ACCOUNT_NAME \
  --namespace $NAMESPACE \
  --cluster $CLUSTER_NAME \
  --attach-policy-arn $POLICY_ARN \
  --approve

# Attach the IAM policy to the IAM role
#echo "Attaching IAM policy to the IAM role..."
#aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn $POLICY_ARN

ROLE_ARN=$(aws iam list-roles --query "Roles[?contains(RoleName, 'eksctl-$CLUSTER_NAME-addon-iamserviceaccount-default')].Arn" --output text)

# Output success message
echo "Service Account:"
kubectl describe serviceaccount $SERVICE_ACCOUNT_NAME
echo "Role policies:"
aws iam list-attached-role-policies --role-name $ROLE_NAME
echo "S3 bucket '$S3_BUCKET_NAME' created and configured for EKS cluster '$CLUSTER_NAME' in region '$AWS_REGION'."
echo "Service account '$SERVICE_ACCOUNT_NAME' created in namespace '$NAMESPACE' with access to the bucket."
