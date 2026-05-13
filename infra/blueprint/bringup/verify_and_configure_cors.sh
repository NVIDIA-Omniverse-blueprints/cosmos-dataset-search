#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

## CORS Verification and Configuration Script for CDS S3 Buckets
## Usage: ./verify_and_configure_cors.sh [-y|--apply]
## Set CUSTOM_API_ENDPOINT env var to skip EKS auto-detection (e.g. OpenShift Route)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null || exit
trap 'popd > /dev/null' EXIT

# Parse arguments
APPLY_CORS=false
if [[ "$1" == "-y" || "$1" == "--apply" ]]; then
    APPLY_CORS=true
fi

# Source EKS configuration only when CUSTOM_API_ENDPOINT is not provided
if [ -z "${CUSTOM_API_ENDPOINT:-}" ]; then
    source ./configuration.sh
fi

echo "=============================================="
echo "  S3 CORS Verification"
echo "=============================================="

# Get hostname: use CUSTOM_API_ENDPOINT if set, otherwise auto-detect from ingress
if [ -n "${CUSTOM_API_ENDPOINT:-}" ]; then
    VS_API="$CUSTOM_API_ENDPOINT"
else
    VS_API=$(kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
fi

# Determine CORS origin based on ingress availability
if [ -n "$VS_API" ]; then
    CORS_ORIGIN="https://$VS_API"
    CORS_MODE="secure (ingress hostname)"
else
    CORS_ORIGIN="*"
    CORS_MODE="wildcard (ingress not available)"
fi

# Check function for a bucket
check_bucket() {
    local bucket=$1
    local region=$2
    local profile=$3
    local bucket_label=$4
    
    echo ""
    echo "[$bucket_label] $bucket"
    
    # Build profile flag
    local profile_flag=""
    if [ -n "$profile" ]; then
        profile_flag="--profile $profile"
    fi
    
    # Check CORS
    if cors_config=$(aws s3api get-bucket-cors --bucket "$bucket" --region "$region" $profile_flag 2>&1); then
        echo "  CORS: ✓ Configured"
        echo "$cors_config" | jq -C '.' 2>/dev/null || echo "$cors_config"
        return 0
    elif echo "$cors_config" | grep -q "NoSuchCORSConfiguration"; then
        echo "  CORS: ✗ Not configured"
        
        # Apply CORS if flag is set
        if [ "$APPLY_CORS" = true ]; then
            echo "  Applying CORS origin: $CORS_ORIGIN"
            
            aws s3api put-bucket-cors --bucket "$bucket" --region "$region" $profile_flag \
                --cors-configuration '{
                    "CORSRules": [{
                        "AllowedOrigins": ["'"$CORS_ORIGIN"'"],
                        "AllowedMethods": ["GET", "HEAD"],
                        "AllowedHeaders": ["*"],
                        "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
                        "MaxAgeSeconds": 3600
                    }]
                }' && echo "  CORS: ✓ Applied successfully"
        else
            echo "  Run with -y flag to apply CORS (will use: $CORS_ORIGIN)"
        fi
        return 1
    elif echo "$cors_config" | grep -q "AccessDenied"; then
        echo "  CORS: ✗ Access Denied (missing s3:GetBucketCors permission)"
        return 2
    else
        echo "  CORS: ? Error checking CORS"
        return 2
    fi
}

# Check main bucket (only when S3_BUCKET_NAME is set, e.g. EKS)
MAIN_STATUS=0
if [ -n "${S3_BUCKET_NAME:-}" ]; then
    check_bucket "$S3_BUCKET_NAME" "$AWS_REGION" "" "Main Bucket"
    MAIN_STATUS=$?
fi

# Check custom bucket if set and different from main
if [ -n "${CUSTOM_S3_BUCKET_NAME:-}" ] && [ "${CUSTOM_S3_BUCKET_NAME:-}" != "${S3_BUCKET_NAME:-}" ]; then
    # Set up AWS profile for custom bucket
    PROFILE_NAME="${CUSTOM_S3_BUCKET_NAME}-profile"
    aws configure set aws_access_key_id "$CUSTOM_AWS_ACCESS_KEY_ID" --profile "$PROFILE_NAME" 2>/dev/null
    aws configure set aws_secret_access_key "$CUSTOM_AWS_SECRET_ACCESS_KEY" --profile "$PROFILE_NAME" 2>/dev/null
    aws configure set region "$CUSTOM_AWS_REGION" --profile "$PROFILE_NAME" 2>/dev/null
    
    check_bucket "$CUSTOM_S3_BUCKET_NAME" "$CUSTOM_AWS_REGION" "$PROFILE_NAME" "Custom Bucket"
    CUSTOM_STATUS=$?
fi

echo ""
echo "=============================================="
if [ -n "$VS_API" ]; then
    echo "Endpoint: https://$VS_API"
else
    echo "Endpoint: Not available"
fi
echo "CORS origin: $CORS_ORIGIN ($CORS_MODE)"

if [ "$APPLY_CORS" = true ]; then
    echo "Mode: Auto-apply"
else
    echo "Mode: Verify only (use -y to apply)"
fi
echo "=============================================="

