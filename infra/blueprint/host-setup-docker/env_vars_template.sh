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

# CDS Deployment Environment Variables Template
# NOTE: Remove 'export' keyword when using with docker --env-file
# For bash sourcing, keep 'export' keywords

# ============================================
# NGC and Docker Credentials
# ============================================
# Get NGC API key from: https://org.ngc.nvidia.com/setup/api-key
NGC_API_KEY=<your-ngc-api-key>

# Docker Hub credentials
# Get PAT from: https://app.docker.com/settings/personal-access-tokens
DOCKER_USER=<your-dockerhub-username>
DOCKER_PAT=<your-dockerhub-personal-access-token>

# ============================================
# Deployment Configuration
# ============================================
# Cluster name (max 20 characters, alphanumeric and hyphens only)
CLUSTER_NAME=<your-cluster-name>

# AWS Region (e.g., us-east-2, us-west-2)
AWS_REGION=us-east-2

# S3 Bucket name (must be globally unique, lowercase)
S3_BUCKET_NAME=<your-unique-bucket-name>

# ============================================
# AWS Credentials
# ============================================
# Use ONE of the following credential types:

# Option A: Permanent Credentials (Recommended for Development)
# - Access Key ID starts with AKIA
# - Leave AWS_SESSION_TOKEN empty
AWS_ACCESS_KEY_ID=<AKIA...>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
AWS_SESSION_TOKEN=

# Option B: Temporary Session Credentials (SSO/STS)
# - Access Key ID starts with ASIA
# - Uncomment and set all three variables
# AWS_ACCESS_KEY_ID=<ASIA...>
# AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
# AWS_SESSION_TOKEN=<your-session-token>

# ============================================
# Custom S3 Credentials (For Data Ingestion)
# ============================================
# Optional: Use if ingesting from a different S3 bucket
# If not set, use the main AWS credentials above instead
CUSTOM_AWS_ACCESS_KEY_ID=<>
CUSTOM_AWS_SECRET_ACCESS_KEY=<>
CUSTOM_AWS_REGION=<>
CUSTOM_S3_BUCKET_NAME=<bucket-name>
CUSTOM_S3_FOLDER=<folder-name-in-bucket>