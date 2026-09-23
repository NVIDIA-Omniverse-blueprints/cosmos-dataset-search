# AWS EKS Deployment Guide

This guide provides complete step-by-step instructions for deploying CDS (Cosmos Dataset Search) on Amazon Elastic Kubernetes Service (EKS).

## Overview

This deployment uses a pre-configured Docker container that includes all necessary tools (AWS CLI, eksctl, kubectl, helm) to deploy CDS on AWS EKS.

The release defaults use the following NGC images. See the [container guide](release-containers.md) for immutable digests and Compose usage.

| Service | Image |
| --- | --- |
| CDS API | `nvcr.io/nvidia/blueprint/cosmos-dataset-search:1.2.0` |
| CE1 OSS | `nvcr.io/nvidia/blueprint/cosmos-embed1-oss:1.2.0` |

The EKS nodes must be able to pull both images. Set `NGC_API_KEY` to a key with access to these NGC containers before running the deployment scripts. The scripts configure the existing `nvcr-io` image pull Secret. GitHub package credentials are not required.

**What gets deployed**:
- EKS cluster with GPU and CPU node groups
- S3 bucket for storage
- Milvus vector database
- CE1 OSS service for video embeddings
- Visual search service

**Total time**: ~30-40 minutes

## Prerequisites

### Storage Secret access

Before deploying, set `secretAccess.allowedNames` in `infra/blueprint/bringup/values.yaml` to the Kubernetes Secret names used by your `storage_secrets` requests, for example:

```yaml
secretAccess:
  allowedNames: [customer-s3-credentials]
```

The default empty list grants no Secret API access. The `s3-access-sa` Role permits only `get` on these names in `default`, never namespace-wide `list` or `watch`. Add new or renamed Secrets to this list and apply a Helm upgrade; otherwise requests needing them fail with a Kubernetes 403. Rotating values under an existing name does not require a Role change. IAM access and kubelet-injected `secretKeyRef` environment variables are unchanged. Review other RoleBindings separately: Kubernetes permissions are additive, so this chart cannot revoke grants from other Roles.

CE1 also verifies its mounted model files against the checksum manifest shipped with the application before loading. See the model integrity notes in the [Compose guide](docker-compose-deployment.md#step-5-download-ce1-oss-model-weights); the same check applies to PVC snapshots.

Before exposing this deployment, configure customer-managed authentication, authorization, trusted TLS and allowed browser origins at your gateway. The quickstart does not set a production-specific CORS policy. For private storage, see [ingestion source settings](import-url-security.md).

### AWS Account Requirements

- **IAM Permissions**: Ability to create EKS, EC2, S3, IAM, VPC, CloudFormation
- **Service Quotas**: Sufficient quotas for GPU instances (g6.xlarge)

### Required Credentials

1. **NGC API Key**
   - Get from: https://org.ngc.nvidia.com/setup/api-key
   - Required for NVIDIA container images

2. **Docker Hub Credentials**
   - Username and Personal Access Token
   - Get PAT from: https://app.docker.com/settings/personal-access-tokens

3. **AWS Credentials** (one of):
   - **Permanent**: `AWS_ACCESS_KEY_ID` (starts with AKIA) + `AWS_SECRET_ACCESS_KEY`
   - **Temporary**: Above + `AWS_SESSION_TOKEN`

4. **IAM Permissions**: Ability to create and manage:
  - EKS clusters
  - EC2 instances (including g6.xlarge GPU instances)
  - S3 buckets
  - IAM roles and policies
  - VPC resources
  - CloudFormation stacks

5. **Service Quotas**: Ensure you have sufficient quotas for:
  - EC2 instances (specifically g6.xlarge for GPU nodes)
  - EBS volumes
  - Elastic IPs
  - VPC resources

### Local Machine Requirements

- Docker Desktop or Docker Engine installed and running
- 10GB+ free disk space
- Stable internet connection

## Deployment Steps

### Step 1: Build Host Setup Container

Navigate to your CDS repository and build the deployment container:

```bash
cd /path/to/cds
make build-host-setup
```

**Expected output**: Container builds successfully (~1 minute with caching)

### Step 2: Configure Environment Variables

Copy the template and fill in your credentials. This template file is an important piece of reference for environment variables required and once we copy to **my-env.sh** in project root, we will refer to it throughout this guide.

```bash
cp infra/blueprint/host-setup-docker/env_vars_template.sh my-env.sh
nano my-env.sh  # or use your preferred editor
```

**Edit the file** with your actual values:

```bash
# NGC and Docker credentials
NGC_API_KEY=<your-ngc-api-key>
DOCKER_USER=<your-dockerhub-username>
DOCKER_PAT=<your-dockerhub-personal-access-token>

# Deployment configuration
CLUSTER_NAME=<your-cluster-name>  # Max 20 characters, alphanumeric and hyphens
AWS_REGION=us-east-2               # or your preferred region
S3_BUCKET_NAME=<unique-bucket-name>  # Must be globally unique

# AWS Credentials
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_SESSION_TOKEN=  # Leave empty if using permanent credentials

# Custom S3 Credentials (For Data Ingestion)
# You can reference the main credentials above, no need to change unless using different bucket with different credentials
CUSTOM_AWS_ACCESS_KEY_ID=<>
CUSTOM_AWS_SECRET_ACCESS_KEY=<>
CUSTOM_AWS_REGION=<>
CUSTOM_S3_BUCKET_NAME=<bucket-name>
CUSTOM_S3_FOLDER=<folder-name-in-bucket>
```

**Verify configuration**:
```bash
set -a && source my-env.sh && set +a
echo "Cluster: $CLUSTER_NAME"
echo "Region: $AWS_REGION"
```

> **"Environment variables are the backbone of your deployment. Set them carefully, and your journey will be smooth."**

**Important Note:**
To properly source the `my-env.sh` file and ensure all environment variables are exported, **always** use the following syntax:

```bash
set -a && source my-env.sh && set +a
```

This practice ensures that every variable in `my-env.sh` is exported into your shell environment and available to all subsequent commands. Omitting `set -a` / `set +a` may lead to subtle bugs where some variables are not exported.

### Step 3: Start Deployment Container

Run the **cds-deployment** container in detached mode, We will use this deployment container for rest of the guide whenever we reference to **cds-deployment**

```bash
docker run -it -d \
  --env-file my-env.sh \
  -v ~/.kube:/root/.kube \
  -v $(pwd)/infra/blueprint:/workspace/blueprint \
  --name cds-deployment \
  host-setup:latest \
  /bin/bash
```

**Verify AWS credentials work**:

```bash
docker exec cds-deployment bash -c "aws sts get-caller-identity"
```

You should see your AWS account ID and user ARN.

**Validate configuration**:

```bash
docker exec cds-deployment bash -c "cd /workspace/blueprint/bringup && ./configuration.sh"
```

Expected output shows all credentials validated with checkmarks (✓).

### Step 4: Create EKS Cluster

Create the cluster with all node groups:

```bash
docker exec cds-deployment bash -c "cd /workspace/blueprint/bringup && ./cluster_up.sh -y"
```

**Duration**: 15-20 minutes

**What happens**:
- Creates VPC and networking resources
- Provisions 4 node groups via CloudFormation:
  - 2x g6.xlarge (GPU nodes)
  - 1x r7i.4xlarge (high-memory for Milvus)
  - 5x m6i.2xlarge (general compute)
  - 1x c6i.2xlarge (optimized compute)
- Configures high-performance GP3 storage class
- Installs EBS CSI driver
- Installs NVIDIA device plugin

**Verify nodes are ready**:

```bash
docker exec cds-deployment bash -c "kubectl get nodes"
```

All 9 nodes should show "Ready" status.

**Troubleshooting**:
- If "Cluster already exists": Script skips creation and uses existing cluster
- If CloudFormation errors: Check AWS console for details
- If quota errors: Request service quota increases

### Step 5: Set Up S3 Bucket and Permissions

Create S3 bucket and configure IAM:

```bash
docker exec cds-deployment bash -c "cd /workspace/blueprint/bringup && ./s3_up.sh -y"
```

**Duration**: 2-3 minutes

**What happens**:
- Creates S3 bucket in your region
- Sets up OIDC provider for EKS
- Creates IAM policy for S3 access
- Creates Kubernetes service account with IAM role

**Verify**:

```bash
docker exec cds-deployment bash -c "kubectl describe serviceaccount s3-access-sa"
```

Should show IAM role ARN annotation.

### Step 6: Deploy Kubernetes Services

Provision the pinned CE1 model PVC before starting the service. Run this from your host; the variables are passed explicitly to this container command:

```bash
docker exec \
  -e COSMOS_EMBED_OSS_MODEL_PVC=ce1-oss-model \
  -e COSMOS_EMBED_OSS_MODEL_HF_REVISION=787e0b996f5260a71ad474a283c90539a2e12986 \
  -e COSMOS_EMBED_OSS_IMAGE_REPOSITORY=nvcr.io/nvidia/blueprint/cosmos-embed1-oss \
  -e COSMOS_EMBED_OSS_IMAGE_TAG=1.2.0 \
  cds-deployment bash -c \
  'cd /workspace/blueprint/bringup && ./seed_ce1_oss_model.sh && ./k8s_up.sh -y'
```

The seed step requires outbound access to Hugging Face and an `HF_TOKEN` if the model requires authentication. For offline installations, provision the same checksum-verified snapshot into the PVC using your own storage process. The CE1 service mounts the snapshot read-only and does not download model weights at runtime by default.

For later deployments after the PVC is seeded, pass its name again. Exports in an earlier `docker exec` shell do not persist into a new one:

```bash
docker exec -e COSMOS_EMBED_OSS_MODEL_PVC=ce1-oss-model \
  cds-deployment bash -c "cd /workspace/blueprint/bringup && ./k8s_up.sh -y"
```

**Duration**: 30-40 minutes total

**What gets deployed**:
- Kubernetes secrets for image pulling
- CE1 OSS service (GPU-accelerated embedding service)
- Milvus vector database (15 pods)
- Visual Search API service
- Nginx ingress controller with TLS

**Monitor progress** (in another terminal):

```bash
# Watch pods come up
docker exec cds-deployment bash -c "kubectl get pods -w"

# Check specific service
docker exec cds-deployment bash -c "kubectl get pods -l app.kubernetes.io/name=cosmos-embed"
```

**Key stages**:
1. Milvus components start (~5 min)
2. Visual Search starts (~2-3 min)
3. CE1 validates and loads the pre-provisioned model snapshot on the GPU
4. Ingress controller creates AWS load balancer (~2-3 min)

**Verify all pods are ready**:

```bash
docker exec cds-deployment bash -c "kubectl get pods"
```

All pods should show "Running" with "1/1" or "2/2" ready.

**Common issues**:

- **Cosmos-embed ContainerCreating for >5 minutes**: Check image pulling, PVC binding and GPU scheduling before inspecting model-load logs
- **Milvus-querynode pod pending**: Check if scheduled on r7i.4xlarge node
- **Image pull errors**: Verify that both NGC `1.2.0` tags are published and that the key in the `nvcr-io` image pull Secret can access them.

### Step 7: Get Deployment URLs

Get your ingress URL:

```bash
docker exec cds-deployment bash -c "kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
```

**Access your deployment**:
- API: `https://<hostname>/api`

**Test the API**:
```bash
# Get hostname
HOSTNAME=$(docker exec cds-deployment bash -c "kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'")

# Test health endpoint
curl -k https://$HOSTNAME/api/health

# Test pipelines endpoint
curl -k https://$HOSTNAME/api/v1/pipelines
```

### Step 8: Install and Configure CDS CLI on Your Local Machine

Install and configure the CDS CLI on your local machine:

```bash
# Navigate to CDS repository on your local machine
cd /path/to/cds

# Run the client setup script from bringup directory
cd infra/blueprint/bringup
bash client_up.sh -y
```

**Duration**: 2-3 minutes

**What happens**:
- Installs CDS CLI from packaged source
- Creates Python virtual environment with dependencies
- Automatically configures CLI with your deployment's API endpoint
- Validates deployment by listing pipelines

**Expected output**:
```
Installing CDS CLI from source...
CDS CLI installed successfully!
CDS CLI version 1.2.0
Configuring CDS CLI...
Pipelines:
{
  "pipelines": [
    {
      "id": "cosmos_video_search_milvus",
      "enabled": true,
      "missing": []
    }
  ]
}
CDS blueprint running at <cluster-name>. Installation complete.
```

**Verify CLI works**:

```bash
# Activate the virtual environment
source /path/to/cds/.venv/bin/activate

# List pipelines
cds pipelines list

# List collections (will be empty initially)
cds collections list
```

**If you see duplicate section error**:

```bash
# Edit the config file if you have duplicates in [default] profile
nano ~/.config/cds/config # or on your preferred editor

# Should have only one [default] section with your API endpoint
```

## Data Ingestion

After deployment, you can ingest videos into CDS. This guide shows how to ingest the MSR-VTT sample dataset.

### Setup: Source Environment Variables

Before starting data ingestion, source your environment variables once:

```bash
# Navigate to CDS repository
cd /path/to/cds

# Source environment variables (sets all AWS credentials and S3 bucket info)
set -a && source my-env.sh && set +a

# Activate virtual environment
source .venv/bin/activate
```

**This makes all variables available** for the remaining steps (AWS credentials, S3 bucket, CUSTOM_* variables).

### Step 1: Prepare Dataset

Download and prepare the MSR-VTT dataset using the provided configuration. Using the sample config in scripts has **max_records** of 100. So 100 videos will be downloaded.

```bash
# Download and prepare videos
make prepare-dataset CONFIG=scripts/msrvtt_test_100.yaml
```

The `scripts/msrvtt_test_100.yaml` configuration contains the following:
```yaml
source: hf
hf_repo: friedrichor/MSR-VTT
hf_config: test_1k
split: test
video_zip_filename: MSRVTT_Videos.zip
max_records: 100
id_field: video_id
video_field: video
text_field: caption
output_jsonl: ~/datasets/msrvtt_test_100.jsonl
copy_videos_to: ~/datasets/msrvtt/videos
```

**Duration**: 5-10 minutes (downloads ~22MB of videos)

**What happens**:
- Downloads MSRVTT_Videos.zip from HuggingFace
- Extracts videos to /tmp/msr_vtt_videos
- Copies 100 videos to ~/datasets/msrvtt/videos/

**Note**: The `prepare-dataset` script only supports HuggingFace datasets that provide videos as a downloadable ZIP file. For datasets without ZIPs, you'll need to download videos separately or provide them locally.

### Step 2: Upload Videos to S3

This step configures an AWS profile using CUSTOM_* credentials and uploads videos to the S3 bucket.

**IMPORTANT - You MUST configure CUSTOM_* variables in `my-env.sh`:**

**Scenario 1: Same bucket as deployment** (most common):
```bash
# In my-env.sh, set CUSTOM_* to reference main credentials:
CUSTOM_AWS_ACCESS_KEY_ID=... # Same as AWS_ACCESS_KEY_ID
CUSTOM_AWS_SECRET_ACCESS_KEY=... # Same as AWS_SECRET_ACCESS_KEY
CUSTOM_AWS_REGION=... # Same as AWS_REGION
CUSTOM_S3_BUCKET_NAME=...  # Same bucket configured
CUSTOM_S3_FOLDER=msrvtt-videos  # Folder where you uploaded videos
```

**Scenario 2: Different dataset bucket with different credentials**:
```bash
# In my-env.sh, set CUSTOM_* to different values:
CUSTOM_AWS_ACCESS_KEY_ID=AKIA...  # Different AWS key
CUSTOM_AWS_SECRET_ACCESS_KEY=...  # Different secret
CUSTOM_AWS_REGION=us-west-2  # Different region
CUSTOM_S3_BUCKET_NAME=my-other-bucket  # Different bucket
CUSTOM_S3_FOLDER=videos  # Folder in that bucket
```

**After configuring CUSTOM_* variables, re-source the environment**

```bash
set -a && source my-env.sh && set +a
source .venv/bin/activate
```
**Prerequisite: Install and Configure AWS CLI**

Make sure you have the AWS CLI installed on your machine. If not, install it by following [AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Verify your installation:

```bash
aws --version
```

The following steps *require* that your `aws` CLI is available and on your PATH.

**AWS CLI Credentials:**
The profile you create below using `aws configure set ... --profile ...` works locally and does not touch your default AWS credentials. If you have not previously used the AWS CLI, you may need to run:

```bash
aws configure
```

...to set up your default credentials, or proceed directly with the per-profile configuration shown below. All `aws` commands that follow use the profile **derived from your CUSTOM_* variables**.


**Configure AWS profile for S3 access** (using CUSTOM_* variables):

```bash
# Create AWS profile for the S3 bucket
export PROFILE_NAME="${CUSTOM_S3_BUCKET_NAME}-profile"

aws configure set aws_access_key_id "${CUSTOM_AWS_ACCESS_KEY_ID}" --profile "${PROFILE_NAME}"
aws configure set aws_secret_access_key "${CUSTOM_AWS_SECRET_ACCESS_KEY}" --profile "${PROFILE_NAME}"
aws configure set region "${CUSTOM_AWS_REGION}" --profile "${PROFILE_NAME}"
```

**Verify the S3 bucket exists**:

```bash
# Check if bucket exists
aws s3 ls s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER --profile $PROFILE_NAME
```

**Note:If bucket doesn't exist, create it and set the environment variables for bucket**

**Upload videos to S3**:

```bash
# Upload videos to the bucket - Adjust where the videos are saved
aws s3 cp ~/datasets/msrvtt/videos/ s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ --recursive --profile $PROFILE_NAME

# Verify upload
echo "Videos uploaded. Checking count:"
aws s3 ls s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ --profile $PROFILE_NAME | wc -l
```

**Duration**: 1-2 minutes

### Step 3: Ingest Videos into CDS

The `ingest_custom_videos.sh` script **requires CUSTOM_*** variables to be set. These variables specify which S3 bucket has the videos to ingest.

**Run ingestion**:

```bash
cd infra/blueprint
bash ingest_custom_videos.sh
```

**What the script does with CUSTOM_* variables:**
- Creates AWS profile using CUSTOM_AWS_ACCESS_KEY_ID and CUSTOM_AWS_SECRET_ACCESS_KEY
- Verifies and creates Kubernetes secret with these credentials
- Uses CUSTOM_S3_BUCKET_NAME and CUSTOM_S3_FOLDER to locate videos
- Ingests from s3://CUSTOM_S3_BUCKET_NAME/CUSTOM_S3_FOLDER/

**Duration**: 2-5 minutes for 5 videos (default limit)

**What happens**:
- Creates AWS profile for S3 access
- Gets deployment URL from ingress
- Creates Kubernetes secret with S3 credentials
- Creates collection with storage configuration
- Ingests videos from S3 into CDS

**Expected output**:
```
Status code 200: 5/5 100%
Processed 5 files successfully
```

### Step 4: Verify Ingestion

```bash
# List collections
cds collections list
```

### Ingest Your Own Videos

To ingest your own videos:

1. **Upload videos to S3**:
   ```bash
   aws s3 cp /path/to/your/videos/ s3://$CUSTOM_S3_BUCKET_NAME/CUSTOM_S3_FOLDER/ --recursive
   ```

2. **Update folder name** in `my-env.sh`:
   ```bash
      # Edit my-env.sh and change:
      CUSTOM_AWS_ACCESS_KEY_ID=<>
      CUSTOM_AWS_SECRET_ACCESS_KEY=<>
      CUSTOM_AWS_REGION=<>
      CUSTOM_S3_BUCKET_NAME=<>
      CUSTOM_S3_FOLDER=my-videos

      # Then re-source:
      set -a && source my-env.sh && set +a
   ```

3. **Run ingestion**:
   ```bash
      cd infra/blueprint
      bash ingest_custom_videos.sh
   ```

**Note**: The `ingest_custom_videos.sh` script creates a collection with proper S3 storage configuration, creates Kubernetes secrets for S3 access, and then ingests the videos. This is the recommended workflow for AWS EKS deployments.

Before using that script, include its generated Secret name, `${CUSTOM_S3_BUCKET_NAME}-secrets-videos`, in `secretAccess.allowedNames` and apply the visual-search Helm values. Creating a Secret does not grant the application permission to read it.

## Advanced Options and Configurations

### Managing Secrets

CDS uses Kubernetes secrets to store sensitive credentials (like S3 access keys) that collections need to access videos in different buckets.

#### Create a Secret for S3 Access

```bash
docker exec cds-deployment kubectl create secret generic my-s3-creds \
  --from-literal=aws_access_key_id=... \
  --from-literal=aws_secret_access_key=secret... \
  --from-literal=aws_region=us-east-2
```

#### List Secrets

This is an operator command using the deployment administrator's credentials. The CDS service account intentionally cannot list Secrets.

```bash
docker exec cds-deployment kubectl get secrets
```

#### Use Secret in Collection

Add `my-s3-creds` to `secretAccess.allowedNames` and apply the visual-search Helm values first. Only the named `get` permission is needed; do not restore namespace-wide `list` or `watch` access.

When creating a collection that references videos in S3, specify the secret in the collection configuration:

```bash
cds collections create --pipeline cosmos_video_search_milvus \
  --name "My Collection" \
  --config-yaml <(echo "
tags:
  storage-template: 's3://my-bucket/videos/{{filename}}'
  storage-secrets: 'my-s3-creds'
")
```

For more details on using secrets with collections, see the [CLI User Guide](cli-user-guide.md#managing-secrets).

## Monitoring and Debugging

### Check Pod Status

```bash
docker exec cds-deployment bash -c "kubectl get pods"
```

### View Logs

```bash
# Visual search logs
docker exec cds-deployment bash -c "kubectl logs deployment/visual-search --tail=100"

# Cosmos-embed logs
docker exec cds-deployment bash -c "kubectl logs deployment/cosmos-embed --tail=100"

# Milvus query node logs
docker exec cds-deployment bash -c "kubectl logs deployment/milvus-querynode --tail=100"
```

### Check Resources

```bash
# Node status
docker exec cds-deployment bash -c "kubectl get nodes"

# Services and ingress
docker exec cds-deployment bash -c "kubectl get svc,ingress"

# Persistent volumes
docker exec cds-deployment bash -c "kubectl get pv,pvc"
```

## Troubleshooting

### Pods Not Starting

**Check pod details**:
```bash
docker exec cds-deployment bash -c "kubectl describe pod <pod-name>"
```

**Common causes**:
- Insufficient resources: Check node capacity
- Image pull errors: Verify NGC_API_KEY
- Volume mounting issues: Check PVC status

### Cosmos-embed Issues

**If pod is pending**:
```bash
docker exec cds-deployment bash -c "kubectl describe pod -l app.kubernetes.io/name=cosmos-embed"
```

Check for:
- GPU nodes available: `kubectl get nodes -l role=cvs-gpu`
- GPU resources: One GPU per pod required
- Proper scheduling: Should be on cvs-gpu labeled nodes

**If the pod stays in `ContainerCreating`**:
- Inspect pod events for image-pull, volume-mount or scheduling errors.
- Confirm `ce1-oss-model` is Bound and the seed job completed successfully.
- Model downloads occur in the seed step, not in the default CE1 runtime. Once the container starts, check its logs for snapshot verification or GPU initialization errors.

### Ingress Not Accessible

**Check ingress status**:
```bash
docker exec cds-deployment bash -c "kubectl get ingress simple-ingress"
```

**Check load balancer**:
```bash
docker exec cds-deployment bash -c "kubectl get svc -n ingress-nginx"
```

**Note**: AWS ALB takes 2-3 minutes to become accessible after creation.

### Service Health Checks

```bash
# Get ingress hostname
docker exec cds-deployment bash -c "kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"

# Test API (replace <hostname> with actual hostname)
curl -k https://<hostname>/api/health
```

## Cleanup

### Complete Removal

When you're done, delete all resources:

```bash
docker exec cds-deployment bash -c "cd /workspace/blueprint/teardown && ./shutdown_sequence.sh -y"
```

**Duration**: 10-15 minutes

**What gets deleted**:
- All Kubernetes resources
- EKS cluster and node groups
- S3 bucket (optional - you'll be prompted)
- IAM roles and policies
- CloudFormation stacks

### Verify Cleanup

```bash
# Check cluster is gone
docker exec cds-deployment bash -c "eksctl get cluster --name \$CLUSTER_NAME"
# Should return: No clusters found
```

## Quick Reference

### Essential Commands

```bash
# Check all pods
docker exec cds-deployment bash -c "kubectl get pods"

# Check nodes
docker exec cds-deployment bash -c "kubectl get nodes"

# Get ingress URL
docker exec cds-deployment bash -c "kubectl get ingress simple-ingress"

# View logs
docker exec cds-deployment bash -c "kubectl logs <pod-name> --tail=100"

# Describe problematic pod
docker exec cds-deployment bash -c "kubectl describe pod <pod-name>"

```

### Deployment Checklist

- [ ] AWS account with required permissions
- [ ] NGC API key obtained
- [ ] Docker Hub credentials ready
- [ ] Docker installed and running
- [ ] Host-setup container built (`make build-host-setup`)
- [ ] Environment variables configured in my-env.sh
- [ ] Deployment container started
- [ ] AWS credentials validated
- [ ] EKS cluster created (`cluster_up.sh -y`)
- [ ] S3 bucket configured (`s3_up.sh -y`)
- [ ] Kubernetes services deployed (`k8s_up.sh -y`)
- [ ] All 17 pods in Running status
- [ ] Deployment URLs obtained
- [ ] CDS CLI installed on local machine
- [ ] CLI configured and tested

## Summary

**Total deployment time**: ~30-40 minutes
**Total pods deployed**: 17

**Next steps**: Install CDS CLI locally, configure with your API endpoint, and start ingesting videos.

For troubleshooting, check pod logs and events using the commands in the Quick Reference section above.
