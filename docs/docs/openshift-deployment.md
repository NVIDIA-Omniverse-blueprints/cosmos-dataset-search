# Deploying Cosmos Dataset Search on Red Hat OpenShift AI

## What We're Deploying

The Cosmos Dataset Search (CDS) Blueprint ingests video datasets, generates embeddings
using an NVIDIA NIM embedding model, and makes them searchable via natural-language queries
through a React web UI.

| Component | Image | GPU | Purpose |
|-----------|-------|-----|---------|
| **cosmos-embed** (NIM Operator) | `nvcr.io/nim/nvidia/cosmos-embed1:1.0.0` | 1 | Video/image embedding via NIMService |
| **visual-search** | `nvcr.io/nvidia/blueprint/cosmos-dataset-search:1.0.0` | 0 | Search API service |
| **visual-search-react-ui** | NVCF subchart | 0 | React web UI |
| **nginx-proxy** | `nginxinc/nginx-unprivileged:1.25.3-alpine` | 0 | Reverse proxy (Route entry point) |
| **milvus** | NVCF subchart (9 pods) | 1 (indexNode) | Vector database (GPU-accelerated indexing) |
| **minio** | NVCF subchart | 0 | S3-compatible object storage |
| **etcd** | NVCF subchart | 0 | Milvus metadata store |
| **kafka + zookeeper** | NVCF subchart | 0 | Milvus message queue |

**Data flow:**

- **Ingest:** videos (S3) → visual-search → cosmos-embed (embeddings) → Milvus
- **Search:** user query → visual-search → cosmos-embed (query embedding) → Milvus (vector search) → results

**Total:** 2 GPUs minimum (cosmos-embed + milvus indexNode).

### NIM Operator Components

When deployed with the NIM Operator (recommended on OpenShift), the cosmos-embed NIM
is managed as a pair of custom resources instead of a standalone Deployment:

- **NIMCache** — downloads and caches the model on a PVC. Annotated with
  `helm.sh/resource-policy: keep` to survive upgrades and uninstalls.
- **NIMService** — runs the inference server, references its NIMCache for model storage,
  manages replicas, GPU allocation, and health probes.

## Tested Hardware

| Parameter | Value |
|-----------|-------|
| Platform | Red Hat OpenShift 4.14+ |
| GPU nodes | 1+ nodes with NVIDIA L40S / A100 40GB / H100 |
| GPUs per node | 1+ |
| Total GPUs | 2 (cosmos-embed + milvus indexNode) |
| VRAM | 16 GB+ per GPU (24 GB+ recommended) |
| CPU | 8+ cores |
| RAM | 32 GB+ |
| Storage | 50 GiB dynamically provisioned PVC (cosmos-embed model cache) + 10 GiB disk for cache/decode buffers |
| API keys | [NGC API key](https://org.ngc.nvidia.com/setup/api-keys) |

Minimum for reproduction: 2 x NVIDIA L40S / A100 / H100 GPUs (16 GB+ VRAM each), 50 GiB storage.

## What's Different from Upstream

| Area | Upstream Default | OpenShift Deployment | Impact |
|------|------------------|----------------------|--------|
| Pod management | Subchart Deployments | NIM Operator (NIMCache + NIMService) | Automated model download, cache lifecycle |
| External access | `kubectl port-forward` | OpenShift Route with TLS + nginx proxy | Production-grade ingress |
| Volume provisioning | `emptyDir` (ephemeral) | Dynamic PVCs via NIM Operator | Model cache persists across restarts |
| Security context | Pods run as specific UIDs (e.g. 1000) | Custom SCC + RoleBinding | Compatible with OpenShift SCC |
| Secrets | Manual creation | Helm-managed via `openshift.secrets` | Single `helm install` creates all secrets |
| Object storage | External AWS S3 | Built-in MinIO (AWS S3 optional) | No external dependency by default |

## Deployment Files

All OpenShift customizations are codified in the `infra/blueprint/bringup/` directory alongside the upstream chart:

- **`values-openshift.yaml`** — Helm values overlay for OpenShift. Contains all OpenShift-specific overrides (NIM Operator, security contexts, tolerations, secrets, storage).
- **`visual-search/templates/openshift.yaml`** — Helm template gated by `openshift.enabled`. Creates custom SCC, RoleBinding, Route, nginx proxy, and secrets.
- **`visual-search/templates/nim-cosmos-embed.yaml`** — NIMCache + NIMService template for cosmos-embed, gated by the NIM Operator CRD (`apps.nvidia.com/v1alpha1`).
- **`visual-search/values.yaml`** — Base chart values with `openshift.enabled: false` and `nimOperator.cosmosEmbed.enabled: false` defaults.

## Prerequisites

### CLI Tools

- `oc` (OpenShift CLI) logged into your cluster
- `helm` v3.12+
- `aws` CLI (for data ingestion with MinIO/S3)
- `python3` and `make` (for CDS CLI and dataset preparation)

### Cluster Requirements

- Red Hat OpenShift 4.14+
- NVIDIA GPU Operator installed and configured
- NIM Operator installed (provides `apps.nvidia.com/v1alpha1` API).
  See [NIM Operator installation guide](https://docs.nvidia.com/nim-operator/latest/install.html)
- At least 1 GPU node available

### Verify GPU Availability

```bash
oc get nodes -l nvidia.com/gpu.present=true
oc describe node <gpu-node> | grep -A 5 "Allocatable"
```

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NGC_API_KEY` | Yes | — | NGC API key for image pulls and model downloads |

### OpenShift Block (`openshift:`)

| Key | Default | Description |
|-----|---------|-------------|
| `openshift.enabled` | `false` | Master toggle for all OpenShift resources |
| `openshift.route.enabled` | `false` | Create an OpenShift Route for the UI |
| `openshift.scc.create` | `false` | Create custom SCC + RoleBinding |
| `openshift.secrets.create` | `false` | Create NGC and encryption secrets from values |
| `openshift.nginx.enabled` | `false` | Deploy nginx reverse proxy for path-based routing |
| `openshift.minio.route.enabled` | `false` | Create a Route for MinIO (data ingestion) |

### NIM Operator Block (`nimOperator:`)

The cosmos-embed NIM has the following configuration:

| Key | Default | Description |
|-----|---------|-------------|
| `nimOperator.cosmosEmbed.enabled` | `false` | Deploy cosmos-embed via NIM Operator |
| `nimOperator.cosmosEmbed.replicas` | `1` | Number of inference replicas |
| `nimOperator.cosmosEmbed.service.name` | `cosmos-embed-nim` | Kubernetes service name (used for DNS) |
| `nimOperator.cosmosEmbed.image.repository` | `nvcr.io/nim/nvidia/cosmos-embed1` | NGC container image |
| `nimOperator.cosmosEmbed.image.tag` | `1.0.0` | Image version |
| `nimOperator.cosmosEmbed.resources.limits.nvidia.com/gpu` | `1` | GPU allocation |
| `nimOperator.cosmosEmbed.storage.pvc.size` | `50Gi` | Model cache PVC size |
| `nimOperator.cosmosEmbed.storage.pvc.storageClass` | `""` | StorageClass (uses cluster default if empty) |
| `nimOperator.cosmosEmbed.secrets.pullSecret` | `ngc-secret` | Image pull secret name |
| `nimOperator.cosmosEmbed.secrets.authSecret` | `ngc-api` | NGC API secret name |
| `nimOperator.cosmosEmbed.expose.service.port` | `8000` | Service port |
| `nimOperator.cosmosEmbed.tolerations` | `[]` | Node tolerations for GPU taints |
| `nimOperator.cosmosEmbed.startupProbe` | (see values) | Startup probe configuration |

## Deployment

### 1. Create Namespace

```bash
oc new-project cosmos-cds
```

### 2. Create Secrets

The chart can create secrets automatically via `openshift.secrets.create: true` —
just pass the NGC API key via `--set` at install time.
Alternatively, create all secrets manually before install (recommended for production).

> **Helm auto-creation:** If using auto-creation, skip this step and pass keys via
> `--set` flags in step 3. If you pre-create secrets here, you **must** disable the
> chart's secret creation to prevent it from overwriting your secrets with empty
> defaults. In `values-openshift.yaml`, set:
> ```yaml
> visual-search:
>   openshift:
>     secrets:
>       create: false
> ```
> Also omit the `--set` flags in step 3.

```bash
export NGC_API_KEY="<your-ngc-api-key>"
export CUSTOM_NAMESPACE=cosmos-cds
export RELEASE_NAME=cosmos-cds

# Image pull secret — ngc-secret (for pulling NIM containers from nvcr.io)
oc create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  -n $CUSTOM_NAMESPACE

oc label secret ngc-secret \
  app.kubernetes.io/managed-by=Helm -n $CUSTOM_NAMESPACE
oc annotate secret ngc-secret \
  meta.helm.sh/release-name=$RELEASE_NAME \
  meta.helm.sh/release-namespace=$CUSTOM_NAMESPACE -n $CUSTOM_NAMESPACE

# Image pull secret — nvcr-io (used by subchart image pulls)
oc create secret docker-registry nvcr-io \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  -n $CUSTOM_NAMESPACE

oc label secret nvcr-io \
  app.kubernetes.io/managed-by=Helm -n $CUSTOM_NAMESPACE
oc annotate secret nvcr-io \
  meta.helm.sh/release-name=$RELEASE_NAME \
  meta.helm.sh/release-namespace=$CUSTOM_NAMESPACE -n $CUSTOM_NAMESPACE

# NGC API secret (for NIM Operator model downloads)
oc create secret generic ngc-api \
  --from-literal=NGC_API_KEY="${NGC_API_KEY}" \
  -n $CUSTOM_NAMESPACE

oc label secret ngc-api \
  app.kubernetes.io/managed-by=Helm -n $CUSTOM_NAMESPACE
oc annotate secret ngc-api \
  meta.helm.sh/release-name=$RELEASE_NAME \
  meta.helm.sh/release-namespace=$CUSTOM_NAMESPACE -n $CUSTOM_NAMESPACE

# Encryption key secret (used by the visual-search API)
export SECRETS_ENCRYPTION_KEY=$(python3 infra/blueprint/bringup/generate_secret_key.py | grep "export" | cut -d"'" -f2)
oc create secret generic secret-encryption-key \
  --from-literal=SECRETS_ENCRYPTION_KEY="$SECRETS_ENCRYPTION_KEY" \
  -n $CUSTOM_NAMESPACE

oc label secret secret-encryption-key \
  app.kubernetes.io/managed-by=Helm -n $CUSTOM_NAMESPACE
oc annotate secret secret-encryption-key \
  meta.helm.sh/release-name=$RELEASE_NAME \
  meta.helm.sh/release-namespace=$CUSTOM_NAMESPACE -n $CUSTOM_NAMESPACE
```

Link the pull secret to the NIM Operator service account:

```bash
oc create sa nim-cache-sa -n $CUSTOM_NAMESPACE || true
oc secrets link nim-cache-sa ngc-secret --for=pull -n $CUSTOM_NAMESPACE
```

### 3. Install the Chart

```bash
helm dependency build infra/blueprint/bringup

helm upgrade --install $RELEASE_NAME infra/blueprint/bringup \
  -n $CUSTOM_NAMESPACE \
  -f infra/blueprint/bringup/values-openshift.yaml \
  --set "visual-search.openshift.secrets.ngcApiKey=$NGC_API_KEY" \
  --timeout 45m
```

> If you pre-created secrets in step 2, omit the `--set` flags above.

This creates:
- 1 x NIMCache (triggers cosmos-embed model download)
- 1 x NIMService (runs inference after cache is ready)
- 1 x OpenShift Route with TLS (external UI access via nginx proxy)
- 1 x Custom SCC + RoleBinding (allows subchart pods to run as image UIDs)
- All required secrets (NGC registry, NGC API key, encryption key)
- Visual Search API, React UI, nginx proxy
- Milvus vector database with MinIO, etcd, Kafka, ZooKeeper
- Subchart cosmos-embed Deployment is disabled (`cosmos-embed.enabled: false`)

### 4. Monitor Model Downloads

NIMCache resources download models from NGC. This can take 10-20 minutes depending
on model size and network speed.

```bash
oc get nimcache -n $CUSTOM_NAMESPACE -w
```

Wait until the cache shows `Ready`:

```
NAME                      STATUS   AGE
cosmos-embed-nim-cache    Ready    15m
```

## Verification

### Check NIMService Status

```bash
oc get nimservice -n $CUSTOM_NAMESPACE
```

### Check Pods

```bash
oc get pods -n $CUSTOM_NAMESPACE
```

All pods should reach `Running 1/1`.

**Expected pods** (pod names are prefixed with your release name):

| Pod | Purpose | Managed By |
|-----|---------|------------|
| `cosmos-embed-nim` | Video embedding NIM (GPU) | NIM Operator |
| `visual-search` | Search API service | Helm |
| `visual-search-react-ui` | Web UI | Helm |
| `nginx-proxy` | Reverse proxy (Route entry point) | Helm |
| `<release>-milvus-*` | Vector database (9 pods) | Helm |
| `<release>-minio` | S3-compatible object storage | Helm |
| `<release>-etcd` | Milvus metadata store | Helm |
| `<release>-kafka` | Milvus message queue | Helm |
| `<release>-zookeeper` | Kafka coordination | Helm |

### Health Endpoints

```bash
echo -n "cosmos-embed: "
oc exec -n $CUSTOM_NAMESPACE deployment/cosmos-embed-nim -- \
  curl -s http://localhost:8000/v1/health/ready
echo
```

## Accessing the UI

The Helm chart creates an OpenShift Route with TLS edge termination. Get the URL:

```bash
export CUSTOM_API_ENDPOINT=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
echo "https://$CUSTOM_API_ENDPOINT/cosmos-dataset-search"
```

Open `https://<route-host>/cosmos-dataset-search` in a browser.

## Install CDS CLI

> **Note**: `CUSTOM_API_ENDPOINT` must be exported (set in Verification step) before running this script.

```bash
cd infra/blueprint/bringup
bash client_up.sh
```

**Duration**: 2-3 minutes

**What happens**:
- Installs CDS CLI from packaged source
- Creates Python virtual environment at the project root (`.venv/`)
- Configures CLI with the OpenShift Route endpoint
- Validates deployment by listing pipelines

**Verify CLI works**:

```bash
cd /path/to/cosmos-dataset-search
source .venv/bin/activate
cds pipelines list
cds collections list
```

## Data Ingestion

### Setup: Source Environment Variables

CDS works with any S3-compatible storage. Configure the `CUSTOM_*` variables for your storage provider:

**Option A: Built-in MinIO** (deployed with the chart):

```bash
cd /path/to/cosmos-dataset-search
source .venv/bin/activate

MINIO_ROUTE=$(oc get route minio -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')

export CUSTOM_AWS_ACCESS_KEY_ID=minioadmin
export CUSTOM_AWS_SECRET_ACCESS_KEY=minioadmin
export CUSTOM_AWS_REGION=us-east-1
export CUSTOM_S3_BUCKET_NAME=cosmos-videos
export CUSTOM_S3_FOLDER=msrvtt-videos
export CUSTOM_ENDPOINT_URL="https://$MINIO_ROUTE"
export CUSTOM_API_ENDPOINT=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
```

**Option B: AWS S3** (or any external S3):

```bash
cd /path/to/cosmos-dataset-search
source .venv/bin/activate

export CUSTOM_AWS_ACCESS_KEY_ID=<your-access-key>
export CUSTOM_AWS_SECRET_ACCESS_KEY=<your-secret-key>
export CUSTOM_AWS_REGION=<your-region>
export CUSTOM_S3_BUCKET_NAME=<your-bucket>
export CUSTOM_S3_FOLDER=<your-folder>
# CUSTOM_ENDPOINT_URL is not needed for AWS S3
export CUSTOM_API_ENDPOINT=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
```

> **Note**: Set `CUSTOM_ENDPOINT_URL` only for non-AWS S3 storage (MinIO, ODF/Noobaa, etc.). AWS S3 uses region-based endpoints automatically.

### Step 1: Prepare Dataset

Download and prepare the MSR-VTT dataset. The sample config downloads 100 videos.

```bash
uv pip install datasets
make prepare-dataset CONFIG=scripts/msrvtt_test_100.yaml
```

**Duration**: 5-10 minutes

**What happens**:
- Downloads MSRVTT_Videos.zip from HuggingFace
- Extracts and copies 100 videos to `~/datasets/msrvtt/videos/`

> **Note**: The `prepare-dataset` script only supports HuggingFace datasets that provide videos as a downloadable ZIP file. For datasets without ZIPs, you'll need to download videos separately or provide them locally.

### Step 2: Upload Videos to S3

```bash
export PROFILE_NAME="${CUSTOM_S3_BUCKET_NAME}-profile"

aws configure set aws_access_key_id "${CUSTOM_AWS_ACCESS_KEY_ID}" --profile "${PROFILE_NAME}"
aws configure set aws_secret_access_key "${CUSTOM_AWS_SECRET_ACCESS_KEY}" --profile "${PROFILE_NAME}"
aws configure set region "${CUSTOM_AWS_REGION}" --profile "${PROFILE_NAME}"
# Set endpoint URL for MinIO (skip this line for AWS S3)
aws configure set endpoint_url "${CUSTOM_ENDPOINT_URL}" --profile "${PROFILE_NAME}"

# Check if bucket exists; create it if not
aws s3 ls s3://$CUSTOM_S3_BUCKET_NAME --profile $PROFILE_NAME || \
  aws --profile $PROFILE_NAME s3 mb s3://$CUSTOM_S3_BUCKET_NAME

# Upload all videos from local dataset directory to MinIO bucket
aws --profile $PROFILE_NAME s3 cp ~/datasets/msrvtt/videos/ \
  s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ --recursive

# Verify upload
echo "Videos uploaded. Checking count:"
aws --profile $PROFILE_NAME s3 ls s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ | wc -l
```

**Duration**: 1-2 minutes

### Step 3: Ingest Videos

The `ingest_custom_videos.sh` script **requires CUSTOM_*** variables to be set. These variables specify which S3 bucket has the videos to ingest.

```bash
cd infra/blueprint
bash ingest_custom_videos.sh
```

**What the script does with CUSTOM_* variables:**
- Creates AWS profile using CUSTOM_AWS_ACCESS_KEY_ID and CUSTOM_AWS_SECRET_ACCESS_KEY
- Gets deployment URL from CUSTOM_API_ENDPOINT (set in the setup step)
- Creates Kubernetes secret with S3 credentials
- Creates collection with storage configuration
- Ingests videos from s3://CUSTOM_S3_BUCKET_NAME/CUSTOM_S3_FOLDER/

**Duration**: 2-5 minutes for 5 videos (default limit)

**Expected output:**

```
Processed files: 5/5 ━━━━ 100%
Status code 200: 5/5 100%
Processed 5 files successfully
```

### Step 4: Configure CORS (External S3 Only)

If using the built-in MinIO, skip this step — no CORS needed (same-origin).

If using external AWS S3, **you must configure CORS on your S3 buckets**. Without it, the web UI cannot load video assets from S3.

> **Note**: Requires `CUSTOM_API_ENDPOINT` and `CUSTOM_S3_BUCKET_NAME` to be set (from the Setup step above).

```bash
cd infra/blueprint/bringup
bash verify_and_configure_cors.sh -y
```

**What the script does:**
- Uses `CUSTOM_API_ENDPOINT` (set in the setup step) as the allowed CORS origin
- Checks IAM permissions for GetBucketCors and PutBucketCors
- Applies CORS policy to all buckets relevant to your deployment

**Example browser error if CORS is missing**:
```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource...
(Reason: CORS header 'Access-Control-Allow-Origin' missing)
```

> **Security tip:** The script uses your Route URL as the allowed origin. Do **not** use `*` for production. For manual CORS configuration, see [Manually Configure CORS for S3 Videos](#manually-configure-cors-for-s3-videos).

### Step 5: Verify Ingestion

```bash
cds collections list

ROUTE_HOST=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
echo "https://$ROUTE_HOST/cosmos-dataset-search"
```

Open the Web UI and search for videos (e.g., "a person playing basketball").

### Ingest Your Own Videos

1. Upload videos:

```bash
aws s3 cp /path/to/your/videos/ s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ \
  --recursive --profile $PROFILE_NAME
```

2. Update `CUSTOM_*` variables if using a different bucket or credentials, then run:

```bash
cd infra/blueprint
bash ingest_custom_videos.sh
```

## Advanced Options

### Using AWS S3 Instead of MinIO

By default, the OpenShift overlay deploys a built-in MinIO pod for Milvus object storage. If you prefer to use AWS S3, override the storage settings at install time:

```bash
helm upgrade --install $RELEASE_NAME infra/blueprint/bringup \
  -n $CUSTOM_NAMESPACE \
  -f infra/blueprint/bringup/values-openshift.yaml \
  --set "visual-search.openshift.secrets.ngcApiKey=$NGC_API_KEY" \
  --set "milvus.minio.enabled=false" \
  --set "milvus.externalS3.enabled=true" \
  --set "milvus.externalS3.host=s3.$AWS_REGION.amazonaws.com" \
  --set "milvus.externalS3.bucketName=$S3_BUCKET_NAME" \
  --set "milvus.externalS3.region=$AWS_REGION" \
  --set "milvus.externalS3.useIAM=false" \
  --set "milvus.externalS3.accessKey=$AWS_ACCESS_KEY_ID" \
  --set "milvus.externalS3.secretKey=$AWS_SECRET_ACCESS_KEY" \
  --timeout 45m
```

> **Note**: Unlike EKS which uses IAM Roles for Service Accounts (IRSA) for keyless S3 access, OpenShift requires explicit AWS credentials via `accessKey` and `secretKey`.

### Managing Secrets

> **Important**: Storage secrets must be created in the `default` namespace due to a hardcoded namespace in the visual-search application ([k8s_secrets.py](../../src/visual_search/v1/apis/k8s_secrets.py)).

```bash
# Create
oc create secret generic my-s3-creds -n default \
  --from-literal=aws_access_key_id=$CUSTOM_AWS_ACCESS_KEY_ID \
  --from-literal=aws_secret_access_key=$CUSTOM_AWS_SECRET_ACCESS_KEY \
  --from-literal=aws_region=$CUSTOM_AWS_REGION

# Use in collection
cds collections create --pipeline cosmos_video_search_milvus \
  --name "My Collection" \
  --config-yaml <(echo "
tags:
  storage-template: 's3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/{{filename}}'
  storage-secrets: 'my-s3-creds'
")

# List / Delete
oc get secrets -n default
oc delete secret my-s3-creds -n default
```

### Manually Configure CORS for S3 Videos

This section applies only when using external AWS S3. If using the built-in MinIO, CORS is not required.

#### Why CORS is Required

When your browser tries to load videos from S3, it makes cross-origin requests. S3 blocks these requests by default unless you explicitly configure CORS rules.

**Error you'll see without CORS**:
```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource...
(Reason: CORS header 'Access-Control-Allow-Origin' missing). Status code: 200.
```

#### Required IAM Permissions

Your AWS credentials must have these permissions on the S3 bucket:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutBucketCORS",
    "s3:GetBucketCORS"
  ],
  "Resource": "arn:aws:s3:::<your-bucket-name>"
}
```

If you see an "AccessDenied" error, contact your AWS administrator to grant these permissions.

#### Configure CORS

```bash
ROUTE_HOST=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')

echo "Configuring CORS for bucket: $CUSTOM_S3_BUCKET_NAME"
echo "Allowing origin: https://$ROUTE_HOST"

aws s3api put-bucket-cors \
  --bucket $CUSTOM_S3_BUCKET_NAME \
  --region $CUSTOM_AWS_REGION \
  --profile $PROFILE_NAME \
  --cors-configuration "{
    \"CORSRules\": [{
      \"AllowedOrigins\": [\"https://${ROUTE_HOST}\"],
      \"AllowedMethods\": [\"GET\", \"HEAD\"],
      \"AllowedHeaders\": [\"*\"],
      \"ExposeHeaders\": [\"ETag\", \"Content-Length\", \"Content-Type\"],
      \"MaxAgeSeconds\": 3600
    }]
  }"

# Verify
aws s3api get-bucket-cors --bucket $CUSTOM_S3_BUCKET_NAME --region $CUSTOM_AWS_REGION --profile $PROFILE_NAME
```

**Alternative: Wildcard CORS (less secure, allows any origin)**:

```bash
aws s3api put-bucket-cors \
  --bucket $CUSTOM_S3_BUCKET_NAME \
  --region $CUSTOM_AWS_REGION \
  --profile $PROFILE_NAME \
  --cors-configuration '{
    "CORSRules": [{
      "AllowedOrigins": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
      "MaxAgeSeconds": 3600
    }]
  }'
```

#### Test CORS Configuration

After configuring CORS, test that videos load in the web UI:

1. Open the web UI in your browser
2. Perform a search
3. Click on a video result
4. The video should load and play without errors

If you still see CORS errors, verify:
- CORS configuration is applied: `aws s3api get-bucket-cors --bucket <bucket-name> --region <region>`
- Your Route hostname matches the allowed origin in CORS rules
- You've configured CORS on all buckets that store videos

## OpenShift-Specific Challenges and Solutions

### 1. Security Context Constraints

**What:** The Milvus MinIO container runs as a specific UID by default. OpenShift's
`restricted-v2` SCC blocks this by forcing a random UID.

**Error:** `container has runAsNonRoot and image has non-numeric user`

**Services affected:** milvus-minio.

**Fix:** The custom SCC (`<release>-nim`) created by the Helm chart when
`openshift.scc.create: true` sets `runAsUser: RunAsAny` and
`seLinuxContext: RunAsAny`. A RoleBinding grants this SCC to the application
service accounts.

### 2. GPU Node Tolerations

**What:** GPU nodes carry `NoSchedule` taints. Without matching tolerations, pods stay
`Pending`.

**Error:** `0/N nodes are available: N node(s) had untolerated taint`

**Services affected:** cosmos-embed-nim, milvus indexNode.

**Fix:** Add tolerations in `values-openshift.yaml` for each GPU service:

```yaml
visual-search:
  nimOperator:
    cosmosEmbed:
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

Check your taints with `oc describe node <gpu-node> | grep Taints`.

### 3. Missing Secrets

**What:** The chart references secrets that don't exist by default. Pods fail with
`secret not found`.

**Error:** `secret "<name>" not found`

**Services affected:** All (shared secrets).

**Fix:** NGC secrets are created via `openshift.secrets.create: true` or manually
in step 2 of the deployment instructions.

### 4. NIMCache PVC Sizing

**What:** NIM model cache PVCs must be large enough to hold all downloaded model
profiles. Undersized PVCs cause download failures.

**Services affected:** cosmos-embed-nim-cache.

**Fix:** PVC size in `values-openshift.yaml`: cosmos-embed 50 GiB. NIMCache PVCs are
immutable — delete the NIMCache and PVC to resize, then re-run `helm install`.

### 5. Helm Secret Adoption

**What:** If you create secrets with `oc create secret` before `helm install`, Helm
refuses to adopt them during install or upgrade.

**Error:** `Error: rendered manifests contain a resource that already exists`

**Services affected:** All (shared secrets).

**Fix:** Pre-label secrets with Helm ownership metadata before install:

```bash
oc label secret <name> app.kubernetes.io/managed-by=Helm -n $CUSTOM_NAMESPACE
oc annotate secret <name> \
  meta.helm.sh/release-name=$RELEASE_NAME \
  meta.helm.sh/release-namespace=$CUSTOM_NAMESPACE -n $CUSTOM_NAMESPACE
```

## Known Limitations

1. **Storage secrets must be in the `default` namespace.** The visual-search application reads S3 credentials from the `default` namespace. The chart creates a RoleBinding granting the app's service accounts access to secrets in `default`. If you deploy to a different namespace, secrets must still be created in `default`.

2. **Resource sizing.** The `values-openshift.yaml` resource values match the production configuration. On smaller clusters, pods like `queryNode` (12 CPU / 96Gi) and `dataNode` (6 CPU / 24Gi) may stay Pending due to insufficient capacity. Reduce resource requests in `values-openshift.yaml` to fit your cluster.

## Cleanup

```bash
# Uninstall the Helm release
helm uninstall $RELEASE_NAME -n $CUSTOM_NAMESPACE

# NIMCache PVCs persist by default (helm.sh/resource-policy: keep).
# Delete manually if you want to reclaim storage:
oc delete nimcache --all -n $CUSTOM_NAMESPACE
oc delete pvc -l app.nvidia.com/nim-cache -n $CUSTOM_NAMESPACE

# Delete the project — removes all namespaced resources (pods, PVCs, services, etc.)
oc delete project $CUSTOM_NAMESPACE

# Clean up resources created in the default namespace
oc delete role ${CUSTOM_NAMESPACE}-secret-access-role -n default --ignore-not-found
oc delete rolebinding ${CUSTOM_NAMESPACE}-secret-access-binding -n default --ignore-not-found
oc delete secret ${CUSTOM_S3_BUCKET_NAME}-secrets-videos -n default --ignore-not-found
```
