# Docker Compose Deployment Guide

This guide walks through deploying CDS using Docker Compose for local development and testing.

## Prerequisites

Before proceeding, ensure you have completed the [Docker Compose Prerequisites](docker-compose-prerequisites.md), including:
- Docker and NVIDIA Container Toolkit installed
- NGC authentication configured
- Environment variables set or persisted
- LocalStack hostname mapping added to `/etc/hosts`

## Overview

Docker Compose deployment provides a complete CDS stack running on a single node. This deployment method is ideal for:
- Local development and testing
- Evaluating CDS capabilities
- Small-scale deployments
- Prototyping and experimentation

The deployment includes the Visual Search API, the source-built CE1 OSS PyTorch
service, Milvus vector database, and LocalStack S3 storage.

**Time Estimate**: First deployment takes 15-30 minutes after the model weights
are available locally. Subsequent deployments take 2-5 minutes.

## Deployment Steps

### Step 1: Clone the Repository

Clone the CDS repository and navigate to the project directory:

```bash
# Clone the repository
git clone https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search
cd cosmos-dataset-search

# Verify you're in the correct directory
ls -la
# You should see Makefile, pyproject.toml, and other project files
```

### Step 2: Review Milvus Configuration (Optional)

CDS includes multiple Milvus configuration files:

- **Default (Recommended)**: `deploy/standalone/milvus_localstack.yaml` - Optimized for CVDS with LocalStack
- **Official Reference**: `deploy/standalone/milvus_official_2_4_4.yaml` - Included Milvus 2.4.4 reference configuration; use a version-compatible configuration for your deployed Milvus image.

**Key optimizations in `milvus_localstack.yaml` (default):**
- GPU memory configuration for shared GPU with cosmos-embed
- Storage V2 enabled with full configuration for better reliability
- LocalStack S3 endpoints pre-configured
- Automatic startup ordering for reliable milvus startup(cosmos-embed → milvus)

**To switch to official Milvus config:**

Edit `deploy/standalone/docker-compose.build.yml`:

```yaml
milvus:
  volumes:
    # Option 1: Use LocalStack-optimized config (default, recommended)
    - ./milvus_localstack.yaml:/milvus/configs/milvus.yaml

    # Option 2: Use official Milvus config (requires manual LocalStack setup)
    # - ./milvus_official_2_4_4.yaml:/milvus/configs/milvus.yaml
```

**If using `milvus_official_2_4_4.yaml`, you MUST manually configure**:
1. S3/LocalStack settings (`minio.address: localstack`, `minio.port: 4566`)
2. GPU memory allocation (`gpu.initMemSize`, `gpu.maxMemSize`)
3. Storage scheme (`common.storage.scheme: s3`)

**To use a custom config:**
1. Create a new copy and update S3 settings to point to LocalStack (see existing `milvus_localstack.yaml` or `milvus_official_2_4_4.yaml`).
2. Set GPU memory allocation based on your hardware

For GPU memory configuration details, see [GPU Memory Management Guide](../guides/gpu-memory-management.md).

### Step 3: Configure Environment Variables

CDS uses a `.env` file to manage all required environment variables. Start by copying the provided template:

```bash
# Copy the environment template
cp deploy/standalone/.env.example deploy/standalone/.env
```

Edit the `.env` file and update the following required variables:

```bash
# Edit the .env file with your preferred editor
nano deploy/standalone/.env
# or
vi deploy/standalone/.env
```

**Required Variables to Update:**

1. **`NVIDIA_API_KEY`** - Your NGC API key (obtained from [NGC](https://org.ngc.nvidia.com/setup/api-key))
   ```bash
   NVIDIA_API_KEY=<your-NGC-API-key>
   ```

2. **`DATA_DIR`** - Path to your data directory (e.g., `$HOME/cds-data`)
   ```bash
   DATA_DIR=/path/to/your/data
   ```

   **Important**: This directory must exist before proceeding. Create it if it doesn't exist:
   ```bash
   mkdir -p $HOME/cds-data
   ```

   This folder is used to enable faster LocalStack uploads by providing direct file system access to the containerized S3 service.

**Default Variables (typically no changes needed for local deployment):**
- `AWS_ACCESS_KEY_ID=test` (for LocalStack)
- `AWS_SECRET_ACCESS_KEY=test` (for LocalStack)
- `AWS_ENDPOINT_URL=http://localstack:4566`
- `COSMOS_EMBED_URI=http://cosmos-embed:8000`
- `GUNICORN_PORT=8888`

**Trusted local HTTP media servers:**

CE1 accepts remote media over HTTPS by default. To ingest from a trusted plain
HTTP server, add its exact origin (scheme, host, and port, without a path) to
`COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HTTP_ORIGINS`. For a server listening on
port 8680 of the Docker host, use URLs beginning with
`http://host.docker.internal:8680` and configure:

```bash
COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HTTP_ORIGINS=http://host.docker.internal:8680
```

Separate multiple origins with commas. Do not add public HTTPS origins to this
setting. Run `make test-integration-up` after changing `.env` so Compose
recreates the CE1 service with the new value.

For a complete list of configuration options, see the comments in the `.env` file.

### Step 4: Install Dependencies

Install Python dependencies and set up the development environment:

```bash
make install
```

This command:
- Creates a Python virtual environment using UV
- Installs all Python dependencies with GPU support
- Generates protobuf definitions for service communication
- Sets up the complete development environment

**Expected output**: Should complete without errors. Look for messages about dependency installation and protobuf generation.

### Step 5: Download CE1 OSS Model Weights

The default deployment is offline at runtime and expects the
`nvidia/Cosmos-Embed1-224p` files at `COSMOS_EMBED_MODEL_HOST_PATH`, which
defaults to `$HOME/.cache/cosmos-embed1`. Download the model snapshot configured
by `COSMOS_EMBED_HF_MODEL_ID` and `COSMOS_EMBED_HF_REVISION`:

```bash
make download-ce1-model
```

The public model snapshot does not require `hf auth login`. The command creates
the model directory and makes it readable by container UID 999. It does not add
weights to the source tree or download them when the service starts.

`make build-docker` runs the same check automatically. If the configured model
directory already contains a complete CE1 snapshot, it skips the download.

Before executing any model or processor code, CE1 verifies the snapshot's Python, configuration, tokenizer and weight files against the trusted `src/cosmos_embed_oss/model_manifest.json` packaged with the application. Missing, altered or additional loadable files prevent readiness with a model-integrity error. This check also covers reused snapshots and Kubernetes PVC mounts; startup includes reading all weight shards once.

The supported snapshot is `nvidia/Cosmos-Embed1-224p` at `787e0b996f5260a71ad474a283c90539a2e12986`. Opt-in online loading downloads that exact revision, verifies it, then loads both model and processor locally. A different model revision requires review and an application manifest update, not only an environment-variable change. There is no verification-disable switch. Keep model files mounted read-only and protect the host and writable HF module cache from untrusted writers; startup verification does not protect against a compromised host changing files afterward.

The model mount is read-only; the Hugging Face module cache is a separate writable volume. Keep the one-shot `cosmos-embed-cache-init` dependency enabled so the non-root CE1 service can write its cache on a fresh deployment. For private storage or custom S3 endpoints, configure the [ingestion source settings](import-url-security.md).

### Step 6: Validate Environment Configuration

Before proceeding, validate your environment configuration:

```bash
# Run the validation script
bash deploy/standalone/scripts/validate_env.sh
```

This script:
- Verifies all required variables are set
- Checks that API keys are not placeholder values
- Validates port numbers, URIs, and other settings
- Provides helpful error messages if issues are found

**Expected output**: Should complete with "Environment validation completed successfully."

If validation fails, review the error messages and update your `.env` file accordingly, then run the validation script again.

### Step 7: Build Docker Images

Build all required Docker images:

```bash
make build-docker
```

This command builds:
- Python base image with CUDA support
- CE1 OSS PyTorch service image
- Visual Search service image

Before building, it downloads the configured CE1 snapshot only when the model
directory is missing or incomplete. The model remains a runtime bind mount and
is not copied into the CE1 image.

**Time Estimate**: 10-20 minutes on first build. Subsequent builds use Docker cache and are much faster (1-2 minutes).

### Step 8: Launch Services

Start the complete CDS stack:

```bash
make test-integration-up
```

This command:
- Starts Milvus vector database with etcd
- Launches the CE1 OSS PyTorch service with the local model files
- Starts Visual Search API service
- Launches LocalStack S3-compatible storage
- Waits for all services to become healthy

**Important Notes**:
- **Model files**: The service does not download model weights at runtime. The
  host directory configured by `COSMOS_EMBED_MODEL_HOST_PATH` must exist and
  be readable by UID 999.
- **GPU check**: The CE1 OSS service requires GPU access. If it fails to start,
  verify your GPU setup with `nvidia-smi`.
- **Monitor progress**: Watch the logs to track service startup.

The expected results from a successful deployment should match

```bash
Starting services...
cd deploy/standalone && docker compose -f docker-compose.build.yml up -d
[+] Running 1/1
 ✔ validate-env Pulled                                                                                                                     1.5s
[+] Running 7/7
 ✔ Container milvus-etcd                Healthy                                                                                            1.2s
 ✔ Container localstack                 Healthy                                                                                            1.2s
 ✔ Container milvus                     Healthy                                                                                            4.9s
 ✔ Container standalone-validate-env-1  Exited                                                                                             0.7s
 ✔ Container standalone-cosmos-embed-cache-init-1 Exited                                                                                   0.7s
 ✔ Container cosmos-embed               Healthy                                                                                           64.2s
 ✔ Container visual-search              Started                                                                                            0.2s

----------------------------------------
Waiting for services to be ready...
This may take a few minutes for GPU services to initialize...
python scripts/wait_for_services.py
INFO:__main__:Waiting for services: milvus, cosmos-embed, visual-search
INFO:__main__:Checking milvus at http://localhost:9091/healthz
INFO:__main__:milvus is ready
INFO:__main__:Checking cosmos-embed at http://localhost:9000/v1/health/ready
INFO:__main__:cosmos-embed is ready
INFO:__main__:Checking visual-search at http://localhost:8888/health
INFO:__main__:visual-search is ready
INFO:__main__:All services are ready!
INFO:__main__:Service check completed in 18.24s
INFO:__main__:All services are ready for testing!

```
You can monitor the startup process:

```bash
# View logs in real-time
make test-integration-logs

# Or view specific service logs
docker compose -f deploy/standalone/docker-compose.build.yml logs -f cosmos-embed
```

### Verify Deployment (Optional)

Once all services are running, you can optionally verify each component service manually:

```bash
# Check service health
curl http://localhost:8888/health

# Expected response: "OK"

# Check the CE1 OSS embedding service
curl http://localhost:9000/v1/health/ready

# Expected response: {"status":"ready"}

# List available embedding pipelines
curl http://localhost:8888/v1/pipelines

# Expected response: JSON list of available pipelines

```

All health checks should return successful responses. If any service fails, see the [Troubleshooting section](#troubleshooting).

### Install and Configure CDS CLI

Install the CDS command-line interface for data ingestion and management.

**Note**: This step is required for the commands in the [Testing the Deployment](#testing-the-deployment) section below.

```bash
# Install CDS CLI
make install-cds-cli

# Activate the virtual environment
source .venv/bin/activate

# Verify CLI installation
cds --help

# Configure CLI to connect to local deployment
cds config set
# When prompted, enter: http://localhost:8888

# Verify configuration
cds pipelines list
```


## Accessing the Services

### API Documentation

Interactive API documentation (Swagger UI) is available at:

```
http://localhost:8888/v1/docs
```

This provides complete API reference with interactive request testing. For detailed API usage, see the [API Reference](api_reference.md).

## Testing the Deployment

### Quick Verification Test

Run the built-in integration tests to verify all services are functioning correctly:

```bash
make test-integration-run
```

This runs:
- Minimal integration test validating service communication
- Cosmos video end-to-end test verifying embedding generation

**Expected result**: All tests should pass. This confirms the deployment is working correctly.

### Ingest Sample Data

Test data ingestion with a small sample dataset:

```bash
# Ingest 100 videos from MSR-VTT dataset for testing
make ingest-msrvtt-small
```

**Note**: This command ingests a small subset of videos for quick testing. The full ingestion process:
1. Downloads videos from the dataset
2. Uploads to LocalStack S3 storage
3. Processes videos through the embedding pipeline
4. Indexes embeddings in Milvus

**Time Estimate**: 5-10 minutes for 100 videos, depending on your system.

**Alternative**: For a more comprehensive test with the full dataset:
```bash
make ingest-msrvtt
```

### Verify Data Ingestion

After ingestion completes, verify the data:

```bash
# List collections (should show the newly created collection)
curl http://localhost:8888/v1/collections

# Or use the CLI
cds collections list

# Perform a test search
cds search --collection-ids <collection-id> \
  --text-query "person walking" \
  --top-k 5
```

## Managing the Deployment

### Important Note on Data Persistence

**All data ingested into CDS is ephemeral and will be lost when services are stopped and restarted.** This includes all collections, ingested videos, and embeddings. You will need to re-ingest your data after restarting the services.

### Stopping and Restarting Services

Stop all running services:

```bash
make test-integration-down
```

Restart services:

```bash
make test-integration-up
```

**Note**: After restarting, all previous data will be gone. You must re-ingest your datasets.

### Clean Restart (Rebuild Images)

To stop services and also remove Docker images (forcing a rebuild on next startup):

```bash
# Stop services and remove images
make test-integration-clean

# Restart and rebuild
make test-integration-up
```

Use `make test-integration-clean` when you want to clear cached images and ensure a fresh build on the next startup.

### Advanced Management

For detailed service management, including viewing logs, restarting individual services, and advanced configuration options, see the [CDS User Guide](user-guide.md).

## Service Endpoints Reference

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Visual Search API | http://localhost:8888 | REST API for search operations |
| API Documentation | http://localhost:8888/v1/docs | Interactive API docs (Swagger) |
| CE1 OSS | http://localhost:9000 | Embedding service |
| Milvus Database | localhost:19530 | Vector database (internal) |
| LocalStack S3 | http://localhost:4566 | S3-compatible storage (internal) |

## Troubleshooting

If you encounter issues during deployment or operation, see the [Docker Compose Troubleshooting Guide](troubleshooting-docker-compose.md) for detailed solutions to common problems.

## Next Steps

After successfully deploying CDS, proceed to:

1. **[CDS User Guide](user-guide.md)** - Learn how to interact with CDS

   The user guide covers CLI and REST API usage for this release.
