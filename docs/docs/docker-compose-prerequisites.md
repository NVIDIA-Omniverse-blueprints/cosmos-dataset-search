# Docker Compose Deployment Prerequisites

This guide covers the prerequisites for deploying CDS using Docker Compose. This deployment method is best for local development, testing, and small-scale deployments.

## System Requirements

### Hardware Requirements

#### Minimum Requirements

- **CPU**: 8+ cores
- **RAM**: 32GB system memory
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **Storage**: 100GB+ available disk space

#### Recommended Requirements

- **CPU**: 16+ cores
- **RAM**: 64GB system memory
- **GPU**: NVIDIA GPU with 24GB+ VRAM (L4, A100, or H100)
- **Storage**: 200GB+ high-performance SSD storage

### GPU Requirements

CDS requires an NVIDIA GPU for running the CE1 OSS service. Supported GPUs:

| GPU                                      | GPU Memory | Support Level |
|------------------------------------------|------------|---------------|
| H100                                     | 80GB       | Preferred     |
| A100, L40s, L4, H20, L20                 | 24GB+      | Recommended   |
| Other Ampere+ GPUs                       | 16GB+      | Functional    |

**Support Level Definitions:**
- **Preferred**: Best tested capacity for the PyTorch backend
- **Recommended**: Suitable for normal local development and evaluation
- **Functional**: Runs end-to-end; lower throughput may be expected

**CE1 OSS Service Requirements:**
- GPU Memory: Minimum 16GB; 24GB+ recommended for optimal performance
- CUDA: Compatible with CUDA 11.8+ runtime
- Model files: Public `nvidia/Cosmos-Embed1-224p` snapshot downloaded to a local directory before startup

### Software Requirements

#### Operating System

CDS has been tested on the following operating systems:

- **Ubuntu 22.04 LTS**
- **Ubuntu 24.04 LTS**

#### Required Software

| Package                                                        | Version     | Purpose                          |
|----------------------------------------------------------------|-------------|----------------------------------|
| [Docker](https://www.docker.com/)                              | 20.10+      | Container runtime                |
| [Docker Compose](https://docs.docker.com/compose/)             | 2.0+        | Multi-container orchestration    |
| [Python](https://www.python.org/)                              | 3.10        | Development and CLI tools        |
| [Git LFS](https://git-lfs.com/)                                | 3.0+        | Model weights and large files    |
| [UV](https://github.com/astral-sh/uv)                          | 0.8.17+     | Python dependency management     |
| [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx)   | 525+        | GPU driver (CUDA 11.8+ support)  |

#### Required Access

- NGC registry access for the other CVDS service images

## Pre-Installation Setup

### Install Docker and Docker Compose

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

For other platforms, see [Docker installation guide](https://docs.docker.com/engine/install/).

### Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker containers to access your GPU.

**Ubuntu/Debian:**
```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

For detailed instructions, see [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Install Python and UV Package Manager

```bash
# Install Python 3.10 (if not already installed)
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installations
python --version
# Should show Python 3.10.x

# Note: On older distributions, python may not be linked to python3
# If the above command fails, create an alias:
# sudo apt-get install -y python-is-python3
# Or manually create a symlink:
# sudo ln -s /usr/bin/python3 /usr/bin/python

# uv will require a shell restart or sourcing the following file. 
source $HOME/.local/bin/env
uv --version
```

### Install Git LFS

Git LFS is required for downloading model weights and large files.

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# Verify installation
git lfs version
```

## NGC Configuration

Access to NGC (NVIDIA GPU Cloud) is required for the remaining CVDS service images. The CE1 OSS image is built from this repository, and its model files are prepared separately.

### Create NGC Account and API Key

1. Create an account at [NGC](https://ngc.nvidia.com/)
2. Generate an [API Key](https://org.ngc.nvidia.com/setup/api-key)
3. Ensure your NGC account can authenticate to the `nvcr.io` container registry.

### Authenticate Docker with NGC

```bash
docker login nvcr.io
Username: $oauthtoken
Password: <your-NGC-API-key>
```

### Verify NGC Access

Run `docker login nvcr.io` and continue with `make build-docker`. Authentication or permission failures for other CVDS images should be resolved with your NVIDIA account team.

## Network Requirements

### LocalStack Hostname Mapping

**Critical**: Docker Compose deployment requires a hostname mapping for LocalStack (S3-compatible storage).

To ensure that your system recognizes "localstack" as an alias for localhost, you must add a hostname mapping to your `/etc/hosts` file. The following command appends the required entry to `/etc/hosts` (requires sudo privileges):

```bash
echo "127.0.0.1   localstack" | sudo tee -a /etc/hosts
```

Verify the mapping:
```bash
grep localstack /etc/hosts
```

**Without this mapping, data ingestion and storage operations will fail.**

### Firewall and Port Requirements

CDS requires the following ports to be available. Any port conflicts must be resolved before deployment:

| Port  | Service              | Purpose                    |
|-------|----------------------|----------------------------|
| 8888  | Visual Search API    | REST API endpoint          |
| 9000  | CE1 OSS service      | Embedding service          |
| 19530 | Milvus               | Vector database            |
| 4566  | LocalStack           | S3-compatible storage      |

**Check for port conflicts:**

```bash
# Verify ports are available
ss -tuln | grep -E ':(8888|9000|19530|4566)'
```

If this command returns any results, those ports are already in use. You must either:
- Stop the services using those ports
- Modify the CDS configuration to use different ports (advanced, see deployment guide)

If no output is returned, all required ports are available.

## Storage Requirements

### Disk Space

- **Base installation**: ~50GB for Docker images and model cache
- **Local model directory**: ~20GB for CE1 OSS model files prepared before startup
- **Data storage**: Varies based on dataset size
  - Video storage: Matches your dataset size
  - Embeddings: ~1.3KB per video frame/segment
  - Metadata: Minimal (MB range)

**Recommended**: 200GB+ free disk space for development and testing with sample datasets.

### Storage Performance

- High-performance SSD recommended for model cache and vector database
- For production workloads, consider NVMe storage

## Environment Variables

CDS uses a `.env` file to manage all required environment variables. The `.env` file is located in the `deploy/standalone/` directory and is automatically loaded by Docker Compose during deployment.

Environment configuration, including setting your NGC API key and data directory, is covered in detail in the [Docker Compose Deployment Guide](docker-compose-deployment.md#step-2-configure-environment-variables).

## Next Steps

After completing all prerequisites, proceed to the [Docker Compose Deployment Guide](docker-compose-deployment.md) to deploy CDS.

For troubleshooting issues during prerequisite setup or deployment, see [Docker Compose Troubleshooting](troubleshooting-docker-compose.md).
