# Docker Compose Troubleshooting Guide

This guide covers common issues and solutions when deploying CDS with Docker Compose.

## General Troubleshooting Steps

### Checking Service Status

Check the status of all services:

```bash
# View all running containers
docker compose -f deploy/standalone/docker-compose.build.yml ps

# View specific service status
docker compose -f deploy/standalone/docker-compose.build.yml ps visual-search
```

### Viewing Logs

Monitor service logs for debugging:

```bash
# View all service logs
make test-integration-logs

# View specific service logs
docker compose -f deploy/standalone/docker-compose.build.yml logs visual-search
docker compose -f deploy/standalone/docker-compose.build.yml logs cosmos-embed-nim
docker compose -f deploy/standalone/docker-compose.build.yml logs milvus

# Follow logs in real-time
docker compose -f deploy/standalone/docker-compose.build.yml logs -f visual-search
```

### Restarting Services

Restart services to resolve transient issues:

```bash
# Restart all services
make test-integration-down
make test-integration-up

# Restart specific service only
docker compose -f deploy/standalone/docker-compose.build.yml restart visual-search
```

## Common Issues

### Prerequisite Setup Issues

#### Docker Permission Denied

If you get permission denied errors when running Docker commands:

**Solution:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

Log out and back in, then try again. This adds your user to the docker group, allowing you to run Docker commands without sudo.

#### NGC Authentication Failed

If you cannot authenticate with NGC or pull images from nvcr.io:

**Solution:**
```bash
# Re-authenticate with NGC
docker logout nvcr.io
docker login nvcr.io
Username: $oauthtoken
Password: <your-NGC-API-key>
```

Ensure you're using `$oauthtoken` as the username and your NGC API key as the password.

### Services Won't Start

#### Docker Daemon Not Running

**Problem**: Docker service is not running.

**Solution:**
```bash
# Check if Docker is running
docker ps

# If not running, start Docker service
sudo systemctl start docker

# Verify Docker is running
docker --version
```

#### Environment Variables Not Set

**Problem**: Required environment variables are missing or incorrect.

**Solution:**
```bash
# Re-validate environment configuration
bash deploy/standalone/scripts/validate_env.sh

# If validation fails, edit your .env file
nano deploy/standalone/.env

# Re-run validation
bash deploy/standalone/scripts/validate_env.sh
```

#### Port Conflicts

**Problem**: Required ports are already in use by other services.

**Solution:**
```bash
# Check for port conflicts
ss -tuln | grep -E ':(8888|9000|19530|4566|8080)'

# If ports are in use, stop conflicting services or change ports in .env file
# Common conflicts: Other web servers on port 8080, other APIs on port 8888
```

#### Insufficient Resources

**Problem**: Not enough system resources (memory, disk space) available.

**Solution:**
```bash
# Check available disk space
df -h

# Check available memory
free -h

# Clean up Docker resources if needed
docker system prune -af
docker volume prune -f
```

### GPU Not Detected

#### NVIDIA Driver Issues

**Solution:**
```bash
# Verify NVIDIA driver is installed and working
nvidia-smi

# If nvidia-smi fails, reinstall NVIDIA drivers
sudo apt-get update
sudo apt-get install --reinstall nvidia-driver-525
sudo reboot
```

After reboot, verify with `nvidia-smi`.

#### NVIDIA Container Toolkit Not Installed

**Solution:**
```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get install --reinstall nvidia-container-toolkit

# Reconfigure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### Docker Not Configured for GPU

**Solution:**
```bash
# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Cosmos-Embed NIM Issues

#### Video Embedding Requests Fail with HTTP 500

If video requests fail with `All N inputs failed during processing` (`Permission denied: '/tmp/ram/...'` in the Triton log) while text requests work, ensure the `cosmos-embed` tmpfs mount in `deploy/standalone/docker-compose.build.yml` is `- /tmp/ram:size=2g,mode=1777`, then recreate the container (`docker compose -f docker-compose.build.yml up -d --force-recreate cosmos-embed`).

### Network Issues

#### Cannot Access Web UI

**Problem**: Web UI loads but shows no pipelines, or UI cannot connect to backend API.

**Solution:**

If accessing the UI from a different machine than the deployment host (e.g., remote server):

```bash
# Edit .env file and set CDS_URL to the deployment host's IP or hostname
nano deploy/standalone/.env

# Add or update:
CDS_URL=http://<deployment-host-ip>:8888

# Example for server with IP 192.168.1.100:
CDS_URL=http://192.168.1.100:8888

# Re-validate and restart services
bash deploy/standalone/scripts/validate_env.sh
make test-integration-down
make test-integration-up
```

**For local access only** (browser on same host as deployment):
- Ensure `CDS_URL` is not set or is set to `http://localhost:8888`
- Access UI at `http://localhost:8080/cosmos-dataset-search`

**Verify the fix:**
```bash
# The UI should now display available pipelines
# Navigate to http://<host-ip>:8080/cosmos-dataset-search
```

#### Services Cannot Communicate

**Problem**: Services cannot communicate with each other (e.g., Visual Search cannot reach Milvus or Cosmos-embed NIM).

**Solution:**
```bash
# Check Docker network
docker network ls
docker network inspect standalone_default

# Verify all services are on the same network
docker compose -f deploy/standalone/docker-compose.build.yml ps

# Restart services to recreate network
make test-integration-down
make test-integration-up
```

### Storage Issues

#### Disk Space Exhausted

**Problem**: Running out of disk space during deployment or operation.

**Solution:**
```bash
# Check available disk space
df -h

# Clean up Docker resources
docker system prune -af
docker volume prune -f

# Remove old/unused images
docker image prune -af

# If still low on space, remove test data and restart
make test-integration-clean
```

#### Volume Mount Issues

**Problem**: Container fails to start due to volume mount errors.

**Solution:**
```bash
# Verify DATA_DIR exists and is accessible
ls -la $DATA_DIR

# Create if missing
mkdir -p $DATA_DIR

# Ensure correct permissions
chmod 755 $DATA_DIR

# Restart services
make test-integration-down
make test-integration-up
```

### Model Loading Issues

#### Cosmos-embed NIM Model Download Failures

**Problem**: Cosmos-embed NIM container fails to download models or times out during startup.

**Solution:**
```bash
# Verify NGC authentication
docker login nvcr.io

# Check NIM cache directory exists with correct permissions
ls -la ~/.cache/nim
chmod 777 ~/.cache/nim

# Check available disk space (models are ~20GB)
df -h ~/.cache

# Monitor NIM container logs to see download progress
docker compose -f deploy/standalone/docker-compose.build.yml logs -f cosmos-embed-nim

# If download fails, remove cache and retry
rm -rf ~/.cache/nim/*
make test-integration-down
make test-integration-up
```

#### Out of Memory When Loading Models

**Problem**: GPU runs out of memory when loading models.

**Solution:**
```bash
# Check GPU memory usage
nvidia-smi

# Verify you have the minimum required GPU memory (16GB+, 24GB+ recommended)
# If using a smaller GPU, this deployment may not work

# Ensure no other processes are using the GPU
nvidia-smi
# Kill other GPU processes if needed

# Restart services
make test-integration-down
make test-integration-up
```

### Database Issues

#### Milvus Connection Errors

**Problem**: Services cannot connect to Milvus or Milvus fails to start.

**Solution:**
```bash
# Check Milvus container status
docker compose -f deploy/standalone/docker-compose.build.yml ps milvus

# Check Milvus logs
docker compose -f deploy/standalone/docker-compose.build.yml logs milvus

# Verify etcd is running (Milvus dependency)
docker compose -f deploy/standalone/docker-compose.build.yml ps milvus-etcd

# Restart Milvus and dependencies
make test-integration-down
make test-integration-up
```

#### Milvus Data Corruption

**Problem**: Milvus reports data corruption or index errors.

**Solution:**
```bash
# Clean up and restart with fresh data
make test-integration-clean
make test-integration-up

# Re-ingest your data
make ingest-msrvtt-small
```

## Debugging Techniques

### Viewing Service Logs

View detailed logs for debugging:

```bash
# View all service logs
make test-integration-logs

# View specific service with timestamps
docker compose -f deploy/standalone/docker-compose.build.yml logs -f --timestamps visual-search

# View last 100 lines of logs
docker compose -f deploy/standalone/docker-compose.build.yml logs --tail=100 cosmos-embed-nim
```

### Inspecting Container State

Check container configuration and state:

```bash
# Inspect container details
docker inspect cosmos-embed-nim

# Check container resource usage
docker stats

# View container environment variables
docker compose -f deploy/standalone/docker-compose.build.yml exec visual-search env
```

### Resource Monitoring

Monitor system resources:

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Check Docker disk usage
docker system df
```

## Data Issues

### Cannot Ingest Data

**Problem**: Data ingestion fails or hangs.

**Solution:**
```bash
# Verify LocalStack hostname mapping
grep localstack /etc/hosts
# Expected: 127.0.0.1   localstack

# If missing, add it
echo "127.0.0.1   localstack" | sudo tee -a /etc/hosts

# Check LocalStack is running and healthy
curl http://localhost:4566/health

# Verify S3 credentials in .env file
bash deploy/standalone/scripts/validate_env.sh

# Check Visual Search service logs
docker compose -f deploy/standalone/docker-compose.build.yml logs visual-search
```

### Search Returns No Results

**Problem**: Search queries return no results even after data ingestion.

**Solution:**
```bash
# Verify collection exists and has data
curl http://localhost:8888/v1/collections

# Check collection details
cds collections get --collection-id <id> --profile local

# Verify embedding pipeline is working
curl http://localhost:9000/v1/health/ready

# Check Milvus status
docker compose -f deploy/standalone/docker-compose.build.yml logs milvus
```

### Corrupted Index

**Problem**: Milvus index appears corrupted or service won't start.

**Solution:**
```bash
# Clean up and restart
make test-integration-clean
make test-integration-up

```

## Recovery Procedures

### Important Note on Data Persistence

**All data ingested into CDS is ephemeral.** Data is stored in Docker volumes that are removed when services are stopped. Any restart requires re-ingesting your data.

### Restart Services

To restart the deployment:

```bash
# Stop services
make test-integration-down

# Restart services
make test-integration-up

# Re-ingest data
make ingest-msrvtt-small
```

### Clean Restart (Rebuild Images)

To stop services and also remove Docker images:

```bash
# Stop services and remove images
make test-integration-clean

# Restart and rebuild
make test-integration-up

# Re-ingest data
make ingest-msrvtt-small
```

Use `make test-integration-clean` when you want to force a rebuild of Docker images on the next startup.

## Getting Help

### Collecting Diagnostic Information

When reporting issues, collect the following information:

```bash
# System information
uname -a
docker --version
docker compose version
nvidia-smi

# Service status
docker compose -f deploy/standalone/docker-compose.build.yml ps

# Service logs (last 200 lines)
docker compose -f deploy/standalone/docker-compose.build.yml logs --tail=200 > cds-logs.txt

# Environment validation
bash deploy/standalone/scripts/validate_env.sh
```

### Additional Resources

- [Docker Compose Deployment Guide](docker-compose-deployment.md) - Complete deployment instructions
- [Docker Compose Prerequisites](docker-compose-prerequisites.md) - System requirements
- [CDS User Guide](user-guide.md) - Using CDS after deployment
- [API Reference](api_reference.md) - REST API documentation
