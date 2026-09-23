# MinIO Support for File Ingestion

This guide explains how to ingest video files and data from MinIO object storage into CVDS.

## Overview

CVDS uses boto3 for S3 operations, providing native compatibility with MinIO. Since MinIO implements the S3 API, it works seamlessly with CVDS for file ingestion. Other S3-compatible storage solutions should work similarly, but this guide focuses on tested MinIO configurations.

## Prerequisites

- CVDS service running
- MinIO server accessible from CVDS
- MinIO credentials (access key and secret key)

## CDS 1.2.0 security configuration

Use credentials and an endpoint appropriate to your deployment; the example `minioadmin` credentials below are for isolated local evaluation only. A service in a pod/container cannot reach a separate MinIO service through its own `localhost`. Configure a hostname reachable from CDS, CE1 and Milvus. If CE1 Compose already uses host port 9000, use a different host port for MinIO.

For private/HTTP media endpoints, configure CE1's explicit endpoint/origin opt-in. CDS text/Parquet imports use a separate [URL and S3 endpoint policy](../docs/import-url-security.md). Configure both policies when both paths are used; do not disable validation to make a private endpoint work.

For Kubernetes, add the created storage Secret name to `secretAccess.allowedNames` in the visual-search Helm values and apply the values before ingestion. The application has `get` access only to configured names, not namespace-wide Secret access. See [storage Secret access](../docs/aws-eks-deployment.md#storage-secret-access).

## MinIO Setup

### 1. Pull the Docker Image

First, pull the `minio/minio` image from Docker Hub:

```bash
docker pull minio/minio
```

### 2. Run the Container

Run the container, specifying the ports, access keys, and a data volume:

```bash
docker run -d --name minio-fileserver \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  -v /mnt/data:/data \
  minio/minio server /data --console-address ":9001"
```

**Parameters explained:**
- `-p 9000:9000`: Maps the MinIO API port
- `-p 9001:9001`: Maps the MinIO Console port
- `-e "MINIO_ROOT_USER=minioadmin"`: Sets the access username
- `-e "MINIO_ROOT_PASSWORD=minioadmin"`: Sets the access password
- `-v /mnt/data:/data`: Creates a persistent volume (change `/mnt/data` to your preferred local directory)
- `--console-address ":9001"`: Specifies the console address

### 3. Access MinIO Console

After the container is running, access the MinIO Console at `http://localhost:9001` using the credentials `minioadmin:minioadmin`.

### 4. Create Bucket and Upload Files

> **Note**: For detailed instructions on uploading videos to MinIO, please refer to the [MinIO documentation](https://min.io/docs/minio/linux/reference/minio-mc.html) or use the AWS CLI S3 commands with the MinIO endpoint.

## Collection Creation with Storage Secrets

Before ingesting files, you need to create a collection with proper storage configuration. This example shows how to create a collection with MinIO/S3 storage secrets using the API directly.

### Kubernetes Deployment

#### 1. Create Kubernetes Secret for MinIO Credentials

```bash
# Set your MinIO credentials
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_REGION="us-east-1"
BUCKET_NAME="video-dataset"
SECRETS_NAME="${BUCKET_NAME}-secrets"

# Create Kubernetes secret with MinIO credentials
kubectl create secret generic $SECRETS_NAME \
  --from-literal=aws_access_key_id=$MINIO_ACCESS_KEY \
  --from-literal=aws_secret_access_key=$MINIO_SECRET_KEY \
  --from-literal=aws_region=$MINIO_REGION \
  --from-literal=endpoint_url=http://localhost:9000
```

#### 2. Create Collection with Storage Configuration

```bash
# Get your API endpoint
VS_API="your-cvds-api-endpoint"  # e.g., localhost:8888 or your ingress hostname

# Create collection with storage secrets using curl
COLLECTION=$(curl -s -X POST "https://$VS_API/v1/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "cosmos_video_search_milvus",
    "name": "MinIO Video Collection",
    "tags": {
      "storage-template": "s3://'$BUCKET_NAME'/videos/{{filename}}",
      "storage-secrets": "'$SECRETS_NAME'"
    },
    "collection_config": {},
    "index_config": {
      "index_type": "GPU_CAGRA",
      "params": {
        "intermediate_graph_degree": 64,
        "graph_degree": 32,
        "build_algo": "IVF_PQ",
        "cache_dataset_on_device": "true",
        "adapt_for_cpu": "true"
      },
      "metric_type": "IP"
    },
    "metadata_config": {
      "allow_dynamic_schema": true,
      "fields": []
    }
  }')

# Extract collection ID from response
COLLECTION_ID=$(echo "$COLLECTION" | jq -r '.collection.id')
echo "Created collection with ID: $COLLECTION_ID"
```

**Key Configuration Elements:**
- `storage-template`: S3 URL template with `{{filename}}` placeholder for dynamic file paths
- `storage-secrets`: Name of the Kubernetes secret containing MinIO credentials
- `tags`: Collection metadata that enables asset URL generation during search

### Docker Compose Deployment

**Important**: MinIO requires a custom endpoint URL, so you **cannot** use simple environment variables like with AWS S3.

### Single File Ingestion

```bash
#!/bin/bash
MINIO_ENDPOINT="http://localhost:9000"
BUCKET="video-dataset"
COLLECTION_ID="your-collection-id"
FILENAME="sample.mp4"

curl -X POST "http://localhost:8888/v1/collections/${COLLECTION_ID}/documents" \
  -H 'Content-Type: application/json' \
  -d "[{
    \"url\": \"${MINIO_ENDPOINT}/${BUCKET}/videos/${FILENAME}\",
    \"mime_type\": \"video/mp4\",
    \"metadata\": {\"filename\": \"${FILENAME}\", \"source\": \"minio\"}
  }]"

echo "Ingested: $FILENAME"
```
