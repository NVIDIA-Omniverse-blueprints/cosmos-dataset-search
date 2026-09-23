# CDS Command Line Interface User Guide

The CDS CLI provides comprehensive command-line tools for managing collections, ingesting data, and performing searches.

## Installation

### Prerequisites

[Prerequisites for CLI installation]

### Installing the CLI

```bash
# Install CDS CLI with all required dependencies
make install-cds-cli
```

The CLI is available when the virtual environment is activated:

```bash
source .venv/bin/activate
cds --help
```

## Configuration

### Setting Up API Endpoint

Configure the CLI to connect to your CDS API endpoint. You can set up multiple profiles for different environments.

#### Configure Default Profile

```bash
cds config set
```

#### Configure Named Profile

```bash
# Configure a local deployment profile
cds config set --profile local

# Configure a production profile
cds config set --profile production
```

### Configuration File

The CLI stores configuration at `~/.config/cds/config`:

```ini
[default]
api_endpoint = http://production-endpoint.example.com

[local]
api_endpoint = http://localhost:8888

[production]
api_endpoint = https://production.example.com
```

### Using Profiles

Use the `--profile` flag to select which endpoint to use:

```bash
cds collections list --profile local
```

## S3 Configuration for Data Ingestion

CDS ingests data from S3-compatible storage (LocalStack, MinIO, AWS S3). Configure S3 access using one of these methods:

### Option 1: Environment Variables (Quick Setup)

```bash
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL=http://localhost:4566  # Required for LocalStack/MinIO
export AWS_DEFAULT_REGION=us-east-1

cds ingest files s3://bucket/videos/ --collection-id <ID> --extensions .mp4
```

### Option 2: AWS Profile (Multiple S3 Endpoints)

Create two configuration files:

**`~/.aws/credentials`** - Access keys:
```ini
[cds-s3]
aws_access_key_id = test
aws_secret_access_key = test
```

**`~/.aws/config`** - Endpoint URL (**required** for non-AWS S3 like localstack/minio):

```ini
[profile cds-s3]
endpoint_url = http://localstack:4566
region = us-east-1
```

For ingestion from AWS S3, your environment variables should look like:

```bash
export AWS_ACCESS_KEY_ID=xxxxxxxxxxxxxxxx
export AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxx
export AWS_DEFAULT_REGION=us-east-2
# Do not set AWS_ENDPOINT_URL when using AWS S3 (only needed for LocalStack/MinIO)
```

Or, set up a profile in your AWS credentials/config files:

**`~/.aws/credentials`**:
```ini
[cds-s3-aws]
aws_access_key_id = xxxxxxxxxxxxxx
aws_secret_access_key = xxxxxxxxxxxxxxxxxxxx
aws_session_token = xxxxxxxxxxxxxxxxxxxxxxxxxxxxx # Not necessary for Permanent keys
```

**`~/.aws/config`**:
```ini
[profile cds-s3-aws]
region = us-east-2
```

Then run ingestion with:
```bash
cds ingest files s3://<bucket>/videos/ \
  --collection-id <ID> \
  --extensions .mp4 \
  --s3-profile cds-s3-aws
```

**Note**: Do *not* set `AWS_ENDPOINT_URL` when using AWS S3.


Then use with `--s3-profile`:
```bash
cds ingest files s3://bucket/videos/ \
  --collection-id <ID> \
  --extensions .mp4 \
  --s3-profile cds-s3
```

**Key Notes:**
- Always use `s3://` URIs (not presigned URLs)
- For LocalStack, default endpoint is `http://localhost:4566`
- Profile name format differs: `[cds-s3]` in credentials, `[profile cds-s3]` in config

## Managing Pipelines

### List Available Pipelines

List all pipelines available for creating collections:

```bash
cds pipelines list
```

**Example output**:
```json
{
  "pipelines": [
    {
      "id": "cosmos_video_search_milvus",
      "enabled": true,
      "missing": []
    }
  ]
}
```

**With verbose output** (shows full pipeline configuration):

```bash
cds pipelines list --verbose
```

This shows the complete pipeline configuration including all components, connections, and initialization parameters.

### Using Different Profiles

```bash
# List pipelines from a different endpoint
cds pipelines list --profile production
```

## Managing Collections

### Create a Collection

Create a new collection for storing and searching videos:

```bash
cds collections create --pipeline cosmos_video_search_milvus --name "My Video Collection"
```

**Example output**:
```json
{
  "collection": {
    "pipeline": "cosmos_video_search_milvus",
    "name": "My Video Collection",
    "tags": {
      "default_index": "GPU_CAGRA"
    },
    "init_params": null,
    "cameras": "camera_front_wide_120fov",
    "id": "a7a5f9d38_078e_49ec_872e_a97a3277db69",
    "created_at": "2025-10-17T21:26:53.748150"
  }
}
```

**Note the collection ID** (`a7a5f9d38_078e_49ec_872e_a97a3277db69`) - you'll need this for ingestion and search.

**Advanced: Override index type**:

```bash
cds collections create \
  --pipeline cosmos_video_search_milvus \
  --name "High Performance Collection" \
  --index-type GPU_CAGRA
```

**Advanced: Use custom configuration**:

```bash
# Create a config file with custom settings
cat > my-collection-config.yaml << EOF
tags:
  storage-template: "s3://my-bucket/videos/{{filename}}"
  storage-secrets: "my-s3-credentials"
index_config:
  index_type: GPU_CAGRA
  params:
    intermediate_graph_degree: 64
    graph_degree: 32
EOF

cds collections create \
  --pipeline cosmos_video_search_milvus \
  --name "Custom Collection" \
  --config-yaml my-collection-config.yaml
```

### List Collections

List all collections in your deployment:

```bash
cds collections list
```

**Example output**:
```json
{
  "collections": [
    {
      "pipeline": "cosmos_video_search_milvus",
      "name": "My Video Collection",
      "tags": {
        "default_index": "GPU_CAGRA"
      },
      "init_params": null,
      "cameras": "camera_front_wide_120fov",
      "id": "a87235cc0_7a76_493a_8610_72080629baeb",
      "created_at": "2025-10-17T20:00:38.827842"
    }
  ]
}
```

### Get Collection Details

Get detailed information about a specific collection:

```bash
cds collections get a87235cc0_7a76_493a_8610_72080629baeb
```

**Example output**:
```json
{
  "collection": {
    "pipeline": "cosmos_video_search_milvus",
    "name": "My Video Collection",
    "tags": {
      "storage-template": "s3://my-video-bucket/msrvtt-videos/{{filename}}",
      "storage-secrets": "my-video-bucket-secrets-videos"
    },
    "init_params": null,
    "cameras": "camera_front_wide_120fov",
    "id": "a87235cc0_7a76_493a_8610_72080629baeb",
    "created_at": "2025-10-17T20:00:38.827842"
  },
  "total_documents_count": 5
}
```

The `total_documents_count` shows how many videos have been ingested.

### Delete a Collection

**Warning**: This is irreversible! All videos and embeddings will be permanently deleted.

```bash
cds collections delete a7a5f9d38_078e_49ec_872e_a97a3277db69
```

**Example output**:
```json
{
  "message": "Collection a7a5f9d38_078e_49ec_872e_a97a3277db69 deleted successfully.",
  "id": "a7a5f9d38_078e_49ec_872e_a97a3277db69",
  "deleted_at": "2025-10-17T21:27:46.499448"
}
```

## Ingesting Data

### Ingest Videos from S3

Ingest video files from an S3 bucket into a collection:

```bash
cds ingest files s3://my-bucket/videos/ \
  --collection-id a87235cc0_7a76_493a_8610_72080629baeb \
  --extensions mp4 \
  --num-workers 3 \
  --limit 10
```

**Parameters**:
- `s3://my-bucket/videos/` - S3 path containing videos
- `--collection-id` - Collection UUID (from `cds collections create`)
- `--extensions` - File extensions to ingest (mp4 for videos)
- `--num-workers` - Number of parallel workers (default: 1)
- `--limit` - Maximum number of files to ingest (optional)

**Example output**:
```
INFO:root:Loading profile default
2025-10-17 13:00:45,409 INFO worker.py:1951 -- Started a local Ray instance.
[13:00:45] 🧠 Spawned 3 file batch processors.

╭─────────────────────────────────── File ingestion ───────────────────────────────────╮╭───────── Responses ──────────╮
│                                                                                      ││                              │
│   Processed files: 5/5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04 0:00:00 ││    Status code 200: 5/5 100% │
│                                                                                      ││                              │
╰──────────────────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────╯

[13:00:52] 🚀 Finished processing job queue!
           Processed 5 files successfully
```

**Status code 200** means successful ingestion!

**Using S3 profiles** (required for AWS S3 access):

```bash
# Configure AWS profile with your S3 credentials
aws configure set aws_access_key_id YOUR_KEY --profile cds-s3-aws
aws configure set aws_secret_access_key YOUR_SECRET --profile cds-s3-aws
aws configure set region us-east-2 --profile cds-s3-aws

# Ingest videos from S3
cds ingest files s3://my-video-bucket/msrvtt-videos/ \
  --collection-id d5aa2e3d_7421_4f42_911d_1a681c43d760 \
  --extensions mp4 \
  --s3-profile cds-s3-aws \
  --limit 3 \
  --num-workers 2
```

**Verified output** (from actual test):
```
INFO:root:Loading profile default
2025-10-17 19:23:49,612 INFO worker.py:1951 -- Started a local Ray instance.
[19:23:50] 🧠 Spawned 2 file batch processors.

╭─────────────────────────────────── File ingestion ───────────────────────────────────╮╭───────── Responses ──────────╮
│                                                                                      ││                              │
│   Processed files: 3/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02 0:00:00 ││    Status code 200: 3/3 100% │
│                                                                                      ││                              │
╰──────────────────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────╯

[19:23:53] 🚀 Finished processing job queue!
           Processed 3 files successfully
200: 3
```

**Status code 200 = Success!** All 3 videos ingested successfully.

### Ingest Precomputed Embeddings

If you have precomputed embeddings in Parquet format, you can bulk insert them directly.

**Parquet Requirements**:
- Must have `id` field (string) - document identifier
- Must have `embedding` field (list of 256 floats) - vector embeddings
- Can include additional metadata fields

**Configure AWS profile**:
```bash
aws configure set aws_access_key_id YOUR_KEY --profile cds-s3-aws
aws configure set aws_secret_access_key YOUR_SECRET --profile cds-s3-aws
aws configure set region us-east-2 --profile cds-s3-aws
```

**Ingest parquet embeddings**:

> **⚠️ Important Requirements**:
> 1. The parquet file **must be in the same S3 bucket that Milvus is configured to use** (check your `milvus-values.yaml` for `externalS3.bucketName`)
> 2. The parquet file should **only contain columns defined in the collection schema** (typically `id` and `embedding`)
> 3. For metadata fields, they must be formatted using Milvus's `$meta` column format (not supported via `--metadata-cols` parameter)

```bash
# Assumption is that your aws cli is configured and you have your milvus bucket configured
# First, ensure your parquet file is in the correct bucket
# Example: if Milvus is configured to use bucket 'my-milvus-bucket'
aws s3 cp s3://your-source-bucket/embeddings.parquet s3://my-milvus-bucket/embeddings.parquet

# Then ingest (without metadata columns)
cds ingest embeddings \
  --parquet-dataset s3://my-video-bucket/milvus_embeddings.parquet \
  --collection-id a68495826_0c1d_4de4_8cdd_9e309d876ad7 \
  --id-cols id \
  --embeddings-col embedding \
  --s3-profile cds-s3-aws
```

**Parameters**:
- `--parquet-dataset` - S3 path to parquet file(s) (must be in Milvus's configured bucket)
- `--id-cols` - Columns to generate document IDs from (required)
- `--embeddings-col` - Column containing embedding vectors (default: "embeddings")
- `--s3-profile` - AWS profile with S3 credentials

**Verified output** (from actual test):
```
INFO:root:Loading profile default
2025-10-19 00:56:17,410 INFO worker.py:1951 -- Started a local Ray instance.
[00:56:17] 🧠 Spawned 1 parquet batch processors.

╭─────────────────────────────────── File ingestion ───────────────────────────────────╮╭───────── Responses ──────────╮
│                                                                                      ││                              │
│   Processed files: 1/1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01 0:00:00 ││    Status code 202: 1/1 100% │
│                                                                                      ││                              │
╰──────────────────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────╯

[00:56:20] 🚀 Finished processing job queue!
           1 files returned status code 202
202: 1
```

**Status code 202 = Accepted!** The bulk insert job was queued successfully and will be processed asynchronously.

**Verify job completion**:

Since 202 means the job was accepted (not completed), verify it succeeded:

```bash
# Get the API endpoint
VS_API=$(kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Check job status for your collection
curl -k "https://$VS_API/api/v1/jobs?collection_name=<your-collection-id>" | jq

# Verify documents were inserted
cds collections get --collection-id <your-collection-id>
```

Look for `"status": "completed"` and `"progress": 100` in the job status, and verify `total_documents_count` > 0 in the collection info.

### Monitoring Ingestion Progress

The CLI shows real-time progress with:
- Progress bar for files processed
- Status codes breakdown (200 = success, 500 = error)
- Time elapsed and remaining
- Final summary

## Searching

### Text-to-Video Search

Search for videos using natural language queries:

```bash
cds search \
  --collection-ids d5aa2e3d_7421_4f42_911d_1a681c43d760 \
  --text-query "a person walking on the street" \
  --top-k 3
```

**Example output**:
```json
{
  "retrievals": [
    {
      "id": "290c3a15d90ccf7fd3ffe5d921150bd7e4da6ae555eea37e2e199a83400b22c7",
      "metadata": {
        "filename": "video7020.mp4",
        "source_id": "04bccb4352ab07cb7a1589c1e578fd464d36b53aff84ffed2868f6ce5ba8a5eb",
        "indexed_at": "2025-10-18T02:23:51.088094",
        "source_url": "https://s3.us-east-2.amazonaws.com/my-video-bucket/msrvtt-videos/video7020.mp4?..."
      },
      "collection_id": "d5aa2e3d_7421_4f42_911d_1a681c43d760",
      "asset_url": null,
      "score": -0.0023960545659065247,
      "content": "",
      "mime_type": "video/mp4",
      "embedding": null
    },
    {
      "id": "5bc56113b11afadbea853e0010dbf593c63290451beab5669bbd76ccbeb39d7a",
      "metadata": {
        "filename": "video7024.mp4",
        "indexed_at": "2025-10-18T02:23:51.887275",
        ...
      },
      "score": -0.024522194638848305,
      ...
    },
    {
      "id": "b746f10ca6800c1f1e004354dfe089cae50bec438e7757323c1478324ce22792",
      "metadata": {
        "filename": "video7021.mp4",
        ...
      },
      "score": -0.03927513211965561,
      ...
    }
  ]
}
```

**Understanding the results**:
- `score` - Similarity score (higher is better)
- `asset_url` - Presigned S3 URL to download/view the video
- `metadata` - Associated metadata (filename, timestamps, etc.)
- Results are sorted by score (most relevant first)

### Search Multiple Collections

Search across multiple collections at once:

```bash
cds search \
  --collection-ids "d5aa2e3d_7421_4f42_911d_1a681c43d760,a87235cc0_7a76_493a_8610_72080629baeb" \
  --text-query "person walking" \
  --top-k 5
```

**Results are merged** from all specified collections and ranked by score. Each result includes the `collection_id` field showing which collection it came from.

### Search Options

**Disable asset URL generation** (faster for large result sets):

```bash
cds search \
  --collection-ids <collection-id> \
  --text-query "query text" \
  --generate-asset-url false
```

**Use different profile**:

First configure the profile:
```bash
# Configure production profile
cds config set --profile production
# Enter API endpoint: https://your-production-hostname/api
```

Then use it for commands:
```bash
cds search \
  --collection-ids d5aa2e3d_7421_4f42_911d_1a681c43d760 \
  --text-query "person walking" \
  --top-k 3 \
  --profile production
```

**Note**: The output will show `INFO:root:Loading profile production` confirming the correct profile is being used.

## Managing Secrets

**Note**: The secrets API endpoint is not implemented in this version. Use Kubernetes secrets directly for S3 credentials instead.

For Kubernetes deployments, add each storage Secret name to `secretAccess.allowedNames` in the visual-search Helm values and apply the values before using it. The application can only `get` explicitly allowed names; an empty list grants no Secret API access. Creating a Secret with `kubectl` does not add it to this allowlist. See [storage Secret access](aws-eks-deployment.md#storage-secret-access).

### Create Kubernetes Secret for S3 Access

For collections that need S3 access, create a Kubernetes secret:

```bash
# Create secret with AWS credentials
docker exec cds-deployment kubectl create secret generic my-s3-creds \
  --from-literal=aws_access_key_id=AKIA... \
  --from-literal=aws_secret_access_key=secret... \
  --from-literal=aws_region=us-east-2
```

### List Kubernetes Secrets

```bash
docker exec cds-deployment kubectl get secrets
```

### Use Secret in Collection

When creating a collection that references S3 videos:

```bash
cds collections create --pipeline cosmos_video_search_milvus \
  --name "My Collection" \
  --config-yaml <(echo "
tags:
  storage-template: 's3://my-bucket/videos/{{filename}}'
  storage-secrets: 'my-s3-creds'
")
```

## Advanced Usage

### Batch Ingestion

For ingesting large numbers of videos, increase workers for parallel processing:

```bash
cds ingest files s3://my-video-bucket/msrvtt-videos/ \
  --collection-id a9fab0958_1079_412e_b7b8_d863fcecccae \
  --extensions mp4 \
  --num-workers 10 \
  --batch-size 5 \
  --limit 30 \
  --s3-profile cds-s3-aws
```

**Verified output** (30 videos in 4 seconds):
```
INFO:root:Loading profile default
[19:53:23] 🧠 Spawned 10 file batch processors.

╭──────────────────────────────────── File ingestion ────────────────────────────────────╮╭────────── Responses ───────────╮
│                                                                                        ││                                │
│   Processed files: 30/30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04 0:00:00 ││    Status code 200: 30/30 100% │
│                                                                                        ││                                │
╰────────────────────────────────────────────────────────────────────────────────────────╯╰────────────────────────────────╯

[19:53:29] 🚀 Finished processing job queue!
           Processed 30 files successfully
200: 30
```

**Performance tips**:
- More workers = faster ingestion (10 workers processed 30 videos in 4 seconds)
- Optimal workers: 3-10 depending on your deployment size
- Use `--limit` for testing before full ingestion
- Batch size affects API request grouping (default: 1)

### Output Logging

Log ingestion results to CSV for analysis:

```bash
cds ingest files s3://my-video-bucket/msrvtt-videos/ \
  --collection-id a9fab0958_1079_412e_b7b8_d863fcecccae \
  --extensions mp4 \
  --limit 10 \
  --num-workers 3 \
  --s3-profile cds-s3-aws \
  --output-log ~/ingestion-results.csv
```

**Output shows**:
```
📖 Logging responses in /home/user/ingestion-results.csv
```

**CSV file contents**:
```csv
file,status
s3://my-video-bucket/msrvtt-videos/video7020.mp4,200
s3://my-video-bucket/msrvtt-videos/video7024.mp4,200
s3://my-video-bucket/msrvtt-videos/video7021.mp4,200
...
```

Each row shows the file path and HTTP status code (200 = success).

## CLI Command Reference

### Quick Command Summary

```bash
# Configuration
cds config set [--profile PROFILE]

# Pipelines
cds pipelines list [--verbose] [--profile PROFILE]

# Collections
cds collections create --pipeline PIPELINE --name NAME [options]
cds collections list [--profile PROFILE]
cds collections get COLLECTION_ID [--profile PROFILE]
cds collections delete COLLECTION_ID [--profile PROFILE]

# Ingestion
cds ingest files S3_PATH --collection-id ID --extensions mp4 [options]
cds ingest embeddings --parquet-dataset S3_PATH --collection-id ID --id-cols COLS [options]

# Search
cds search --collection-ids ID --text-query "TEXT" --top-k K [options]

# Secrets (use kubectl directly)
docker exec cds-deployment kubectl create secret generic NAME --from-literal=key=value
docker exec cds-deployment kubectl get secrets
docker exec cds-deployment kubectl delete secret NAME
```

## Troubleshooting

### Common CLI Issues

**Issue**: `Profile 'xyz' is not available`

**Solution**: Configure the profile first:
```bash
cds config set --profile xyz
```

**Issue**: `Collection not found`

**Solution**: List collections to verify the ID:
```bash
cds collections list
```

**Issue**: `S3 access denied`

**Solution**: Check your S3 credentials or profile configuration:
```bash
aws s3 ls s3://your-bucket/  # Test AWS credentials work
```

**Issue**: `Status code 500` during ingestion

**Solution**: Check:
- Video format is supported (MP4 recommended)
- Videos are accessible from S3
- Visual search service logs: `docker exec kubectl logs deployment/visual-search`

### Getting Help

Run `cds --help` for command overview, or `cds <command> --help` for detailed command information:

```bash
cds --help
cds collections create --help
cds ingest files --help
```

## Related Documentation

- [User Guide Overview](user-guide.md) - Main user guide
- [API Reference](api_reference.md) - REST API documentation
- [Docker Compose Troubleshooting](troubleshooting-docker-compose.md)
- [Kubernetes Troubleshooting](troubleshooting-kubernetes.md)
