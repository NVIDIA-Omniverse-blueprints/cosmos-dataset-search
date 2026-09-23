# CVDS API Guide

This guide provides practical curl examples for interacting with the Cosmos Dataset Search (CVDS) API. For the complete API specification, see [API Reference](../docs/api_reference.md).

## Base URLs

- **Local Development**: `http://localhost:8888/v1`
- **Standalone Deployment**: `http://<host_ip>:8888/v1`
- **EKS Deployment**: `https://<ingress-hostname>/api/v1`

## Service Health & Status

### Health Check
```bash
curl -X GET http://localhost:8888/health
```

### List Available Pipelines
```bash
curl -X GET http://localhost:8888/v1/pipelines \
  -H 'accept: application/json'
```

**Expected Response:**
```json
{
  "pipelines": [
    {
      "id": "cosmos_video_search_milvus",
      "enabled": true,
      "missing": [],
      "config": {
        "index": {
          "description": "Video indexing and search pipeline using Milvus."
        }
      }
    }
  ]
}
```

### Visualize Pipeline
```bash
curl -X GET http://localhost:8888/v1/pipelines/draw/cosmos_video_search_milvus \
  -H 'accept: application/json'
```

## Collection Management

### Create Collection
```bash
curl -X POST http://localhost:8888/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{
    "pipeline": "cosmos_video_search_milvus",
    "name": "my_video_collection",
    "tags": {
      "storage-template": "s3://my-bucket/videos/{{filename}}"
    }
  }'
```

### List Collections
```bash
curl -X GET http://localhost:8888/v1/collections
```

**Expected Response:**
```json
{
  "collections": [
    {
      "pipeline": "cosmos_video_search_milvus",
      "name": "MSR-VTT Collection Accuracy Test",
      "tags": {
        "default_index": "GPU_CAGRA"
      },
      "id": "a08101386_b2ab_4887_9cac_34a152f21e8c",
      "created_at": "2025-10-03T00:30:18.608917"
    }
  ]
}
```

### Get Collection Details
```bash
curl -X GET http://localhost:8888/v1/collections/{collection_id}
```

### Update Collection
```bash
curl -X PATCH http://localhost:8888/v1/collections/{collection_id} \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "updated_collection_name",
    "tags": {
      "updated_tag": "value"
    }
  }'
```

### Delete Collection
```bash
curl -X DELETE http://localhost:8888/v1/collections/{collection_id}
```

### Get Pipeline Collections
```bash
curl -X GET http://localhost:8888/v1/pipelines/cosmos_video_search_milvus/collections
```

## Document Indexing

### Index Documents
```bash
curl -X POST http://localhost:8888/v1/collections/{collection_id}/documents \
  -H 'Content-Type: application/json' \
  -d '[
    {
      "url": "https://my-bucket.s3.amazonaws.com/video.mp4",
      "mime_type": "video/mp4",
      "metadata": {
        "title": "Sample Video",
        "duration": "120",
        "resolution": "1920x1080"
      }
    }
  ]'
```

### Delete All Documents from Collection
```bash
curl -X DELETE http://localhost:8888/v1/collections/{collection_id}/documents
```

### Delete Specific Document
```bash
curl -X DELETE http://localhost:8888/v1/collections/{collection_id}/documents/{document_id}
```

## Bulk Data Operations

### Bulk Insert Data (Parquet Files)

> **⚠️ Important**: Parquet files must be accessible to the Milvus instance. They need to be stored in the same S3 bucket or LocalStack instance that Milvus is configured to use (as defined in your Milvus configuration YAML). The Milvus service must have read access to these files for bulk insertion to work.

```bash
curl -X POST http://localhost:8888/v1/insert-data \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_name": "my_collection",
    "parquet_paths": [
      "s3://cosmos-test-bucket/embeddings.parquet"
    ],
    "access_key": "test",
    "secret_key": "test",
    "endpoint_url": "http://localstack:4566"
  }'
```

**Storage Requirements:**
- Files must be in S3 bucket or LocalStack accessible to Milvus
- Use the same storage endpoint/credentials as configured in Milvus
- For the Compose deployment: `http://localstack:4566` with bucket `cosmos-test-bucket`. The import request uses the service-visible endpoint, not the host's `localhost` address.
- For production: your configured S3 bucket with proper IAM permissions

### Check Job Status
```bash
curl -X GET http://localhost:8888/v1/job-status/{job_id}
```

### List All Jobs
```bash
curl -X GET http://localhost:8888/v1/jobs
```

## Search & Retrieval

### Text-to-Video Search
```bash
curl -X POST http://localhost:8888/v1/collections/{collection_id}/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [{"text": "car driving through tunnel"}],
    "top_k": 10
  }'
```

**Response Format:**
```json
{
  "retrievals": [
    {
      "id": "doc_id",
      "score": 0.95,
      "metadata": {"filename": "video.mp4", "source_url": "..."},
      "collection_id": "...",
      "mime_type": "video/mp4"
    }
  ]
}
```

### Video-to-Video Search

Search using a video file as the query:

```bash
curl -X POST http://localhost:8888/v1/collections/{collection_id}/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [{"video": "s3://cosmos-test-bucket/videos/query-video.mp4"}],
    "top_k": 5
  }'
```

### Advanced Search with Filters

Filter search results based on metadata:

```bash
curl -X POST http://localhost:8888/v1/collections/{collection_id}/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [{"text": "people dancing"}],
    "top_k": 10,
    "filters": {"category": "outdoor"},
    "reconstruct": false,
    "search_params": {}
  }'
```

### Multi-Collection Retrieval
```bash
curl -X POST http://localhost:8888/v1/retrieval \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "text": "car driving through tunnel"
    },
    "collections": ["collection_id_1", "collection_id_2"],
    "params": {
      "nb_neighbors": 10,
      "reconstruct": false
    },
    "generate_asset_url": true
  }'
```

## Machine Learning Operations

### Linear Probe (Deprecated)
```bash
curl -X POST http://localhost:8888/v1/linear_probe \
  -H 'Content-Type: application/json' \
  -d '{
    "grounding_queries": [
      {
        "text": "picture of a cat"
      }
    ],
    "labels": [
      {
        "collection_name": "collection_id",
        "labelled_documents": {
          "document_id_1": true,
          "document_id_2": false
        }
      }
    ]
  }'
```

### Search Refinement Training
```bash
curl -X POST http://localhost:8888/v1/search_refinement/train \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "linear_probe",
    "grounding_queries": [
      {
        "text": "example query"
      }
    ],
    "labels": [
      {
        "collection_name": "collection_id",
        "labelled_documents": {
          "document_id": true
        }
      }
    ]
  }'
```

## Administration

### Flush Collection (Admin)
```bash
curl -X POST http://localhost:8888/v1/admin/collections/{collection_id}/flush
```

### Metrics (Prometheus)
```bash
curl -X GET http://localhost:8888/v1/metrics
```

## CE1 OSS Service (Direct Access)

### Text Embeddings
```bash
# Single text embedding
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "nvidia/cosmos-embed1"
  }'

# Multiple text embeddings
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello, world!", "How are you?", "This is a test"],
    "model": "nvidia/cosmos-embed1"
  }'
```

### CE1 OSS Health Check
```bash
curl -X GET http://localhost:9000/v1/health/live
curl -X GET http://localhost:9000/v1/health/ready
```

## EKS Deployment Examples

### Get Ingress Hostname
```bash
INGRESS_HOSTNAME=$(kubectl get ingress simple-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "API URL: https://$INGRESS_HOSTNAME/api"
```

### Search with EKS Deployment
```bash
curl -k -X POST https://$INGRESS_HOSTNAME/api/v1/collections/{collection_id}/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "text": "your search term"
    },
    "top_k": 5
  }'
```

## Common Response Examples

### Successful Search Response
```json
{
  "retrievals": [
    {
      "id": "doc_123",
      "score": 0.95,
      "metadata": {
        "title": "Sample Video",
        "duration": "120"
      },
      "asset_url": "https://my-bucket.s3.amazonaws.com/video.mp4"
    }
  ]
}
```

### Error Response
```json
{
  "detail": "Collection not found",
  "status": 404
}
```

### Job Status Response
```json
{
  "job_id": "job_123",
  "status": "completed",
  "progress": 100,
  "message": "Bulk insert completed successfully"
}
```

## Environment Variables

For easier testing, set these environment variables:

```bash
export API_BASE_URL="http://localhost:8888/v1"
export COLLECTION_ID="your-collection-id"
export COSMOS_EMBED_URL="http://localhost:9000"
```

Then use in curl commands:
```bash
curl -X GET ${API_BASE_URL}/collections
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the service is running
   ```bash
   curl -X GET http://localhost:8888/health
   ```

2. **404 Not Found**: Check collection ID exists
   ```bash
   curl -X GET http://localhost:8888/v1/collections
   ```

3. **Pipeline Errors**: Verify pipeline is enabled
   ```bash
   curl -X GET http://localhost:8888/v1/pipelines
   ```

4. **Bulk Insert Failures**: Ensure parquet files are in Milvus-accessible storage
   - Check Milvus configuration for storage settings
   - For Compose: use LocalStack bucket `cosmos-test-bucket` at the service-visible endpoint `http://localstack:4566`
   - For production: verify file paths use correct S3 bucket and Milvus has read permissions

### Debug Mode
Add `-v` flag to curl for verbose output:
```bash
curl -v -X GET http://localhost:8888/v1/collections
```

## Rate Limits & Constraints

- **Query Rate**: ~200 queries/second (sustained)
- **Video Upload**: Max 50MB per video file
- **Batch Size**: Max 100 documents per request
- **Concurrent Processing**: Limited by GPU memory allocation
- **Bulk Insert**: Supports Parquet files with proper schema validation

## Next Steps

- [API Reference](../docs/api_reference.md) - Complete OpenAPI specification
- [Docker Compose Deployment Guide](../docs/docker-compose-deployment.md) - Local deployment
- [AWS EKS Deployment Guide](../docs/aws-eks-deployment.md) - Kubernetes deployment on AWS
