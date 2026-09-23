# REST API User Guide

This guide provides a hands-on tutorial for interacting with CDS using the REST API. You'll learn how to create collections, ingest video data, perform searches, and manage your data using direct HTTP calls.

This guide assumes you have successfully deployed CDS using Docker Compose and the virtual environment is activated.

## Prerequisites

Before starting, ensure:
- CDS services are running (`make test-integration-up`)
- Virtual environment is activated: `source .venv/bin/activate`
- For Python examples: `requests` library is available (installed with CDS)

## Tutorial Overview

This tutorial walks through a complete workflow:

1. Verify the API is accessible
2. Check available embedding pipelines
3. Create a new collection
4. Prepare video files for ingestion
5. Ingest videos into the collection
6. Perform a text-to-video search
7. Clean up by deleting the collection

You can follow along using either **curl** (terminal) or **Python** code.

## Step 1: Verify API is Accessible

Check that the Visual Search API is running and responsive.

**Using curl:**

```bash
curl http://localhost:8888/health
```

Expected response: `OK`

**Using Python:**

```python
import requests

response = requests.get("http://localhost:8888/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
```

If you get a connection error, verify the services are running with `docker compose -f deploy/standalone/docker-compose.build.yml ps`

## Step 2: List Available Pipelines

Pipelines define the embedding model and vector database configuration. Check which pipelines are available.

**Using curl:**

```bash
curl http://localhost:8888/v1/pipelines
```

**Using Python:**

```python
import requests

response = requests.get("http://localhost:8888/v1/pipelines")
pipelines = response.json()

for pipeline in pipelines.get("pipelines", []):
    print(f"Pipeline: {pipeline['id']}")
    print(f"  Enabled: {pipeline['enabled']}")
    print(f"  Description: {pipeline['config']['index']['description']}")
```

You should see the `cosmos_video_search_milvus` pipeline, which uses CE1 OSS service for video embeddings and Milvus for vector storage.

## Step 3: Create a Collection

Create a new collection using the `cosmos_video_search_milvus` pipeline. Collections organize your video embeddings and enable search.

**Using curl:**

```bash
curl -X POST http://localhost:8888/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{
    "pipeline": "cosmos_video_search_milvus",
    "name": "My First Video Collection",
    "tags": {
      "storage-template": "s3://cosmos-test-bucket/videos/{{filename}}"
    }
  }'
```

**Using Python:**

```python
import requests

payload = {
    "pipeline": "cosmos_video_search_milvus",
    "name": "My First Video Collection",
    "tags": {
        "storage-template": "s3://cosmos-test-bucket/videos/{{filename}}"
    }
}

response = requests.post(
    "http://localhost:8888/v1/collections",
    json=payload
)

collection = response.json()
collection_id = collection['collection']['id']
print(f"Created collection: {collection_id}")
```

Save the collection ID from the response - you'll need it for the following steps.

## Step 4: List Collections

Verify your collection was created and retrieve its ID.

**Using curl:**

```bash
curl http://localhost:8888/v1/collections
```

**Using Python:**

```python
import requests

response = requests.get("http://localhost:8888/v1/collections")
collections = response.json()

for collection in collections.get("collections", []):
    print(f"Collection ID: {collection['id']}")
    print(f"  Name: {collection['name']}")
    print(f"  Pipeline: {collection['pipeline']}")
    print(f"  Created: {collection['created_at']}")

collection_id = collections["collections"][0]["id"]
print(f"\nUsing collection ID: {collection_id}")
```

## Step 5: Prepare Videos for Ingestion

For LocalStack (Docker Compose deployment), you need to upload videos to the LocalStack S3 bucket. We'll download a small subset of MSR-VTT videos as an example.

### Prepare Sample Videos

For this tutorial, you have two options:

**Option 1: Use Your Own Videos**

If you have your own video files (.mp4, .avi, etc.), place them in a directory:

```bash
mkdir -p ~/cds-data/sample-videos
cp /path/to/your/videos/*.mp4 ~/cds-data/sample-videos/
```

**Option 2: Download from MSR-VTT Dataset**

Download sample videos from the MSR-VTT dataset. First, download and extract the MSR-VTT video archive from HuggingFace:

```bash
mkdir -p ~/cds-data/msrvtt-videos
cd ~/cds-data/msrvtt-videos

python << 'EOF'
from huggingface_hub import hf_hub_download
import zipfile
from pathlib import Path

print("Downloading MSR-VTT videos from HuggingFace Hub...")
print("Note: This is a large file (~2.1GB) and will take several minutes")

zip_path = hf_hub_download(
    repo_id="friedrichor/MSR-VTT",
    filename="MSRVTT_Videos.zip",
    repo_type="dataset",
    resume_download=True
)

print(f"Downloaded to: {zip_path}")
print("Extracting videos (this may take a while)...")

with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(".")

print("MSR-VTT videos extracted successfully!")

Path(".videos_extracted").touch()
EOF
```

Now copy a few videos to use for the tutorial:

```bash
mkdir -p ~/cds-data/sample-videos

find ~/cds-data/msrvtt-videos -name "video70*.mp4" | head -5 | xargs -I {} cp {} ~/cds-data/sample-videos/

cd ~/cds-data/sample-videos
ls -lh *.mp4
```

This copies 5 MSR-VTT videos to your sample directory.

### Upload Videos to LocalStack

Upload the videos to the LocalStack S3 bucket using boto3 (Python S3 client):

```bash
cd ~/cds-data/sample-videos

python << 'EOF'
import boto3
from pathlib import Path

s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

bucket = 'cosmos-test-bucket'
prefix = 'videos'

video_files = list(Path('.').glob('*.mp4'))

print(f"Uploading {len(video_files)} videos to s3://{bucket}/{prefix}/")

for video_file in video_files:
    key = f"{prefix}/{video_file.name}"
    print(f"  Uploading {video_file.name}...")
    s3_client.upload_file(str(video_file), bucket, key)

print("\nUpload complete! Verifying...")

response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
if 'Contents' in response:
    print(f"\nFiles in s3://{bucket}/{prefix}/:")
    for obj in response['Contents']:
        size_mb = obj['Size'] / (1024 * 1024)
        print(f"  {obj['Key']} ({size_mb:.2f} MB)")
else:
    print("No files found in bucket")
EOF
```

You should see your uploaded video files listed with their sizes.

**Alternative**: For ingesting from AWS S3 or other S3-compatible storage, see the [MinIO Support Guide](../guides/minio-support.md) for configuration details.

## Step 6: Ingest Videos

Now ingest the videos into your collection using the REST API. We'll use presigned URLs which allow the service to access videos from S3 storage.

**Using Python:**

Create a Python script for ingestion:

```bash
cat > ingest_videos.py << 'EOF'
import boto3
import requests

collection_id = "<your-collection-id>"  # Replace with your actual collection ID

s3_client = boto3.client(
    's3',
    endpoint_url='http://localstack:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

bucket = 'cosmos-test-bucket'
prefix = 'videos'
video_files = ['video70.mp4', 'video700.mp4', 'video7000.mp4', 'video7001.mp4', 'video7002.mp4']

# First, test that we can generate and access presigned URLs
print("Testing presigned URL generation...")
test_key = f"{prefix}/{video_files[0]}"
test_url = s3_client.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket, 'Key': test_key},
    ExpiresIn=3600
)
print(f"Sample presigned URL: {test_url[:80]}...")

# Verify URL is accessible
try:
    test_response = requests.head(test_url, timeout=5)
    print(f"✓ Presigned URL is accessible (status: {test_response.status_code})")
except Exception as e:
    print(f"✗ Warning: Could not access presigned URL: {e}")
    print("  Continuing anyway...")

# Generate documents with presigned URLs
documents = []
for filename in video_files:
    key = f"{prefix}/{filename}"
    
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=3600
    )
    
    documents.append({
        'url': presigned_url,
        'mime_type': 'video/mp4',
        'metadata': {'filename': filename}
    })
    
    print(f"Prepared {filename} for ingestion")

print(f"\nRequest preview (first document):")
import json
print(json.dumps(documents[0], indent=2))

print(f"\nIngesting {len(documents)} videos via REST API...")

response = requests.post(
    f"http://localhost:8888/v1/collections/{collection_id}/documents",
    json=documents
)

print(f"Response: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✓ Successfully ingested {len(result.get('documents', []))} videos")
    print(f"Document IDs: {[d['id'][:16] + '...' for d in result.get('documents', [])]}")
else:
    print(f"✗ Error: {response.text}")
EOF
```

Edit the script to replace `<your-collection-id>` with your actual collection ID, then run it:

```bash
# Edit the collection ID in the script
nano ingest_videos.py

# Run the ingestion
python ingest_videos.py
```

**Key Points:**
- Use the `url` field with the presigned URL directly (no special formatting needed)
- The API automatically handles the internal formatting for CE1 OSS service
- The `mime_type` must be `'video/mp4'`
- Script tests URL accessibility before ingesting

## Step 7: Perform a Search

Now that videos are ingested, perform a text-to-video search to find relevant content.

**Using curl:**

```bash
COLLECTION_ID="<your-collection-id>"

curl -X POST http://localhost:8888/v1/collections/${COLLECTION_ID}/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [{"text": "person walking outdoors"}],
    "top_k": 3
  }'
```

**Using Python:**

Create a search script:

```bash
cat > search_videos.py << 'EOF'
import requests
import json

collection_id = "<your-collection-id>"  # Replace with your actual collection ID

search_payload = {
    "query": [{"text": "person walking outdoors"}],
    "top_k": 3
}

print(f"Searching collection: {collection_id}")
print(f"Query: {search_payload['query'][0]['text']}\n")

response = requests.post(
    f"http://localhost:8888/v1/collections/{collection_id}/search",
    json=search_payload
)

print(f"Response status: {response.status_code}")

if response.status_code != 200:
    print(f"Error: {response.text}")
else:
    results = response.json()
    
    retrievals = results.get('retrievals', [])
    print(f"Found {len(retrievals)} results:\n")
    
    for i, result in enumerate(retrievals, 1):
        print(f"Result {i}:")
        print(f"  Score: {result['score']:.4f}")
        if 'metadata' in result:
            if 'filename' in result['metadata']:
                print(f"  Filename: {result['metadata']['filename']}")
            if 'source_url' in result['metadata']:
                print(f"  Video: {result['metadata']['source_url'][:70]}...")
EOF
```

Edit and run:

```bash
nano search_videos.py  # Update collection_id
python search_videos.py
```

The response includes ranked video segments with similarity scores. Higher scores indicate better matches to your text query.

## Step 8: Delete the Collection

When you're done experimenting, clean up by deleting the collection. This removes all ingested data and embeddings.

**Using curl:**

```bash
COLLECTION_ID="<your-collection-id>"

curl -X DELETE http://localhost:8888/v1/collections/${COLLECTION_ID}
```

**Using Python:**

```python
import requests

collection_id = "<your-collection-id>"

response = requests.delete(
    f"http://localhost:8888/v1/collections/{collection_id}"
)

print(f"Collection deleted: {response.status_code}")
```

**Warning**: Deletion is permanent and cannot be undone.

## Complete Example Script

Here's a complete Python script that performs all the tutorial steps from start to finish:

```bash
cat > api_tutorial_complete.py << 'EOF'
import requests
import boto3
from pathlib import Path
import time

BASE_URL = "http://localhost:8888"

print("=" * 60)
print("CDS REST API Complete Tutorial")
print("=" * 60)

# Step 1: Check API Health
print("\n1. Checking API health...")
health = requests.get(f"{BASE_URL}/health")
assert health.status_code == 200, "API not healthy"
print("   ✓ API is healthy")

# Step 2: List Available Pipelines
print("\n2. Listing available pipelines...")
pipelines_resp = requests.get(f"{BASE_URL}/v1/pipelines")
pipelines = pipelines_resp.json()
pipeline_id = "cosmos_video_search_milvus"
print(f"   ✓ Using pipeline: {pipeline_id}")

# Step 3: Create Collection
print("\n3. Creating collection...")
collection_payload = {
    "pipeline": pipeline_id,
    "name": "API Tutorial Collection",
    "tags": {"storage-template": "s3://cosmos-test-bucket/videos/{{filename}}"}
}
collection_resp = requests.post(f"{BASE_URL}/v1/collections", json=collection_payload)
collection_id = collection_resp.json()["collection"]["id"]
print(f"   ✓ Created collection: {collection_id}")

# Step 4: List Collections
print("\n4. Listing collections...")
collections = requests.get(f"{BASE_URL}/v1/collections").json()
print(f"   ✓ Total collections: {len(collections.get('collections', []))}")

# Step 5: Prepare and Upload Videos to LocalStack
print("\n5. Preparing sample videos...")

# Assuming videos are already downloaded to ~/cds-data/sample-videos
sample_dir = Path.home() / "cds-data" / "sample-videos"
video_files = list(sample_dir.glob("*.mp4"))[:5]

if not video_files:
    print("   ✗ No videos found. Please download videos first (see Step 5 in guide)")
    exit(1)

print(f"   Found {len(video_files)} videos in {sample_dir}")

# Upload to LocalStack
s3_client = boto3.client(
    's3',
    endpoint_url='http://localstack:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

bucket = 'cosmos-test-bucket'
prefix = 'videos'

print("   Uploading videos to LocalStack...")
for video_file in video_files:
    key = f"{prefix}/{video_file.name}"
    s3_client.upload_file(str(video_file), bucket, key)
    print(f"      Uploaded {video_file.name}")

# Step 6: Ingest Videos
print("\n6. Ingesting videos via REST API...")

documents = []
for video_file in video_files:
    key = f"{prefix}/{video_file.name}"
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=3600
    )
    documents.append({
        'url': presigned_url,
        'mime_type': 'video/mp4',
        'metadata': {'filename': video_file.name}
    })

ingest_resp = requests.post(
    f"{BASE_URL}/v1/collections/{collection_id}/documents",
    json=documents
)

if ingest_resp.status_code == 200:
    print(f"   ✓ Ingested {len(documents)} videos")
else:
    print(f"   ✗ Ingestion failed: {ingest_resp.text}")
    exit(1)

print("   Waiting for embeddings to be generated (30 seconds)...")
time.sleep(30)

# Step 7: Perform Search
print("\n7. Performing text-to-video search...")
search_payload = {
    "query": [{"text": "person walking"}],
    "top_k": 3
}

search_resp = requests.post(
    f"{BASE_URL}/v1/collections/{collection_id}/search",
    json=search_payload
)

results = search_resp.json()
retrievals = results.get('retrievals', [])
print(f"   ✓ Found {len(retrievals)} results")

for i, result in enumerate(retrievals, 1):
    print(f"\n   Result {i}:")
    print(f"      Score: {result['score']:.4f}")
    if 'metadata' in result and 'filename' in result['metadata']:
        print(f"      Filename: {result['metadata']['filename']}")

# Step 8: Delete Collection
print("\n8. Cleaning up - deleting collection...")
delete_resp = requests.delete(f"{BASE_URL}/v1/collections/{collection_id}")
print(f"   ✓ Collection deleted (status: {delete_resp.status_code})")

print("\n" + "=" * 60)
print("✅ Tutorial complete!")
print("=" * 60)
EOF
```

Run the complete tutorial:

```bash
python api_tutorial_complete.py
```

**Note**: Ensure you have already downloaded videos to `~/cds-data/sample-videos` as described in Step 5 of the tutorial.

## Next Steps

Now that you've learned the basics of the CDS REST API, continue your learning:

**[Back to User Guide](user-guide.md)** - Explore the CDS CLI and REST API

### Continue Learning

- **[Complete API Reference](../guides/api.md)** - Advanced operations including video-to-video search, filters, bulk operations, and more
- **[API Troubleshooting](../guides/troubleshooting.md#rest-api-troubleshooting)** - Common API issues and solutions
- **[Live OpenAPI Schema](api_reference.md#live-openapi-schema)** - Machine-readable API specification from your deployed service

### Other Resources

- **[MinIO Support Guide](../guides/minio-support.md)** - Configure alternative S3-compatible storage
- **[Docker Compose Troubleshooting](troubleshooting-docker-compose.md)** - Deployment troubleshooting
