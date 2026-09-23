#!/usr/bin/env python3
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

"""
L40 Bulk Ingestion Benchmark

Measures bulk ingestion throughput for Milvus with GPU_CAGRA indexing on L40.
"""

import argparse
import subprocess
import time

import boto3
import requests
from pymilvus import Collection, connections


def upload_to_s3(file_path: str, bucket: str, key: str, endpoint_url: str):
    """Upload parquet file to S3/LocalStack."""
    print(f"Uploading {file_path} to s3://{bucket}/{key}...")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )
    
    s3_client.upload_file(file_path, bucket, key)
    print(f"✓ Uploaded successfully")


def create_collection(pipeline: str = "cosmos_video_search_milvus"):
    """Create a GPU_CAGRA collection using CDS CLI."""
    print(f"Creating collection with pipeline: {pipeline}")
    
    result = subprocess.run(
        [
            "cds", "collections", "create",
            "--pipeline", pipeline,
            "--name", "L40 Bulk Ingestion Benchmark",
            "--config-yaml", "src/visual_search/client/default.yaml"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error creating collection: {result.stderr}")
        raise RuntimeError("Failed to create collection")
    
    # Parse collection ID from output
    import json
    output = result.stdout
    # Find JSON in output
    json_start = output.find('{')
    if json_start >= 0:
        json_data = json.loads(output[json_start:])
        collection_id = json_data['collection']['id']
        print(f"✓ Collection created: {collection_id}")
        return collection_id
    else:
        raise RuntimeError("Could not parse collection ID from output")


def run_bulk_ingestion(
    collection_id: str,
    parquet_path: str,
    expected_count: int,
    endpoint_url: str = "http://localstack:4566"
):
    """Run bulk ingestion and measure throughput."""
    print()
    print("=" * 80)
    print("BULK INGESTION BENCHMARK")
    print("=" * 80)
    print(f"Collection ID: {collection_id}")
    print(f"Parquet file: {parquet_path}")
    print(f"Expected vectors: {expected_count:,}")
    print()
    
    # Initiate bulk insert
    payload = {
        "collection_name": collection_id,
        "parquet_paths": [parquet_path],
        "access_key": "test",
        "secret_key": "test",
        "endpoint_url": endpoint_url
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] Initiating bulk insert...")
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8888/v1/insert-data",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=600
    )
    
    if response.status_code not in [200, 202]:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        raise RuntimeError("Bulk insert failed")
    
    result = response.json()
    job_id = result.get("job_id")
    print(f"✓ Job initiated: {job_id}")
    print()
    
    # Monitor progress
    print(f"{'Elapsed (s)':<12} {'Count':<15} {'Progress (%)':<15} {'Rate (vec/s)':<15} {'Status':<20}")
    print("-" * 85)
    
    connections.connect(host='localhost', port='19530')
    coll = Collection(collection_id)
    
    last_count = 0
    last_time = start_time
    
    while True:
        elapsed = time.time() - start_time
        
        # Get count
        current_count = coll.num_entities
        progress_pct = (current_count / expected_count * 100) if expected_count > 0 else 0
        
        # Calculate instantaneous rate
        time_delta = time.time() - last_time
        count_delta = current_count - last_count
        rate = count_delta / time_delta if time_delta > 0 else 0
        
        # Get job status
        try:
            status_response = requests.get(f"http://localhost:8888/v1/job-status/{job_id}", timeout=5)
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status", "unknown")
            else:
                status = "unknown"
        except:
            status = "unknown"
        
        print(f"{elapsed:<12.1f} {current_count:<15,} {progress_pct:<15.1f} {rate:<15,.0f} {status:<20}")
        
        # Check completion
        if status == "completed" or current_count >= expected_count:
            print()
            print("=" * 80)
            print("✓ BULK INGESTION COMPLETED!")
            print("=" * 80)
            total_time = time.time() - start_time
            avg_throughput = current_count / total_time if total_time > 0 else 0
            print(f"Total vectors ingested: {current_count:,}")
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Average throughput: {avg_throughput:,.0f} vectors/second")
            print(f"Average throughput: {avg_throughput*60:,.0f} vectors/minute")
            print(f"Average throughput: {avg_throughput*3600:,.0f} vectors/hour")
            print("=" * 80)
            break
        
        if status == "failed":
            print()
            print(f"✗ Job failed")
            break
        
        last_count = current_count
        last_time = time.time()
        time.sleep(5)
    
    connections.disconnect("default")


def main():
    parser = argparse.ArgumentParser(description="Run L40 bulk ingestion benchmark")
    parser.add_argument(
        "--parquet-file",
        type=str,
        default="benchmark_data/embeddings_1m.parquet",
        help="Path to parquet file"
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=1_000_000,
        help="Expected number of vectors"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip S3 upload (file already uploaded)"
    )
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Skip collection creation (use existing)"
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        help="Existing collection ID (if --skip-create)"
    )
    
    args = parser.parse_args()
    
    # Upload to S3
    if not args.skip_upload:
        upload_to_s3(
            file_path=args.parquet_file,
            bucket="cosmos-test-bucket",
            key=f"benchmark_data/{args.parquet_file.split('/')[-1]}",
            endpoint_url="http://localhost:4566"
        )
    
    # Create collection
    if args.skip_create:
        if not args.collection_id:
            raise ValueError("--collection-id required when using --skip-create")
        collection_id = args.collection_id
    else:
        collection_id = create_collection()
    
    # Run benchmark
    s3_path = f"s3://cosmos-test-bucket/benchmark_data/{args.parquet_file.split('/')[-1]}"
    run_bulk_ingestion(
        collection_id=collection_id,
        parquet_path=s3_path,
        expected_count=args.num_vectors
    )


if __name__ == "__main__":
    main()

