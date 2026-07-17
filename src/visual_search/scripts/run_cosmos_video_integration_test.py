# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import sleep, monotonic
from uuid import UUID
import boto3
import uuid
from src.visual_search.client import Client
from src.visual_search.client.config import DEFAULT  # we will override with "local" profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure boto3 inside Ray workers uses LocalStack
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ["AWS_ENDPOINT_URL"] = "http://localstack:4566"

SAMPLE_VIDEO_DIR = Path(__file__).parent / "sample_data"
SAMPLE_VIDEO_FILE = SAMPLE_VIDEO_DIR / "video1.mp4"

PIPELINE_ID = "cosmos_video_search_milvus"
COLLECTION_NAME = "Cosmos Video Integration Test"

INGEST_TIMEOUT_S = 300


PROFILE = "default"

def _wait_for_document_count(client: Client, collection_id: str, expected: int) -> None:
    """Poll the collection endpoint until total_documents_count == expected."""
    start = monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            col = client.collections.get(collection_id=collection_id, profile=PROFILE)
            count = col["total_documents_count"]
            if attempt == 1 or attempt % 10 == 0:  # Log every 10th attempt to reduce noise
                logging.info("Attempt %d: Current document count: %s (waiting for %s)", attempt, count, expected)
            if count >= expected:
                logging.info("Collection now contains %s documents after %d attempts", count, attempt)
                return
        except Exception as e:
            logging.error("Error checking collection status on attempt %d: %s", attempt, e)
        
        if monotonic() - start > INGEST_TIMEOUT_S:
            logging.error("Timeout after %d attempts - Collection data: %s", attempt, col)
            raise TimeoutError(f"Timed-out waiting for {expected} docs (currently {count}) after {attempt} attempts")
        sleep(5)


def main() -> None:
    if not SAMPLE_VIDEO_FILE.exists():
        raise FileNotFoundError(f"Sample video not found at {SAMPLE_VIDEO_FILE}")

    client = Client()
    collection_id: str | None = None
    exception: Exception | None = None

    try:
        logging.info("Listing pipelines – expecting %s", PIPELINE_ID)
        pipelines = client.pipelines.list(profile=PROFILE)
        ids = {p["id"] for p in pipelines["pipelines"]}
        assert PIPELINE_ID in ids, f"Pipeline {PIPELINE_ID} missing in {ids}"

        logging.info("Creating collection")
        col_resp = client.collections.create(pipeline=PIPELINE_ID, name=COLLECTION_NAME, profile=PROFILE)
        collection_id = col_resp["collection"]["id"]
        logging.info("Collection ID: %s", collection_id)

        logging.info("Ensuring sample video is present in LocalStack S3")
        s3_endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localstack:4566")
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        bucket = "cosmos-test-bucket"
        try:
            s3.create_bucket(Bucket=bucket)
        except s3.exceptions.BucketAlreadyOwnedByYou:
            pass
        key_prefix = "cosmos-tests"
        key = f"{key_prefix}/video1.mp4"
        s3.upload_file(Filename=str(SAMPLE_VIDEO_FILE), Bucket=bucket, Key=key)
        s3_path = f"s3://{bucket}/{key}"
        logging.info(f"s3_path: {s3_path}")
        logging.info("Uploaded sample video to %s (key %s)", bucket, key)
        logging.info("AWS_ENDPOINT_URL=%s", os.environ.get("AWS_ENDPOINT_URL"))
        ingest_stats = client.ingest.files(
            directory_path=s3_path,
            collection_id=collection_id,
            num_workers=1,
            batch_size=1,
            extensions=[".mp4"],
            profile=PROFILE,
            s3_profile=None,
            timeout=300,
        )
        logging.info("Ingest completed with stats: %s", ingest_stats)
        assert ingest_stats.get(200, 0) >= 1, f"Ingest returned stats {ingest_stats}"

        logging.info("Forcing collection flush so that documents are immediately searchable")
        from src.visual_search.client.client import load_profile, get_headers
        import requests
        cfg = load_profile(PROFILE)
        headers = get_headers(cfg)
        flush_resp = requests.post(
            f"{cfg.api_endpoint}/v1/admin/collections/{collection_id}/flush",
            headers=headers,
            verify=False,
        )
        flush_resp.raise_for_status()
        logging.info("Flush successful: %s", flush_resp.json())

        logging.info("Waiting for document to be indexed in Milvus")
        _wait_for_document_count(client, collection_id, expected=1)

        logging.info("Running text search query")
        results = client.search(
            collection_ids=[collection_id],
            text_query="person singing in mic",
            top_k=1,
            profile=PROFILE,
        )
        retrievals = results["retrievals"]
        assert len(retrievals) == 1, f"Expected 1 retrieval, got {len(retrievals)}"  # should at least return the single doc
        logging.info("Search successful – retrieved document ID %s", retrievals[0]["id"])

    except Exception as exc:
        exception = exc
    finally:
        if collection_id:
            try:
                logging.info("Cleaning up – deleting collection %s", collection_id)
                client.collections.delete(collection_id=collection_id, profile=PROFILE)
            except Exception as cleanup_exc:
                logging.warning("Failed to delete collection: %s", cleanup_exc)
        if exception:
            raise exception


if __name__ == "__main__":
    main()
