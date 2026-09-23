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

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import boto3
from tqdm import tqdm

from dataset_loader import DatasetRecord

LOGGER = logging.getLogger("video-manager")

class S3VideoManager:
    def __init__(
        self, 
        bucket: str, 
        endpoint_url: str,
        *,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        region_name: str = None
    ):
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID", "test")
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
        self.region_name = region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )
        return self._client
    
    def ensure_bucket_exists(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            LOGGER.info("Bucket %s already exists", self.bucket)
        except Exception:
            LOGGER.info("Creating bucket %s", self.bucket)
            try:
                if self.region_name == "us-east-1":
                    self.client.create_bucket(Bucket=self.bucket)
                else:
                    self.client.create_bucket(
                        Bucket=self.bucket, 
                        CreateBucketConfiguration={"LocationConstraint": self.region_name}
                    )
                LOGGER.info("Successfully created bucket %s", self.bucket)
            except Exception as e:
                LOGGER.error("Failed to create bucket %s: %s", self.bucket, e)
                raise
    
    def upload_unique_videos(
        self, 
        records: List[DatasetRecord], 
        prefix: str = "dataset"
    ) -> Tuple[str, Dict[str, str]]:
        self.ensure_bucket_exists()
        
        uploaded: Dict[str, str] = {}
        seen_files: Set[str] = set()
        
        LOGGER.info("Uploading videos to s3://%s/%s", self.bucket, prefix)
        
        for record in tqdm(records, desc="Uploading videos"):
            video_id, video_path, _ = record
            
            if video_path in seen_files:
                continue
            seen_files.add(video_path)
            
            src = Path(video_path)
            if not src.exists():
                LOGGER.warning("Video path not found locally, skipping: %s", video_path)
                continue
                
            key = f"{prefix}/{src.name}"
            if self._object_exists(key):
                uploaded[key] = video_id
                continue
                
            try:
                self.client.upload_file(str(src), self.bucket, key)
                uploaded[key] = video_id
                LOGGER.debug("Uploaded %s -> s3://%s/%s", src.name, self.bucket, key)
            except Exception as e:
                LOGGER.error("Failed to upload %s: %s", src, e)
                continue

        LOGGER.info("Uploaded %d unique videos to s3://%s/%s", len(uploaded), self.bucket, prefix)
        return prefix, uploaded
    
    def _object_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "404":
                return False
            raise

class MetadataGenerator:
    
    @staticmethod
    def build_metadata_json(
        s3_mapping: Dict[str, str], 
        output_path: Path
    ) -> Path:
        mapping: Dict[str, Dict[str, str]] = {}
        
        for key, video_id in s3_mapping.items():
            rel_name = Path(key).name
            mapping[rel_name] = {"video_id": video_id}
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as fp:
            json.dump(mapping, fp, indent=2)
            
        LOGGER.info("Created metadata file with %d entries: %s", len(mapping), output_path)
        return output_path

class VideoPathResolver:
    
    @staticmethod
    def validate_records(records: List[DatasetRecord]) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
        valid_records = []
        invalid_records = []
        
        for record in records:
            video_id, video_path, caption = record
            
            if Path(video_path).exists():
                valid_records.append(record)
            else:
                invalid_records.append(record)
                LOGGER.warning("Video file not found: %s (ID: %s)", video_path, video_id)
        
        LOGGER.info("Validated %d records: %d valid, %d invalid", 
                   len(records), len(valid_records), len(invalid_records))
        
        return valid_records, invalid_records
    
    @staticmethod
    def get_unique_videos(records: List[DatasetRecord]) -> List[DatasetRecord]:
        seen_paths = set()
        unique_records = []
        
        for record in records:
            video_id, video_path, caption = record
            
            if video_path not in seen_paths:
                seen_paths.add(video_path)
                unique_records.append(record)
        
        LOGGER.info("Found %d unique videos from %d total records", 
                   len(unique_records), len(records))
        
        return unique_records

class VideoIngestionPipeline:
    
    def __init__(self, s3_manager: S3VideoManager):
        self.s3_manager = s3_manager
        self.metadata_generator = MetadataGenerator()
        self.path_resolver = VideoPathResolver()
    
    def prepare_for_ingestion(
        self,
        records: List[DatasetRecord],
        dataset_name: str,
        metadata_output_dir: Path,
        *,
        limit: int = None
    ) -> Tuple[str, Path, List[DatasetRecord]]:
        LOGGER.info("Preparing %d records for ingestion (dataset: %s)", 
                   len(records), dataset_name)
        valid_records, invalid_records = self.path_resolver.validate_records(records)
        
        if invalid_records:
            LOGGER.warning("Found %d invalid video paths", len(invalid_records))
        
        if not valid_records:
            raise ValueError("No valid video records found")
        if limit and limit > 0:
            query_records = valid_records[:limit]
            limited_video_ids = {record.video_id for record in query_records}
            records_for_upload = [r for r in valid_records if r.video_id in limited_video_ids]
            LOGGER.info("Limited to %d queries, uploading %d unique videos", 
                       len(query_records), len(records_for_upload))
        else:
            query_records = valid_records
            records_for_upload = valid_records
        unique_video_records = self.path_resolver.get_unique_videos(records_for_upload)
        prefix, s3_mapping = self.s3_manager.upload_unique_videos(
            unique_video_records, 
            prefix=dataset_name
        )
        metadata_json_path = metadata_output_dir / f"{dataset_name}_metadata.json"
        self.metadata_generator.build_metadata_json(s3_mapping, metadata_json_path)
        s3_path = f"s3://{self.s3_manager.bucket}/{prefix}"
        
        LOGGER.info("Video ingestion preparation complete:")
        LOGGER.info("  S3 path: %s", s3_path)
        LOGGER.info("  Metadata: %s", metadata_json_path)
        LOGGER.info("  Query records: %d", len(query_records))
        LOGGER.info("  Uploaded videos: %d", len(s3_mapping))
        
        return s3_path, metadata_json_path, query_records

def create_s3_manager(
    bucket: str = "cosmos-test-bucket",
    endpoint_url: str = "http://localstack:4566"
) -> S3VideoManager:
    LOGGER.info("Creating S3 manager with bucket: %s, endpoint: %s", bucket, endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint_url)
    return S3VideoManager(bucket, endpoint_url)