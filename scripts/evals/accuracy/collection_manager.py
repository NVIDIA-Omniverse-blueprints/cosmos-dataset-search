#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

LOGGER = logging.getLogger("collection-manager")

class CVDSCollectionManager:
    
    def __init__(self, client, profile: str = "local"):
        self.client = client
        self.profile = profile
        self._created_collections: List[str] = []
    
    def create_collection(
        self,
        pipeline_id: str,
        dataset_name: str
    ) -> str:
        collection_name = dataset_name
        
        LOGGER.info("Creating collection: %s (pipeline: %s)", collection_name, pipeline_id)
        try:
            response = self.client.collections.create(
                pipeline=pipeline_id,
                name=collection_name,
                profile=self.profile
            )
            collection_id = response["collection"]["id"]
            self._created_collections.append(collection_id)
            LOGGER.info("Successfully created collection: %s", collection_id)
            return collection_id
        except Exception as e:
            LOGGER.error("Failed to create collection: %s", e)
            raise
    
    def delete_collection(self, collection_id: str) -> bool:
        try:
            LOGGER.info("Deleting collection: %s", collection_id)
            self.client.collections.delete(collection_id=collection_id, profile=self.profile)
            if collection_id in self._created_collections:
                self._created_collections.remove(collection_id)
            LOGGER.info("Successfully deleted collection: %s", collection_id)
            return True
        except Exception as e:
            LOGGER.warning("Failed to delete collection %s: %s", collection_id, e)
            return False
    
    def cleanup_all_collections(self) -> None:
        if not self._created_collections:
            LOGGER.info("No collections to clean up")
            return
        LOGGER.info("Cleaning up %d collections", len(self._created_collections))
        for collection_id in self._created_collections.copy():
            self.delete_collection(collection_id)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all_collections()

class CVDSIngestionManager:
    
    def __init__(self, client, profile: str = "local"):
        self.client = client
        self.profile = profile
    
    def ingest_videos(
        self,
        collection_id: str,
        s3_path: str,
        metadata_json_path: Path,
        *,
        num_workers: int = 4,
        batch_size: int = 1,
        timeout: int = 600,
        extensions: List[str] = None
    ) -> Dict[str, int]:
        if extensions is None:
            extensions = [".mp4", ".mov", ".mkv", ".avi"]
        LOGGER.info("Ingesting videos into collection %s", collection_id)
        LOGGER.info("  S3 path: %s", s3_path)
        LOGGER.info("  Metadata: %s", metadata_json_path)
        LOGGER.info("  Workers: %d, Batch size: %d", num_workers, batch_size)
        try:
            stats = self.client.ingest.files(
                directory_path=s3_path,
                collection_id=collection_id,
                num_workers=num_workers,
                batch_size=batch_size,
                metadata_json=str(metadata_json_path),
                strip_directory_path=True,
                extensions=extensions,
                profile=self.profile,
                s3_profile=None,
                timeout=timeout,
                existence_check="skip",
            )
            LOGGER.info("Ingestion completed. Status codes: %s", dict(stats))
            return dict(stats)
            
        except Exception as e:
            LOGGER.error("Ingestion failed: %s", e)
            raise
    
    def flush_collection(self, collection_id: str) -> bool:
        try:
            from src.visual_search.client.client import load_profile, get_headers
            LOGGER.info("Flushing collection: %s", collection_id)
            cfg = load_profile(self.profile)
            headers = get_headers(cfg)
            response = requests.post(
                f"{cfg.api_endpoint}/v1/admin/collections/{collection_id}/flush",
                headers=headers,
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            LOGGER.info("Successfully flushed collection: %s", collection_id)
            return True
        except Exception as e:
            LOGGER.error("Failed to flush collection %s: %s", collection_id, e)
            return False
    
    def wait_for_ingestion(
        self,
        collection_id: str,
        expected_count: int,
        max_wait_time: int = 300,
        check_interval: int = 10
    ) -> bool:
        LOGGER.info("Waiting for ingestion to complete (expected: %d documents)", expected_count)
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                LOGGER.debug("Checking ingestion progress...")
                time.sleep(check_interval)
                if time.time() - start_time > 30:
                    LOGGER.info("Ingestion wait period completed")
                    return True
            except Exception as e:
                LOGGER.warning("Error checking ingestion status: %s", e)
                time.sleep(check_interval)
        
        LOGGER.warning("Ingestion wait timeout after %d seconds", max_wait_time)
        return False

class CVDSSearchManager:
    
    def __init__(self, client, profile: str = "local"):
        self.client = client
        self.profile = profile
    
    def search(
        self,
        collection_id: str,
        text_query: str,
        top_k: int = 10
    ) -> List[Dict]:
        try:
            response = self.client.search(
                collection_ids=[collection_id],
                text_query=text_query,
                top_k=top_k,
                profile=self.profile
            )
            
            retrieved_docs = response.get("retrievals", [])
            LOGGER.debug("Search query: '%s' -> %d results", text_query, len(retrieved_docs))
            return retrieved_docs
            
        except Exception as e:
            LOGGER.error("Search failed for query '%s': %s", text_query, e)
            raise
    
    def create_search_function(self, collection_id: str):
        def search_function(text_query: str, top_k: int = 10) -> List[Dict]:
            return self.search(collection_id, text_query, top_k)
        return search_function

class CVDSPipelineValidator:
    
    def __init__(self, client, profile: str = "local"):
        self.client = client
        self.profile = profile
    
    def validate_pipeline(self, pipeline_id: str) -> bool:
        try:
            LOGGER.info("Validating pipeline: %s", pipeline_id)
            response = self.client.pipelines.list(profile=self.profile)
            available_pipelines = {p["id"] for p in response["pipelines"]}
            if pipeline_id in available_pipelines:
                LOGGER.info("Pipeline %s is available", pipeline_id)
                return True
            else:
                LOGGER.error("Pipeline %s not available. Available: %s", 
                           pipeline_id, sorted(available_pipelines))
                return False
        except Exception as e:
            LOGGER.error("Failed to validate pipeline %s: %s", pipeline_id, e)
            return False
    
    def list_available_pipelines(self) -> List[str]:
        try:
            response = self.client.pipelines.list(profile=self.profile)
            pipeline_ids = [p["id"] for p in response["pipelines"]]
            LOGGER.info("Available pipelines: %s", pipeline_ids)
            return pipeline_ids
        except Exception as e:
            LOGGER.error("Failed to list pipelines: %s", e)
            return []

def create_collection_lifecycle_manager(client, profile: str = "local"):
    collection_manager = CVDSCollectionManager(client, profile)
    ingestion_manager = CVDSIngestionManager(client, profile)
    search_manager = CVDSSearchManager(client, profile)
    validator = CVDSPipelineValidator(client, profile)
    return collection_manager, ingestion_manager, search_manager, validator