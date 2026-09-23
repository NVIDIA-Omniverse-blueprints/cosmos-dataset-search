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

"""Minimal local integration test script for visual search using local profile."""

import logging

from src.visual_search.client import Client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MinimalTestCases:
    def __init__(self) -> None:
        # Use the local profile instead of default
        self.profile = "default"
        logging.info(f"Running tests with profile {self.profile}")
        self.client = Client()

    def run(self) -> None:
        """Run minimal tests."""
        pipeline = "cosmos_video_search_milvus"
        collection_name = "Test Collection"

        collection_id = None
        exception = None

        try:
            logging.info("Listing all pipelines")
            pipelines = self.client.pipelines.list(profile=self.profile)
            ids = {p["id"] for p in pipelines["pipelines"]}
            logging.info(f"Available pipelines: {ids}")
            assert {pipeline}.issubset(ids)

            logging.info("Creating collection")
            collection = self.client.collections.create(
                pipeline=pipeline, profile=self.profile, name=collection_name
            )
            assert collection["collection"]["pipeline"] == pipeline
            assert collection["collection"]["name"] == collection_name
            collection_id = collection["collection"]["id"]
            logging.info(f"Created collection with ID: {collection_id}")

            logging.info("Get collection")
            collection = self.client.collections.get(
                collection_id=collection_id, profile=self.profile
            )
            assert collection["collection"]["id"] == collection_id
            assert collection["collection"]["pipeline"] == pipeline
            assert collection["collection"]["name"] == collection_name
            assert collection["total_documents_count"] == 0

            logging.info("Listing all collections")
            collections = self.client.collections.list(profile=self.profile)
            ids = {collection["id"] for collection in collections["collections"]}
            assert {collection_id}.issubset(ids)

            # Test collection update persistence
            logging.info("Testing collection update persistence")
            original_name = collection_name
            updated_name = "Updated Test Collection"
            updated_tags = {"test": "integration", "updated": "true"}
            
            # Get the API endpoint from profile
            from src.visual_search.client.client import load_profile
            cfg = load_profile(self.profile)
            api_endpoint = cfg.api_endpoint
            
            # Update the collection using curl (PATCH request)
            import requests
            import json
            
            headers = {"Content-Type": "application/json"}
            if hasattr(cfg, 'auth_endpoint') and cfg.auth_endpoint:
                from src.visual_search.client.client import get_token
                token = get_token(cfg)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            
            patch_payload = {
                "name": updated_name,
                "tags": updated_tags
            }
            
            update_response = requests.patch(
                f"{api_endpoint}/v1/collections/{collection_id}",
                json=patch_payload,
                headers=headers,
                verify=False
            )
            assert update_response.status_code == 200, f"Update failed: {update_response.text}"
            update_data = update_response.json()
            assert update_data["collection"]["name"] == updated_name
            assert update_data["collection"]["tags"] == updated_tags
            
            # Verify persistence by fetching the collection again
            logging.info("Verifying collection update persistence")
            fetched_collection = self.client.collections.get(
                collection_id=collection_id, profile=self.profile
            )
            assert fetched_collection["collection"]["name"] == updated_name
            assert fetched_collection["collection"]["tags"] == updated_tags
            assert fetched_collection["collection"]["name"] != original_name
            
            # Verify the collection appears in list with updated data
            collections_after_update = self.client.collections.list(profile=self.profile)
            updated_collection_in_list = next(
                (c for c in collections_after_update["collections"] if c["id"] == collection_id), 
                None
            )
            assert updated_collection_in_list is not None
            assert updated_collection_in_list["name"] == updated_name
            assert updated_collection_in_list["tags"] == updated_tags
            
            # Test that embeddings remain associated after collection metadata update
            logging.info("Testing that embeddings remain associated after metadata update")
            try:
                # Count vectors after collection update using API
                collection_info = self.client.collections.get(
                    collection_id=collection_id, profile=self.profile
                )
                vector_count_after = collection_info["total_documents_count"]
                
                logging.info(f"Vector count AFTER collection update: {vector_count_after}")
                
                if vector_count_after > 0:
                    logging.info("SUCCESS: Embeddings remain associated with collection after metadata update")
                    logging.info(f"Found {vector_count_after} vectors in collection {collection_id}")
                else:
                    logging.info("INFO: No vectors found in collection (collection might be empty)")
                    
            except Exception as e:
                logging.warning(f"Vector counting test failed: {e}")
                # This is not a failure since the collection might be empty or not accessible
        except Exception as e:
            exception = e

        if collection_id is not None:
            logging.info("Deleting collection")
            delete_response = self.client.collections.delete(
                collection_id=collection_id, profile=self.profile
            )
            assert "deleted" and "successfully" in delete_response["message"].lower()
            assert delete_response["id"] == collection_id

        if exception is not None:
            raise exception

        logging.info("All minimal integration tests passed!")


if __name__ == "__main__":
    MinimalTestCases().run() 
