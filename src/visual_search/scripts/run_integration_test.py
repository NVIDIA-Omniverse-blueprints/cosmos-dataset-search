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

"""Image ingestion script for visual search."""

import logging

from src.visual_search.client import Client
from src.visual_search.client.config import DEFAULT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestCases:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        logging.info(f"Running tests with profile {profile}")
        self.client = Client()

    def run(self) -> None:
        """Run tests."""

        pipeline = "c_radio_image_search_milvus"
        collection_name = "Test Collection"

        collection_id = None
        exception = None

        try:
            logging.info("Listing all pipelines")
            pipelines = self.client.pipelines.list(profile=self.profile)
            ids = {p["id"] for p in pipelines["pipelines"]}
            assert {pipeline}.issubset(ids)

            logging.info("Creating collection")
            collection = self.client.collections.create(
                pipeline=pipeline, profile=self.profile, name=collection_name
            )
            assert collection["collection"]["pipeline"] == pipeline
            assert collection["collection"]["name"] == collection_name
            collection_id = collection["collection"]["id"]

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

            logging.info("Searching on collection")
            results = self.client.search(
                collection_ids=[collection_id],
                text_query="picture of a car",
                top_k=2,
                profile=self.profile,
            )
            retrievals = results["retrievals"]
            assert len(retrievals) == 0
        except Exception as e:
            exception = e

        if collection_id is not None:
            logging.info("Deleting collection")
            delete_response = self.client.collections.delete(
                collection_id=collection_id, profile=self.profile
            )
            assert "deleted" in delete_response.get("message", "").lower()
            assert delete_response["id"] == collection_id

        if exception is not None:
            raise exception


if __name__ == "__main__":
    TestCases(profile=DEFAULT).run()
