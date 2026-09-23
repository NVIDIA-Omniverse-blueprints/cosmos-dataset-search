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
Unit tests for curator_parquet_converter.py
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.visual_search.v1.apis.utils.curator_parquet_converter import (
    ParquetConverterError,
    process_directory,
    process_single_file,
)


class TestCuratorParquetConverter(unittest.TestCase):
    def setUp(self):
        """Set up test data and temporary directories"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_dir = os.path.join(self.temp_dir, "iv2_embd_parquet")
        self.metas_dir = os.path.join(self.temp_dir, "metas", "v0")
        os.makedirs(self.parquet_dir)
        os.makedirs(self.metas_dir)

        # Create test data
        self.test_uuid = "test-uuid-123"
        self.test_embedding = np.random.rand(512).tolist()  # Example 512-dim embedding

        # Create test parquet file
        self.parquet_path = os.path.join(self.parquet_dir, "test.parquet")
        pd.DataFrame(
            {"id": [self.test_uuid], "embedding": [self.test_embedding]}
        ).to_parquet(self.parquet_path)

        # Create test metadata file
        self.metadata = {
            "span_uuid": self.test_uuid,
            "source_video": "test_video.mp4",
            "duration_span": 5.0,
            "clip_location": {"start": 0, "end": 5},
            "windows": [{"description": "test scene"}],
        }
        self.metadata_path = os.path.join(self.metas_dir, f"{self.test_uuid}.json")
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_process_single_file_local(self):
        """Test processing a single local file"""
        result = process_single_file(self.parquet_path, self.temp_dir, "local")

        # Check the result structure
        self.assertEqual(len(result), 1)
        self.assertEqual(result["id"].iloc[0], self.test_uuid)
        # Compare embeddings as lists since they're numpy arrays
        self.assertTrue(
            np.array_equal(result["embedding"].iloc[0], self.test_embedding)
        )

        # Check metadata
        loaded_meta = json.loads(result["$meta"].iloc[0])
        self.assertEqual(loaded_meta["span_uuid"], self.test_uuid)
        self.assertEqual(loaded_meta["source_video"], "test_video.mp4")

    @patch("pandas.read_parquet")
    @patch("s3fs.S3FileSystem")
    def test_process_single_file_s3(self, mock_s3fs, mock_read_parquet):
        """Test processing a single S3 file"""
        # Mock S3 file system
        mock_s3 = MagicMock()
        mock_s3fs.return_value = mock_s3

        # Mock parquet reading
        mock_read_parquet.return_value = pd.DataFrame(
            {"id": [self.test_uuid], "embedding": [self.test_embedding]}
        )

        # Mock metadata file reading
        mock_meta_file = MagicMock()
        mock_meta_file.__enter__.return_value.read.return_value = json.dumps(
            self.metadata
        )
        mock_s3.open.return_value = mock_meta_file

        result = process_single_file(
            "s3://bucket/path/test.parquet", "s3://bucket/path/", "s3"
        )

        # Check the result structure
        self.assertEqual(len(result), 1)
        self.assertEqual(result["id"].iloc[0], self.test_uuid)
        self.assertTrue(
            np.array_equal(result["embedding"].iloc[0], self.test_embedding)
        )

        # Check metadata
        loaded_meta = json.loads(result["$meta"].iloc[0])
        self.assertEqual(loaded_meta["span_uuid"], self.test_uuid)

    def test_process_directory_local(self):
        """Test processing a local directory"""
        output_path = os.path.join(self.temp_dir, "output.parquet")
        process_directory(self.temp_dir, output_path, "local")

        # Check if output file was created
        self.assertTrue(os.path.exists(output_path))

        # Load and verify output
        result = pd.read_parquet(output_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["id"].iloc[0], self.test_uuid)

    @patch("pandas.read_parquet")
    @patch("s3fs.S3FileSystem")
    def test_process_directory_s3(self, mock_s3fs, mock_read_parquet):
        """Test processing an S3 directory"""
        # Mock S3 file system
        mock_s3 = MagicMock()
        mock_s3fs.return_value = mock_s3

        # Mock parquet reading
        mock_read_parquet.return_value = pd.DataFrame(
            {"id": [self.test_uuid], "embedding": [self.test_embedding]}
        )

        # Mock directory listing
        mock_s3.ls.return_value = ["s3://bucket/path/iv2_embd_parquet/test.parquet"]
        mock_s3.glob.return_value = ["s3://bucket/path/iv2_embd_parquet/test.parquet"]

        # Mock metadata file reading
        mock_meta_file = MagicMock()
        mock_meta_file.__enter__.return_value.read.return_value = json.dumps(
            self.metadata
        )
        mock_s3.open.return_value = mock_meta_file

        output_path = os.path.join(self.temp_dir, "output.parquet")
        process_directory("s3://bucket/path/", output_path, "s3")

        # Check if output file was created
        self.assertTrue(os.path.exists(output_path))

        # Load and verify output
        result = pd.read_parquet(output_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["id"].iloc[0], self.test_uuid)

    def test_uuid_mismatch(self):
        """Test handling of UUID mismatch between parquet and metadata"""
        # Create metadata with different UUID
        with open(self.metadata_path, "w") as f:
            json.dump({**self.metadata, "span_uuid": "different-uuid"}, f)

        with self.assertRaises(ParquetConverterError) as context:
            process_single_file(self.parquet_path, self.temp_dir, "local")
        self.assertIn("UUID mismatch", str(context.exception))


if __name__ == "__main__":
    unittest.main()
