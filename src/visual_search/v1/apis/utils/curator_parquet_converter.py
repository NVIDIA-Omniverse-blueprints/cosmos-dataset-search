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
Script to convert Curator parquet files with their metadata into the desired format.
Supports both local filesystem and S3 storage.
"""

import json
import os

import pandas as pd
import s3fs

from src.visual_search.logger import logger


class ParquetConverterError(Exception):
    """Base exception for parquet conversion errors."""

    pass


def process_single_file(
    parquet_path: str, base_dir: str, source_type: str = "local"
) -> pd.DataFrame:
    """
    Process a single parquet file and its related metadata.

    Args:
        parquet_path: Path to the parquet file
        base_dir: Base directory containing all the data
        source_type: Either "local" or "s3"

    Returns:
        DataFrame containing the processed data with columns:
            - id: UUID of the span
            - embedding: Vector embedding
            - $meta: JSON string containing metadata
    """
    # Read the parquet file
    if source_type == "s3":
        s3 = s3fs.S3FileSystem()
        df = pd.read_parquet(f"s3://{parquet_path}")
    else:
        df = pd.read_parquet(parquet_path)

    # Get the id from the parquet file
    uuid = df["id"].iloc[0]

    # Construct path to metadata file
    if source_type == "s3":
        metadata_path = f"{base_dir}/metas/v0/{uuid}.json"
        s3 = s3fs.S3FileSystem()
        with s3.open(metadata_path[5:], "r") as f:  # Remove 's3://' prefix
            metadata = json.load(f)
    else:
        metadata_path = os.path.join(base_dir, "metas", "v0", f"{uuid}.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Verify the UUID matches
    if metadata["span_uuid"] != uuid:
        raise ParquetConverterError(
            f"UUID mismatch: parquet file has {uuid}, metadata has {metadata['span_uuid']}"
        )

    # Create new DataFrame with combined data
    return pd.DataFrame(
        {
            "id": [uuid],
            "embedding": [df["embedding"].iloc[0]],
            "$meta": [json.dumps(metadata)],
        }
    )


def process_directory(
    base_dir: str, output_path: str, source_type: str = "local"
) -> None:
    """
    Process all parquet files in the base directory.

    Args:
        base_dir: Base directory containing all data
        output_path: Where to save the output parquet file
        source_type: Either "local" or "s3"
    """
    try:
        # Find all parquet files in the iv2_embd_parquet directory
        if source_type == "s3":
            parquet_dir = f"{base_dir}/iv2_embd_parquet"
            s3 = s3fs.S3FileSystem()
            logger.info(f"Looking for parquet files in: {parquet_dir}")
            parquet_files = s3.glob(f"{parquet_dir[5:]}/**/*.parquet")
        else:
            parquet_dir = os.path.join(base_dir, "iv2_embd_parquet")
            logger.info(f"Looking for parquet files in: {parquet_dir}")
            parquet_files = [
                f for f in os.listdir(parquet_dir) if f.endswith(".parquet")
            ]

        if not parquet_files:
            logger.warning("No parquet files found!")
            return

        logger.info(f"Found {len(parquet_files)} parquet files")

        # Process each file and combine results
        dfs = []
        for file in parquet_files:
            try:
                if source_type == "s3":
                    full_path = file
                else:
                    full_path = os.path.join(parquet_dir, file)
                logger.info(f"Processing {full_path}")
                df = process_single_file(full_path, base_dir, source_type)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                continue

        if not dfs:
            logger.warning("No files were successfully processed")
            return

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Save to output path
        combined_df.to_parquet(output_path, index=False)
        logger.info(
            f"Successfully processed {len(dfs)} files and saved to {output_path}"
        )
    except Exception as e:
        raise ParquetConverterError(f"Error processing directory: {str(e)}")


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        int: 0 on success, 1 on error
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Curator parquet files with metadata"
    )
    parser.add_argument(
        "base_dir", help="Base directory containing all data (local path or S3 URI)"
    )
    parser.add_argument("output_path", help="Where to save the output parquet file")
    parser.add_argument(
        "--source-type",
        choices=["local", "s3"],
        default="local",
        help="Source type: local directory or S3",
    )

    try:
        args = parser.parse_args()

        # Validate paths
        if args.source_type == "s3" and not args.base_dir.startswith("s3://"):
            raise ValueError(
                "base_dir must be an S3 URI (starting with s3://) when source-type is s3"
            )

        process_directory(args.base_dir, args.output_path, args.source_type)
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
