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
import os

import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.bulk_writer import BulkFileType, LocalBulkWriter

# --- Configuration ---
DIM = 10  # Example dimension for vector fields, adjust as needed
OUTPUT_SUBDIR_NAME = "parquet_output"  # Subdirectory for generated Parquet files
INPUT_JSON_FILENAME = "json_example.json"  # Name of the external JSON file


def main():
    print("Starting JSON to Parquet conversion POC...")

    # Load JSON data
    input_json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), INPUT_JSON_FILENAME
    )
    try:
        with open(input_json_path, "r") as f:
            data_rows = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        return

    # Process data rows
    processed_rows = []
    for row in data_rows:
        if "id" in row:
            row["id"] = str(row["id"])  # Ensure ID is a string
        # Ensure embedding is a float32 list, handling cases where it might already be a list
        if "embedding" in row and isinstance(row["embedding"], list):
            row["embedding"] = np.array(row["embedding"], dtype=np.float32).tolist()
        processed_rows.append(row)

    if not processed_rows:
        print("No valid data rows to process.")
        return

    # Debug: Print processed rows
    print("Processed Rows:", processed_rows)

    # Define Milvus schema using FieldSchema and CollectionSchema directly.
    id_field = FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        max_length=255,
        is_primary=True,
        auto_id=False,
    )
    embedding_field = FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM
    )

    schema = CollectionSchema(
        fields=[id_field, embedding_field],
        description="Collection for visual search data",
        enable_dynamic_field=True,
    )

    # Debug: Print schema details
    print("Schema Fields:", schema.fields)
    print(
        f"Primary field in schema object: {[f.name for f in schema.fields if f.is_primary]}"
    )

    # Initialize LocalBulkWriter
    output_base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), OUTPUT_SUBDIR_NAME
    )
    os.makedirs(output_base_path, exist_ok=True)

    # Print the full path where files are expected to be generated ---
    print(f"Output directory created/checked at: {output_base_path}")

    writer = LocalBulkWriter(
        schema=schema, local_path=output_base_path, file_type=BulkFileType.PARQUET
    )

    # Append rows to writer
    for row in processed_rows:
        try:
            print(f"Appending row: {row}")
            writer.append_row(row)
        except Exception as e:
            print(f"Error appending row: {e}")

    # Commit data
    try:
        writer.commit()
        print("Data committed successfully.")
        # List files in the output directory after commit ---
        generated_files = os.listdir(output_base_path)
        if generated_files:
            print(f"Files generated in {output_base_path}: {generated_files}")
        else:
            print(
                f"No files found in {output_base_path} after commit. This might indicate an issue with data volume or writer configuration."
            )
    except Exception as e:
        print(f"Error during commit: {e}")


if __name__ == "__main__":
    main()
