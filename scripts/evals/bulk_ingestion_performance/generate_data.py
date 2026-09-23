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
Generate benchmark parquet files for Milvus bulk ingestion testing.

This script generates parquet files with embeddings in float32 format,
compatible with Milvus 2.4.4-gpu bulk import.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def generate_embeddings_parquet(
    output_path: str,
    num_vectors: int = 1_000_000,
    embedding_dim: int = 256,
    batch_size: int = 100_000,
):
    """
    Generate a parquet file with embeddings for bulk ingestion.
    
    Args:
        output_path: Output file path
        num_vectors: Number of vectors to generate
        embedding_dim: Dimension of embeddings
        batch_size: Batch size for writing (memory management)
    """
    print(f"Generating {num_vectors:,} embeddings ({embedding_dim}D)")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size:,}")
    print()
    
    # Define schema - MUST use float32 for Milvus 2.4.4 compatibility
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),  # float32!
        pa.field("$meta", pa.string()),
    ])
    
    start_time = time.time()
    writer = None
    
    try:
        for batch_start in range(0, num_vectors, batch_size):
            batch_end = min(batch_start + batch_size, num_vectors)
            batch_count = batch_end - batch_start
            
            # Generate IDs
            ids = [f"vec_{i:08d}" for i in range(batch_start, batch_end)]
            
            # Generate normalized embeddings (float32)
            embeddings = np.random.randn(batch_count, embedding_dim).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
            # Generate metadata
            metadata = [f'{{"index": {i}}}' for i in range(batch_start, batch_end)]
            
            # Create PyArrow arrays
            id_array = pa.array(ids, type=pa.string())
            embedding_array = pa.array(
                [emb.tolist() for emb in embeddings],
                type=pa.list_(pa.float32())
            )
            meta_array = pa.array(metadata, type=pa.string())
            
            # Create batch
            batch = pa.RecordBatch.from_arrays(
                [id_array, embedding_array, meta_array],
                schema=schema
            )
            
            # Write batch
            if writer is None:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, schema, compression='snappy')
            
            writer.write_batch(batch)
            
            # Progress update
            if batch_end % 100_000 == 0 or batch_end == num_vectors:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                print(f"  {batch_end:,}/{num_vectors:,} ({100*batch_end/num_vectors:.1f}%) - {rate:,.0f} vec/s")
    
    finally:
        if writer:
            writer.close()
    
    duration = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    print()
    print(f"✓ Generated {num_vectors:,} embeddings in {duration:.2f}s")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Average rate: {num_vectors/duration:,.0f} vectors/sec")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark parquet data")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_data/embeddings_1m.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=1_000_000,
        help="Number of vectors to generate"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for writing"
    )
    
    args = parser.parse_args()
    
    generate_embeddings_parquet(
        output_path=args.output,
        num_vectors=args.num_vectors,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

