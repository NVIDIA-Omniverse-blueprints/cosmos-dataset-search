# Performance Benchmarks

## Overview

This document provides comprehensive performance benchmarks for the Cosmos Video Dataset Search (CVDS) system, covering three key performance areas:

1. **Bulk Ingestion Performance**: GPU-accelerated embedding ingestion using Milvus GPU_CAGRA indexing
2. **Video Ingestion Performance**: End-to-end video processing and embedding generation
3. **Text Search Latency Performance**: Query response times for semantic search

All benchmarks were conducted on NVIDIA L40 GPU hardware with optimized Milvus configurations.

> **Note:** The benchmark uses the optimized Milvus configuration from [`deploy/standalone/milvus_l40_standalone_optimized.yaml`](../../deploy/standalone/milvus_l40_standalone_optimized.yaml).
To apply this configuration, ensure it is mounted in [`deploy/standalone/docker-compose.build.yml`](../../deploy/standalone/docker-compose.build.yml) under the `milvus` service (using a `volumes` entry). For example:

```yaml
services:
  milvus:
    # ...
    volumes:
      - ./deploy/standalone/milvus_l40_standalone_optimized.yaml:/milvus/configs/milvus.yaml
```

This ensures the Milvus instance uses the performance-optimized configuration for the benchmarks described below.


## Reference Documentation

For more information about the technologies used in these benchmarks:

- **GPU_CAGRA Index**: [Milvus GPU_CAGRA Documentation](https://milvus.io/docs/gpu-cagra.md) - Official documentation for GPU_CAGRA index configuration and usage
- **GPU_CAGRA Technical Overview**: [Milvus Introduces GPU Index CAGRA](https://zilliz.com/blog/Milvus-introduces-GPU-index-CAGRA) - Technical deep dive into CAGRA architecture and performance
- **Milvus GPU Configuration**: [Install Milvus Standalone with Docker Compose (GPU)](https://milvus.io/docs/install_standalone-docker-compose-gpu.md) - GPU setup and configuration guide
- **CAGRA Parameter Tuning**: [cuVS Automated Tuning Guide](https://docs.rapids.ai/api/cuvs/nightly/tuning_guide/) - Guide for automated hyper-parameter optimization (HPO) of CAGRA graph parameters using GPU-accelerated tuning
- **Comparing Vector Indexes**: [cuVS Comparing Performance of Vector Indexes](https://docs.rapids.ai/api/cuvs/nightly/comparing_indexes/) - Methodology for recall-aware performance comparison and benchmarking of vector search indexes

## Test Environment

### Hardware Configuration

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA L40 (48GB VRAM) |
| **CPU** | AMD EPYC 7232P 8-Core Processor (16 threads) |
| **RAM** | 126GB DDR4 |
| **Storage** | 1.7TB NVMe SSD |
| **OS** | Ubuntu 22.04 LTS (Linux 5.15.0-153) |

### Software Stack

| Component | Version |
|-----------|---------|
| **Milvus** | 2.4.4-gpu (standalone mode) |
| **CUDA** | 13.0 |
| **NVIDIA Driver** | 580.65.06 |
| **Index Type** | GPU_CAGRA |
| **Storage Backend** | LocalStack S3 (development) |

---

## 1. Bulk Ingestion Performance

### Overview

Bulk ingestion benchmarks measure the throughput of loading pre-computed embeddings into Milvus with GPU_CAGRA indexing. This is useful for initial dataset loading or batch updates.

### How to Run Bulk Ingestion Benchmarks

#### Step 1: Generate Embedding Data

```bash
source .venv/bin/activate
```

Use the `generate_data.py` script to create synthetic embedding datasets in Parquet format:

```bash
# Generate 10M vectors (256-dim, float32)
python3 scripts/evals/bulk_ingestion_performance/generate_data.py \
  --output benchmark_data/embeddings_10m.parquet \
  --num-vectors 10000000 \
  --embedding-dim 256 \
  --batch-size 250000
```

**Parameters:**
- `--output`: Output parquet file path
- `--num-vectors`: Number of vectors to generate (default: 1M)
- `--embedding-dim`: Embedding dimension (default: 256)
- `--batch-size`: Batch size for writing (default: 100K)

**Output:** A parquet file with schema:
- `id` (string): Unique vector ID
- `embedding` (list of float32): Normalized embedding vector
- `$meta` (string): JSON metadata

#### Step 2: Run the Benchmark

Use the `run_benchmark.py` script to upload data to S3 and measure ingestion throughput:

```bash
# Full benchmark (upload to localstack + create collection + ingest)
python3 scripts/evals/bulk_ingestion_performance/run_benchmark.py \
  --parquet-file benchmark_data/embeddings_10m.parquet \
  --num-vectors 10000000

# Skip upload if data already in S3
python3 scripts/evals/bulk_ingestion_performance/run_benchmark.py \
  --parquet-file benchmark_data/embeddings_10m.parquet \
  --num-vectors 10000000 \
  --skip-upload

# Use existing collection
python3 scripts/evals/bulk_ingestion_performance/run_benchmark.py \
  --parquet-file benchmark_data/embeddings_10m.parquet \
  --num-vectors 10000000 \
  --skip-upload \
  --skip-create \
  --collection-id <collection_id>
```

**The benchmark will:**
1. Upload parquet file to LocalStack S3 (if not skipped)
2. Create a GPU_CAGRA collection (if not skipped)
3. Initiate bulk insert via `/v1/insert-data` API
4. Monitor progress every 5 seconds
5. Report final throughput metrics

#### Step 3: Configuration

**Key Settings:**
- **GPU Memory Pool**: 16GB init / 30GB max (optimized for L40 48GB VRAM)
- **Segment Size**: 2GB (optimal for GPU_CAGRA indexing)
- **Build Parallel**: 1 (CUVS team recommendation for single GPU)
- **Concurrent Import Tasks**: 16 (balanced for S3 I/O)
- **Storage V2**: Disabled (required for Milvus 2.4.4 bulk import compatibility)

### 10 Million Embeddings Benchmark

**Test Configuration:**
- **Dataset Size**: 10,000,000 vectors
- **Embedding Dimension**: 256-dimensional float32
- **File Size**: 9.9 GB (parquet format)
- **Batch Size**: 250,000 vectors per batch
- **Segment Size**: 2GB (optimized)

**Results:**

| Metric | Value |
|--------|-------|
| **Total Time** | 419.38 seconds |
| **Average Throughput** | **23,845 vectors/second** |
| **Throughput (per minute)** | 1,430,700 vectors/minute |
| **Throughput (per hour)** | 85,842,000 vectors/hour |

### Performance Breakdown

The bulk ingestion process consists of three main phases:

1. **Data Upload to S3**: ~20 seconds (S3 transfer)
2. **Data Loading**: ~280 seconds (61% of total time)
3. **GPU Index Building**: ~140 seconds (32% of total time)

### GPU Indexing Performance (Isolated)

When measured independently (without S3 I/O overhead), the GPU_CAGRA indexing performance for our setup is significantly higher:

| Metric | Value |
|--------|-------|
| **Pure GPU Indexing Throughput** | **~36,966 vectors/second** |

**Key Insight**: The GPU can index vectors 2x faster than the end-to-end throughput, confirming that **S3 I/O is the primary bottleneck**, not GPU compute capacity. The GPU spends significant time idle, waiting for data to arrive from storage.

### GPU Memory Allocation

The system uses a conservative GPU memory allocation strategy:

- **CE1 OSS service**: ~12 GB (embedding model)
- **Milvus GPU Pool**: 13.4 GB active (configured: 16GB init / 30GB max)
- **Available Headroom**: ~28 GB unused

## Performance Analysis

### Current Scenario

1. **S3 I/O Throughput** (Primary Bottleneck)
   - LocalStack adds significant overhead compared to production S3
   - Data loading phase dominates total time (61%)
   - Network transfer and deserialization overhead


### Optimization Opportunities

#### High-Impact Optimizations

1. **Production S3 Backend**
   - **Expected Improvement**: 2-3x throughput
   - Replace LocalStack with AWS S3 or MinIO cluster
   - Reduces I/O bottleneck significantly

2. **Increase Segment Size**
   - **Current**: 2GB segments (~2M vectors)
   - **Recommended**: 4GB segments (~4M vectors)
   - **Benefit**: Better GPU batching, fewer index builds

3. **Pipeline Parallelism**
   - **Build parallel**: `buildParallel=2`
   - **Benefit**: Overlap data loading with index building
   - **Risk**: Potential GPU memory contention

4. **Increase Concurrent Tasks**
   - **Current**: 16 concurrent import tasks
   - **Recommended**: 32-64 concurrent tasks
   - **Benefit**: Better S3 parallelization, keeps GPU fed

#### Medium-Impact Optimizations

5. **Larger Read Buffers**
   - Increase from 16MB to 64-128MB
   - Reduces number of S3 requests

6. **Pre-stage Data**
   - Upload data to S3 before benchmarking
   - Measure pure indexing performance

7. **Multiple Parquet Files**
   - Split 10M vectors into multiple files
   - Enable parallel file processing

### Expected Performance with Optimizations

| Optimization Level | Expected Throughput | Estimated Time (10M) |
|-------------------|---------------------|---------------------|
| **Current** | 23,845 vec/s | 6.98 minutes

---

## 2. Video Ingestion Performance

### Overview

Video ingestion benchmarks measure the end-to-end performance of processing video files, generating embeddings using CE1 OSS service, and storing them in Milvus. This represents the real-world use case of ingesting video datasets.

### How to Run Video Ingestion Benchmarks

#### Step 1: Download Video Dataset

Download the MSR-VTT test dataset (1000 videos):

```bash
# Download using prepared configuration
make prepare-dataset CONFIG=scripts/msrvtt_test_1000.yaml

# Videos will be downloaded to: ~/datasets/msrvtt/videos/
```

#### Step 2: Host Videos via HTTP

Start a local HTTP server to serve the video files:

```bash
# Host videos on local network interface
python3 scripts/evals/video_ingestion_performance/http_file_host.py \
  --url-host-mode interface \
  --interface enp2s0f0np0 \
  --root ~/datasets/msrvtt/videos \
  --port 8234 \
  --csv-path hosted_files.csv
```

This will:
- Start an HTTP server on port 8234
- Generate a CSV file (`hosted_files.csv`) with video URLs
- Output the base URL (e.g., `http://192.0.2.10:8234`)

Allow that exact trusted HTTP origin in `deploy/standalone/.env`, using the base
URL printed by the server without a trailing path:

```bash
COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HTTP_ORIGINS=http://192.0.2.10:8234
```

Then run `make test-integration-up` to recreate the CE1 service with the new
value. Multiple trusted HTTP origins must be comma-separated. HTTPS media does
not require this setting.

**Parameters:**
- `--url-host-mode`: How to determine the host URL (`interface`, `localhost`, or `ip`)
- `--interface`: Network interface name (e.g., `enp2s0f0np0`)
- `--root`: Root directory containing videos
- `--port`: HTTP server port
- `--csv-path`: Output CSV file with video URLs

#### Step 3: Create Collection

Create a new collection for video ingestion:

```bash
cds collections create \
  --pipeline cosmos_video_search_milvus \
  --name "MSR-VTT Performance Test"
```

Note the returned `collection_id` for the next step.

#### Step 4: Run Ingestion Benchmark

Ingest videos using the `url_ingestion.py` script:

```bash
# Run video ingestion benchmark
python3 scripts/evals/video_ingestion_performance/url_ingestion.py \
  --collection-id <collection_id> \
  --csv hosted_files.csv \
  --verbose \
  --max-videos 1000 \
  --batch-size 64 \
  --base-url http://localhost:8888 \
  --cosmos-embed-base-url http://localhost:9000 \
  --measure-embed-delta \
  --csv-out report.csv
```

**Parameters:**
- `--collection-id`: Target collection ID
- `--csv`: CSV file with video URLs (from Step 2)
- `--max-videos`: Maximum number of videos to process
- `--batch-size`: Number of videos per batch sent to embedding service
- `--base-url`: Visual search API endpoint
- `--cosmos-embed-base-url`: CE1 OSS service endpoint
- `--measure-embed-delta`: Measure CE1 OSS service embedding latency separately
- `--csv-out`: Output CSV file for detailed per-batch metrics

### MSR-VTT 1000 Videos Benchmark

> **Legacy baseline:** The static figures in this section predate the
> source-built CE1 OSS backend. Re-run the command above before using results
> as CE1 OSS performance evidence.

**Test Configuration:**
- **Dataset**: MSR-VTT test set (1000 videos)
- **Video Format**: MP4
- **Embedding Model**: Legacy pre-OSS service (running on same L40 GPU)
- **Batch Size**: 64 videos per batch
- **Total Batches**: 16 batches (15 full batches of 64 + 1 partial batch of 40)
- **Video Hosting**: Local HTTP server (port 8234)

**Results:**

| Metric | Value |
|--------|-------|
| **Total Videos** | 1000 |
| **Total Time** | 53.09 seconds |
| **Average Throughput** | **18.84 videos/second** |
| **Throughput (per minute)** | 1,130 videos/minute |
| **Throughput (per hour)** | 67,800 videos/hour |
| **Per-Video Processing Time** | 0.053 seconds |
| **Success Rate** | 100% (1000/1000) |

### Per-Batch Performance Breakdown

**CDS API (End-to-End):**

| Metric | Value |
|--------|-------|
| **Min Latency** | 2.09 seconds |
| **Avg Latency** | 3.14 seconds |
| **Max Latency** | 4.25 seconds |

**Legacy Embedding Service (Embedding Generation):**

| Metric | Value |
|--------|-------|
| **Min Latency** | 1.71 seconds |
| **Avg Latency** | 2.75 seconds |
| **Max Latency** | 2.96 seconds |

**CDS Overhead (CDS API - Legacy Embedding Service):**

| Metric | Value |
|--------|-------|
| **Avg Overhead** | 0.39 seconds |
| **Overhead Percentage** | 12.4% of total time |

### Performance Analysis

**Key Insights:**

1. **Embedding Generation Dominates**: The legacy embedding service takes ~2.75s per batch (64 videos), which is **87.6%** of total processing time. Re-measure this for CE1 OSS.

2. **CDS API Overhead is Minimal**: The CDS API overhead (video download from HTTP, Milvus insertion, orchestration) is only ~0.39s per batch (**12.4%** of total time), showing efficient pipeline implementation.

3. **Batch Processing is Efficient**: Processing 64 videos in ~3.14s achieves high GPU utilization on the embedding model, with minimal per-video overhead.

4. **Consistent Performance**: Low latency variance (2.09s - 4.25s) indicates stable processing across all batches.

5. **High Throughput**: At **18.84 videos/second**, the system can process approximately **68K videos per hour**, making it suitable for large-scale video dataset ingestion.

### Detailed Batch Statistics

Sample of per-batch performance :

| Batch | Size | CDS Latency (s) | Legacy Embed Latency (s) | Overhead (s) |
|-------|------|-----------------|-----------------|--------------|
| 1 | 64 | 4.25 | 2.86 | 1.39 |
| 2 | 64 | 2.92 | 2.77 | 0.16 |
| 3 | 64 | 3.26 | 2.79 | 0.46 |
| 8 | 64 | 3.19 | 2.80 | 0.38 |
| 15 | 64 | 2.98 | 2.83 | 0.15 |
| 16 | 40 | 2.09 | 1.71 | 0.38 |

**Observations:**
- First batch has higher latency (4.25s) due to cold start / initialization
- Subsequent batches stabilize around 3.0-3.3s
- Last batch (40 videos) is proportionally faster (2.09s)
- The legacy embedding-service latency was consistent (2.75s ± 0.1s)

### Video Ingestion Optimization Opportunities

1. **Optimize Batch Size**
   - Current: 64 videos/batch
   - Trade-off: Larger batches = better GPU utilization but longer latency per batch
   - Smaller batches = more frequent progress updates but potential GPU underutilization
   - Recommendation: Test batch sizes 16, 32, 64 to find optimal balance

2. **Dedicated GPU for Embedding**
   - Current: The embedding service shares an L40 GPU with Milvus
   - Alternative: Dedicated GPU for embeddings
   - Benefit: Eliminate GPU contention, potentially 20-30% throughput improvement

3. **Parallel Processing**
   - Current: Single-threaded batch processing
   - Alternative: Multiple parallel workers with smaller batches, `cds` cli implements Ray workers for ingestion
   - Benefit: Overlap downloads and request preparation. CDS 1.2.0 CE1 video decoding is GPU-only; these historical measurements do not benchmark the new runtime.

---

## 3. Text Search Latency Performance

### Overview

Search latency benchmarks measure the query response time for semantic text-to-video search over large collections. This represents the end-user experience when searching the video dataset.

### How to Run Search Latency Benchmarks

Use the `latency_test.py` script to measure search performance:

```bash
# Run 60-second latency test with 200 diverse queries
python3 scripts/evals/video_ingestion_performance/latency_test.py \
  --base-url http://localhost:8888 \
  --latency-test \
  --verbose \
  --collection-id <collection_id> \
  --csv-out latency_report.csv \
  --cosmos-embed-base-url http://localhost:9000 \
  --query-pool-size 200 \
  --duration 60 \
  --top-k 20
```

**Parameters:**
- `--base-url`: Visual search API endpoint
- `--collection-id`: Target collection ID
- `--cosmos-embed-base-url`: CE1 OSS service endpoint
- `--query-pool-size`: Number of diverse queries to generate (default: 200)
- `--duration`: Test duration in seconds (default: 60)
- `--top-k`: Number of results to retrieve per query (default: 20)
- `--csv-out`: Output CSV file for detailed results

**The benchmark will:**
1. Generate a diverse pool of text queries
2. Measure baseline API latency (health checks)
3. Run continuous searches for the specified duration
4. Measure CE1 OSS service embedding latency separately
5. Report detailed latency statistics and breakdowns

### 10 Million Embeddings Search Benchmark

> **Legacy baseline:** The static figures below predate the source-built CE1
> OSS backend. Re-run the command above to produce current CE1 OSS results.

**Test Configuration:**
- **Collection Size**: 10,000,000 embeddings (256-dim)
- **Index Type**: GPU_CAGRA
- **Query Pool**: 200 diverse text queries
- **Test Duration**: 60 seconds
- **Top-K**: 20 results per query
- **Pattern**: Continuous (no delays between queries)

**Results:**

| Metric | Value |
|--------|-------|
| **Total Searches** | 190 queries |
| **Test Duration** | 60.2 seconds |
| **Throughput** | **3.16 searches/second** |
| **Min Latency** | 0.147 seconds |
| **Avg Latency** | 0.317 seconds |
| **Max Latency** | 0.464 seconds |

### Latency Breakdown

| Component | Avg Latency | Percentage |
|-----------|-------------|------------|
| **Total End-to-End** | 0.317 seconds | 100% |
| **Milvus Search** | 0.305 seconds | 96.2% |
| **Legacy Embedding Service** | 0.007 seconds | 2.2% |
| **Network/API Overhead** | 0.004 seconds | 1.3% |
| **Visual Search API Baseline** | 0.004 seconds | 1.3% |

**Key Insights:**

1. **Milvus GPU Search Dominates**: The GPU_CAGRA index search takes 96.2% of total latency (0.305s), which is expected for a 10M vector collection.

2. **Embedding Generation Was Fast in the Legacy Run**: Re-measure text embedding latency with CE1 OSS before drawing a current performance conclusion.

3. **Low API Overhead**: The visual search API adds minimal overhead (0.004s or 1.3%).

4. **Consistent Performance**: Latency variance is low (0.147s - 0.464s), with most queries completing in 0.2-0.4s range.

5. **Sub-Second Response**: Average 317ms end-to-end latency provides good user experience for semantic search.

### Query Performance Distribution

Sample query latencies from the benchmark:

| Query | Count | Min (s) | Avg (s) | Max (s) |
|-------|-------|---------|---------|---------|
| waves lapping against a pier | 1 | 0.147 | 0.147 | 0.147 |
| cargo ship entering harbor | 2 | 0.191 | 0.193 | 0.195 |
| penguins waddling on ice | 2 | 0.195 | 0.198 | 0.201 |
| street food vendor cooking | 5 | 0.197 | 0.321 | 0.404 |
| chocolate melting in bowl | 3 | 0.382 | 0.397 | 0.409 |
| loading clothes into washer | 1 | 0.464 | 0.464 | 0.464 |

**Observations:**
- Simple queries (e.g., "waves lapping") complete faster (~0.15s)
- Complex queries (e.g., "loading clothes") take longer (~0.46s)
- Repeated queries show consistent latency
- Most queries complete in 0.2-0.4s range

### Search Latency Optimization Opportunities

1. **Increase GPU Memory Allocation**
   - Current: 13.4GB active (29% of 46GB available)
   - Recommended: Increase to 20-25GB
   - Benefit: More index data cached on GPU, faster searches

2. **Tune GPU_CAGRA Parameters**
   - Current: Default settings
   - Optimize: `intermediate_graph_degree`, `graph_degree`, `itopk_size`
   - Trade-off: Search speed vs. recall accuracy

3. **Query Batching**
   - Current: Single query per request
   - Alternative: Batch multiple queries
   - Benefit: Better GPU utilization, higher throughput

4. **Index Optimization**
   - Current: GPU_CAGRA with default build parameters
   - Tune: Build-time parameters for search performance
   - Benefit: Faster searches at cost of longer index build time

---

## 4. Configuration Parameters

### Key Milvus Settings

```yaml
# GPU Memory Pool (L40 48GB)
gpu:
  initMemSize: 16384  # 16GB initial
  maxMemSize: 30720   # 30GB maximum

# Segment Configuration
dataCoord:
  segment:
    maxSize: 2048  # 2GB segments

# Index Building
indexNode:
  scheduler:
    buildParallel: 1  # Conservative for stability

# Data Import
dataNode:
  import:
    maxConcurrentTaskNum: 16
    maxImportFileSizeInGB: 16
    readBufferSizeInMB: 16
```

### Recommended Production Settings

For production deployments with real S3:

```yaml
# Aggressive segment sizing
dataCoord:
  segment:
    maxSize: 4096  # 4GB segments

# Increased parallelism
dataNode:
  import:
    maxConcurrentTaskNum: 32
    readBufferSizeInMB: 64

# Experimental: Pipeline overlap
indexNode:
  scheduler:
    buildParallel: 2  # Test carefully
```

## Scaling Considerations

### GPU Memory Capacity

The L40's 48GB VRAM can accommodate approximately:
- **With 30GB allocated to Milvus**: ~25-30M vectors (256-dim) with GPU_CAGRA index
- **Theoretical maximum**: ~40M vectors before requiring batch processing

### Distributed vs. Standalone

**Standalone Mode (Current)**
- Simpler configuration
- Lower overhead
- Suitable for <30M vectors
- Single point of failure
- Limited by single GPU

**Distributed Mode (Multi-node DataNode & IndexNode)**
- Horizontal scaling by deploying multiple DataNode and IndexNode processes
- Enables parallel data ingestion and simultaneous GPU index building across nodes
- Higher throughput with multiple GPUs, as DataNode and IndexNode workloads are distributed
- Built-in fault tolerance (node failures don't halt the cluster)
- More complex cluster management and monitoring
- Higher operational and infrastructure overhead, but supports >50M or 100M+ vectors and production-scale workloads

## 5. Summary and Recommendations

### Performance Summary

| Benchmark Type | Key Metric | Value | Notes |
|---------------|------------|-------|-------|
| **Bulk Ingestion** | Throughput | 23,845 vectors/s | 10M embeddings in 6.98 min |
| **Video Ingestion** | Throughput | 18.84 videos/s | 1000 videos in 53 seconds |
| **Search Latency** | Avg Response | 0.317 seconds | 10M collection, top-20 |

### For Development/Testing
- **Current configuration is optimal**
- Focus on application development
- LocalStack S3 is sufficient

### For Production Deployment

1. **Immediate Actions**
   - Switch to production S3 (AWS/MinIO cluster)
   - Increase segment size to 4GB
   - Increase concurrent tasks to 32

2. **Performance Tuning**
   - Monitor GPU utilization during bulk imports
   - Adjust `buildParallel` based on memory usage
   - Tune read buffer sizes based on S3 latency

3. **Scaling Strategy**
   - Standalone mode for <30M vectors
   - Consider distributed mode for >50M vectors
   - Plan for multiple L40 GPUs for >100M vectors


---

*Last Updated: October 24, 2025*  
*Benchmark Version: 1.0*  
*Configuration: milvus_l40_standalone_optimized.yaml*
