# NVIDIA Cosmos Dataset Search (CDS)

NVIDIA Cosmos Dataset Search (CDS) is a comprehensive platform for semantic search across video datasets using advanced AI models. The platform enables text-to-video and video-to-video queries against large-scale video collections with GPU-accelerated inference and vector similarity search.

## Key Features

### Search and Retrieval
- Multimodal semantic search with text and video queries
- GPU-accelerated embedding generation using NVIDIA CE1 OSS service
- Fast vector similarity search powered by Milvus with NVIDIA cuVS acceleration
- Support for multiple collections and cross-modal retrieval

### Data Management
- Automated video ingestion pipeline with frame extraction and metadata indexing
- S3-compatible object storage (LocalStack, MinIO, AWS S3)
- Batch and incremental data ingestion with Ray-based parallel processing
- Support for various video formats and frame rates

### User Experience
- REST API for programmatic access with comprehensive documentation
- CDS CLI for collection management and data ingestion
- Search results with video metadata for downstream applications

### Deployment and Operations
- Flexible deployment: Docker Compose for local development, Helm for production
- Production-ready Helm charts for AWS EKS and generic Kubernetes
- Comprehensive monitoring, logging, and observability
- Modular architecture enabling independent scaling of components

## Software Components

### CE1 OSS Embedding Service
- **CE1 OSS service** - Unified embedding service providing state-of-the-art text and video embeddings in a common semantic space optimized for temporal understanding and cross-modal search.

CE1 requires an NVIDIA CUDA GPU with NVDEC video decoding support. For
source-based CE1 development, install `uv sync --extra gpu-media`. Video inputs
use GPU decoding with eight uniformly sampled frames and the existing resize
and embedding APIs. Unsupported inputs are rejected with a re-encoding hint;
there is no CPU video decoder or automatic CPU fallback. Setting
`COSMOS_EMBED_GPU_VIDEO=false` is no longer supported. Already-extracted image
frames can still use Pillow preprocessing with `COSMOS_EMBED_GPU_FRAMES=false`.
The system `ffprobe` media checks remain in use before GPU video decoding.

Neither service container includes PyAV. It remains an optional dependency of
the offline benchmark tools; `make install-python` includes all extras.

### Integration and Orchestration Layer
- **Visual Search Service** - FastAPI-based REST API orchestrating ingestion and search operations
- **Milvus Vector Database** - GPU-accelerated vector storage and similarity search with NVIDIA cuVS
- **Object Storage** - S3-compatible storage for video files, frames, and metadata (LocalStack, MinIO, or AWS S3)

The Visual Search service coordinates CE1 embedding generation, Milvus vector
storage, and S3-compatible object storage for ingestion and retrieval requests.

## Technical Diagram

<img src="docs/docs/images/cds-architecture.jpg" alt="CDS architecture: CLI/API, CDS API, CE1 OSS, Milvus and object storage" width="700" />

## Workflow

See the [runtime architecture](docs/docs/architecture.md) for the component diagram and [ingestion source settings](docs/docs/import-url-security.md) when using private storage or custom S3 endpoints.

1. **Install Prerequisites** - Set up required software and dependencies for your chosen deployment method.
2. **Deploy Services** - Launch CDS using Docker Compose for local deployment or Helm charts for Kubernetes deployment.
3. **Create and Manage Collections** - Use the CDS CLI or REST API to create vector collections for organizing your video datasets.
4. **Ingest Video Data** - Upload videos using the ingestion pipeline or bulk indexing, which processes videos and extracts embeddings.
5. **Search via CLI or API** - Query collections programmatically using the CDS CLI or REST API for integration into workflows and applications.

## Deployment Options
Prebuilt CDS and CE1 images are available through the [1.2.0 container guide](docs/docs/release-containers.md).

CDS supports two primary deployment methods: Docker Compose for local or small-scale setups, and Kubernetes with Helm charts for scalable, production-grade environments. Easy setup and deployment guides for both methods are provided below.

### Docker Compose Deployment

Best for local development, testing, and small-scale deployments. All services run on a single node with simplified configuration.

- **[Docker Compose Deployment Prerequisites](docs/docs/docker-compose-prerequisites.md)** - System requirements and setup
- **[Docker Compose Deployment Guide](docs/docs/docker-compose-deployment.md)** - Step-by-step deployment instructions
- **[Docker Compose Troubleshooting](docs/docs/troubleshooting-docker-compose.md)** - Common issues and solutions

### Kubernetes Deployment with Helm

Recommended for production deployments requiring scalability, high availability, and enterprise features.


#### Deployment Guides

- **[AWS EKS Deployment](docs/docs/aws-eks-deployment.md)** - Deploy on Amazon Elastic Kubernetes Service

## Using CDS

After deploying CDS, use the REST API or CDS CLI to manage collections, ingest
videos, and run searches.

### User Guide

**[CDS User Guide](docs/docs/user-guide.md)** - Tutorials for the REST API and CLI

## Documentation

For comprehensive documentation, see the **[Complete Documentation Index](docs/docs/documentation.md)**.

## Repository Structure

```
cds/
├── src/                          # Source code and source-local tests
│   ├── visual_search/           # Visual search service
│   ├── cosmos_embed_oss/        # CE1 PyTorch inference service
│   ├── haystack/                # Haystack integration components
│   ├── models/                  # Model implementations
│   └── triton/                  # Triton inference server configs
├── deploy/                       # Deployment configurations
│   └── standalone/              # Standalone deployment options
├── docker/                       # Service and base-image Dockerfiles
├── infra/                        # Infrastructure as code
│   └── blueprint/               # Helm charts and Kubernetes manifests
├── docs/                         # Documentation
│   ├── docs/                    # User documentation
│   └── guides/                  # Deployment and usage guides
└── scripts/                      # Dataset and evaluation utilities
    └── evals/                   # Evaluation scripts
```

## License

The NVIDIA-owned source code in this repository is licensed under the [Apache License 2.0](LICENSE). Third-party software and separately downloaded model weights retain their respective licenses. NGC container distributions are subject to the terms listed on their NGC catalog pages.

## Security

For security concerns, please refer to our [Security Policy](SECURITY.md).

## Security Considerations
The Cosmos Dataset Search Blueprint is shared as a reference and is provided "as is". The security in the production environment is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats; define the trust boundaries, implement logging and monitoring capabilities, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment up to date, ensure the containers/source code are secure and free of known vulnerabilities.

Configure your gateway's authentication, TLS and allowed browser origins before exposing CDS or CE1. Storage source settings are documented in the [ingestion guide](docs/docs/import-url-security.md); Kubernetes credential access is covered in the [EKS guide](docs/docs/aws-eks-deployment.md#storage-secret-access).
