# Introduction

Cosmos Video Dataset Search (CVDS) is a suite of visual search and analytics micro-services to ingest, index,
search and curate multi-modal data with a focus on video understanding and temporal reasoning.

The system has the following main components:

* A Core Service orchestrating ingestion and search queries.
* **CE1 OSS Service**: A unified embedding service powered by the state-of-the-art NVIDIA Cosmos model,
  providing superior text and video embeddings in a common semantic space.
* A vector database (Milvus) to store embeddings, perform embedding-space search and persist collection metadata in `_collection_registry`.
* Customer-managed object storage for media. Kubernetes deployments can fetch storage credentials from explicitly allowed Secret names; no separate PostgreSQL metadata service is required by this release.

* The indexing service ingests videos and associated metadata into named collections, either from scratch or incrementally.
* A retrieval service retrieves assets from the queried volume by loading and querying the associate
  index. The retrieval service is queried programmatically through the CDS CLI or REST API.
* The CE1 OSS service embeds high dimensional video and text data into a unified low-dimensional semantic space
  optimized for temporal understanding and cross-modal search.

See the [runtime architecture diagram](architecture.md), [release containers](release-containers.md), and [ingestion source settings](import-url-security.md). CDS is a blueprint; customers provide authentication, authorization, TLS and deployment-level access controls.
