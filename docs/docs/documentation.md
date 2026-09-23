# CDS Documentation

Welcome to the NVIDIA Cosmos Dataset Search (CDS) documentation. This page provides a comprehensive index of all available documentation organized by topic.

## Overview

- **[Introduction](introduction.md)** - Overview of CDS architecture and capabilities
- **[Runtime Architecture](architecture.md)** - CDS, CE1, storage and deployment boundaries
- **[1.2.0 Container Images](release-containers.md)** - Published NGC tags and digests

## Quick Start Deployment

Get CDS up and running quickly with these deployment guides:

### Docker Compose Deployment

Recommended for local development, testing, and evaluation:

- **[Docker Compose Prerequisites](docker-compose-prerequisites.md)** - System requirements and setup
- **[Docker Compose Deployment Guide](docker-compose-deployment.md)** - Step-by-step deployment
- **[Docker Compose Troubleshooting](troubleshooting-docker-compose.md)** - Common issues and solutions

### Kubernetes Deployment

Deployment templates to configure and harden for your environment:

- **[AWS EKS Quickstart Deployment Guide](aws-eks-deployment.md)** - Deploy on Amazon EKS

## User Guides

Learn how to interact with CDS after deployment:

### Quick Start Guides

- **[CDS User Guide](user-guide.md)** - Choose the CLI or REST API
  - **[CLI User Guide](cli-user-guide.md)** - Command-line interface tutorial
  - **[REST API User Guide](api-user-guide.md)** - REST API hands-on tutorial

## Configuration and Tuning

Customize and optimize your CDS deployment:

- **[GPU Memory Management](../guides/gpu-memory-management.md)** - GPU configuration and optimization
- **[MinIO Support](../guides/minio-support.md)** - Configure MinIO as object storage backend

## Reference Documentation

### API References

- **[REST API Reference](../guides/api.md)** - Complete API endpoint documentation
- **[OpenAPI Specification](api_reference.md)** - API schema and specification
- **[Live OpenAPI Schema](api_reference.md#live-openapi-schema)** - Schema served by the deployed release

### Performance and Evaluation

- **[Performance Guide](performance.md)** - Performance benchmarks and optimization

### Security

- **[Security Policy](../../SECURITY.md)** - Security guidelines and reporting
- **[Ingestion Source Settings](import-url-security.md)** - Supported URLs, private storage and custom S3 endpoints
- **[Import URL Security](import-url-security.md)** - Text and S3 Parquet destination configuration

## Troubleshooting

- **[Docker Compose Troubleshooting](troubleshooting-docker-compose.md)** - Local deployment issues
- **[Kubernetes Troubleshooting](troubleshooting-kubernetes.md)** - Kubernetes deployment issues
- **[General Troubleshooting](../guides/troubleshooting.md)** - Additional troubleshooting resources

## Support and Community

- **[Contributing Guidelines](../../CONTRIBUTING.md)** - How to contribute (if applicable)
- **[GitHub Repository](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search)** - Source code and issues
