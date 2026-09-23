# API reference

## Live OpenAPI schema

Use the schema served by your running CDS instance for request and response details. The public source export does not include a static schema snapshot.

| Endpoint | Standalone Compose | EKS ingress |
| --- | --- | --- |
| Swagger UI | `http://localhost:8888/v1/docs` | `https://<ingress-host>/api/v1/docs` |
| ReDoc | `http://localhost:8888/v1/redoc` | `https://<ingress-host>/api/v1/redoc` |
| Raw schema | `http://localhost:8888/v1/openapi.json` | `https://<ingress-host>/api/v1/openapi.json` |
| CDS health | `http://localhost:8888/health` | `https://<ingress-host>/api/health` |

The application schema still has a legacy version label. Identify the deployed release using its [image tag and digest](release-containers.md), not that label alone. CDS health checks process availability; verify CE1 readiness and ingestion/search separately.

## CDS endpoints

Paths below are relative to the service root; add `/api` when using the EKS ingress.

| Method and path | Purpose |
| --- | --- |
| `GET /health` | CDS health |
| `GET /v1/pipelines` | Available pipelines |
| `GET /v1/pipelines/draw/{name}` | Pipeline diagram |
| `POST /v1/collections` | Create a collection |
| `GET /v1/collections` | List collections |
| `GET /v1/collections/{collection_id}` | Collection details |
| `PATCH /v1/collections/{collection_id}` | Update collection metadata |
| `DELETE /v1/collections/{collection_id}` | Delete a collection |
| `GET /v1/pipelines/{pipeline_id}/collections` | Collections for a pipeline |
| `POST /v1/collections/{collection_id}/documents` | Index documents and generate embeddings |
| `POST /v1/insert-data` | Start asynchronous S3 Parquet bulk import |
| `GET /v1/job-status/{job_id}` | Bulk-import job status |
| `GET /v1/jobs` | Bulk-import jobs |
| `POST /v1/collections/{collection_id}/search` | Search a collection |
| `POST /v1/retrieval` | Cross-collection retrieval |
| `POST /v1/search_refinement/train` | Train a search-refinement model |
| `GET /v1/metrics` | Prometheus metrics |

There is no default `/v1/secrets` CRUD API or `/search/hybrid` endpoint in this release. Provision storage credentials through your deployment's secret-management process and configure [storage Secret access](aws-eks-deployment.md#storage-secret-access).

## Direct CE1 OSS API

Standalone Compose publishes CE1 on `http://localhost:9000` (container port 8000). These endpoints belong to CE1, not CDS's `/v1` router.

| Method and path | Purpose |
| --- | --- |
| `POST /v1/embeddings` | Text, video and batch embeddings |
| `GET /v1/health/live` | Process liveness |
| `GET /v1/health/ready` | Backend/model readiness; 503 when unavailable |
| `GET /v1/models` | Model list |
| `GET /v1/metadata` | Model/runtime metadata |
| `GET /v1/license` | Service license metadata |
| `GET /v1/manifest` | Service manifest |
| `GET /v1/metrics` | Prometheus metrics |
| `GET /v1/health/metrics` | JSON request statistics; `/health/metrics` is an alias |

Bulk text and bulk video use `request_type: "bulk_text"` and `request_type: "bulk_video"` on `/v1/embeddings`, not separate bulk routes. See the [API user guide](api-user-guide.md) and [curl examples](../guides/api.md).

## Access and limits

CDS and CE1 do not implement customer authentication or authorization. Protect both APIs at your deployment boundary and configure TLS, allowed browser origins and rate limits. For supported source URLs and private storage, see [ingestion source settings](import-url-security.md).
