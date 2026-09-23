# Ingestion source settings

CDS validates caller-supplied text URLs and S3 Parquet paths before importing. This protects the service's outbound network access; customer-managed authentication and authorization are unchanged.

## Default behavior

- Text URLs may use public HTTPS. Each DNS result and each redirect is checked. Connections use the checked numeric addresses while retaining TLS hostname verification and the original presigned query string.
- Loopback, link-local, cloud metadata, multicast and reserved destinations are rejected. Private storage requires an operator-configured exact origin.
- Parquet import accepts plain `s3://bucket/key` paths only, not local files, HTTP URLs, glob patterns or chained fsspec protocols. Every path is checked before any file is read or any bulk job is created.
- A caller cannot select an arbitrary S3 endpoint. Custom endpoints must be configured by the operator.
- Text downloads are limited to 16 MiB by default. Rejected or failed text downloads do not replace existing documents.

## Configuration

These environment variables belong to the CDS visual-search service, not CE1:

| Variable | Purpose |
| --- | --- |
| `CDS_FETCH_ALLOWED_ORIGINS` | Optional comma-separated exact public origins, such as `https://datasets.example.com`. Empty allows public HTTPS. |
| `CDS_FETCH_PRIVATE_ORIGINS` | Exact trusted storage origins permitted to use private RFC1918/ULA addresses or HTTP, for example `http://minio.storage.svc.cluster.local:9000`. It does not permit loopback or metadata addresses. |
| `CDS_FETCH_S3_ENDPOINTS` | Exact custom S3 endpoint origins callers may select. Operator `AWS_ENDPOINT_URL_S3`/`AWS_ENDPOINT_URL` also authorize that endpoint. For a private endpoint, configure `CDS_FETCH_PRIVATE_ORIGINS` as well. |
| `CDS_FETCH_MAX_TEXT_BYTES` | Positive text download limit in bytes; default `16777216`. |

Use the `env` keys with the same names in Helm values. Standalone Compose forwards these variables and explicitly permits its own LocalStack service. Never use a caller-supplied value to populate these server settings.

For private MinIO, configure both `CDS_FETCH_PRIVATE_ORIGINS=http://minio:9000` and `CDS_FETCH_S3_ENDPOINTS=http://minio:9000`. The request's `endpoint_url` must match that origin. Credential handling and S3 bucket permissions remain unchanged.

Text imports use direct outbound connections, not ambient HTTP proxy variables. Deployments requiring a proxy need an operator-managed egress design; do not bypass URL validation. Restrict outbound network access at the cluster/firewall layer as defense in depth.

## Video and image sources (CE1)

Public HTTPS and presigned media URLs work with the default CE1 configuration. For private storage, configure CE1 separately from the CDS text/Parquet settings above:

| CE1 variable | Purpose |
| --- | --- |
| `COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HOSTS` | Optional comma-separated host allowlist. |
| `COSMOS_EMBED_PRESIGNED_URL_ALLOWED_HTTP_ORIGINS` | Exact trusted HTTP storage origins, for example `http://minio:9000`. |
| `COSMOS_EMBED_PRESIGNED_URL_ENDPOINT_URL` | Operator-controlled endpoint override for compatible private object storage. |

Set these on the `cosmos-embed` service. A CDS setting does not configure CE1, or vice versa. Allow only storage you control, keep presigned URLs out of logs, and see [MinIO setup](../guides/minio-support.md) for an example. Video input uses GPU decoding; unsupported formats must be re-encoded before ingestion.

## Validation

`make test-visual-search-local` includes the fetch-policy and ingestion regressions. Release validation should additionally exercise real presigned text/video URLs, S3 Parquet ingestion and search in the deployment, and confirm rejected URLs cannot reach a private destination or remove an existing document.
