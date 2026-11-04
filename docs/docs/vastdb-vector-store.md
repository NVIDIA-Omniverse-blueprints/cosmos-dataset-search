# Cosmos Dataset Search on VastDB

This guide shows how to run **NVIDIA Cosmos Dataset Search (CDS)** with
**VAST Database (VastDB)** as the vector store and **Vast S3** for media —
as the sole vector and object-storage path.

CDS keeps the same REST API, CLI, CE1 embeddings, and collection workflows.
VastDB, etcd, and Vast S3 are not part of this branch.

> **Status:** reference integration. Structurally complete and intended for
> customer trials on a VAST cluster. Validate end-to-end against your VIP pool
> and ADBC driver build before production use.

## Why VastDB

| Concern | Upstream CDS (VastDB) | This branch (VastDB) |
| --- | --- | --- |
| Vector scale | Separate GPU/CPU ANN service | Native vectors in the VAST DataBase |
| Object storage | Vast S3 / MinIO / AWS S3 | **Vast S3** on the same platform |
| Metadata registry | `_collection_registry` in VastDB | `_collection_registry` table in VastDB |
| Similarity search | IVF / GPU_CAGRA indexes | Query Engine distance functions |
| Ops surface | etcd + VastDB + S3 emulator | VastDB VIP + ADBC driver |

Public VAST references:

- [Introducing VAST Vector Search](https://www.vastdata.com/blog/introducing-vast-vector-search-real-time-ai-retrieval-without-limits)
- [Vector Search (Administrator Guide)](https://kb.vastdata.com/documentation/docs/vector-search-55)
- [VAST Query Engine](https://kb.vastdata.com/documentation/docs/vast-query-engine)
- [Vector DB Quickstart (field docs)](https://vast-data.github.io/data-platform-field-docs/vast_vectordb/quickstart-new.html)
- [vastdb Python SDK](https://github.com/vast-data/vastdb_sdk)
- [vastdb ADBC driver](https://github.com/vast-data/vastdb-adbc-driver)
- [VAST Database product](https://www.vastdata.com/database)

For architecture diagrams see [VastDB architecture](vastdb-architecture.md).

## Prerequisites

1. A VAST cluster with:
   - VastDB / Query Engine VIP pool
   - Vast S3 endpoint
   - S3 user access + secret keys with DB and bucket permissions
2. The [ADBC driver](https://github.com/vast-data/vastdb-adbc-driver) installed
   on the CDS host (or mounted into the `visual-search` container).
3. CDS 1.2.0 sources from this branch, with optional deps:

```bash
pip install -e ".[vastdb]"
# or inside your image build:
#   vastdb, pyarrow, adbc-driver-manager
```

4. Cosmos Embed (CE1) still required — unchanged from stock CDS.

## Quick start (Docker Compose)

```bash
cd deploy/standalone
cp .env.vastdb.example .env
# Edit VASTDB_ENDPOINT, VAST_* keys, VAST_S3_ENDPOINT, DATA_DIR, NVIDIA_API_KEY

docker compose \
  -f docker-compose.build.yml \
  -f docker-compose.ce1-oss.yml \
  -f docker-compose.vastdb.yml \
  up -d cosmos-embed visual-search
```

The VastDB overlay **disables** `vastdb`, `etcd`, and `vast-s3`.

Pipeline selected by env:

```bash
ALLOWED_PIPELINES=cosmos_video_search_vastdb
```

## Create a collection

Same API as VastDB — only the pipeline id changes:

```bash
curl -s -X POST "http://localhost:8888/v1/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "cosmos_video_search_vastdb",
    "name": "VastDB Video Collection",
    "tags": {
      "storage-template": "s3://cds-media/videos/{{filename}}",
      "vector-backend": "vastdb"
    },
    "collection_config": {},
    "index_config": {},
    "metadata_config": {
      "allow_dynamic_schema": true,
      "fields": []
    }
  }'
```

CLI:

```bash
cds collections create \
  --pipeline cosmos_video_search_vastdb \
  --name "VastDB Video Collection"
```

Notes:

- VastDB `index_config` (GPU_CAGRA / IVF_SQ8) is accepted but **ignored**.
  VastDB performs similarity search with native distance functions.
- Each CDS collection becomes a VastDB **table** under
  `VASTDB_BUCKET` / `VASTDB_SCHEMA`.
- Collection metadata is stored in the `_collection_registry` table.

## Ingest and search

Ingest and search paths are identical to the stock user guide
([API user guide](api-user-guide.md), [CLI user guide](cli-user-guide.md)):

```bash
# Text search
cds search --collection-ids <ID> --text-query "person walking outdoors" --top-k 5
```

Under the hood:

1. CE1 embeds the query to a **256-d** vector.
2. CDS runs ADBC SQL against VastDB, for example:

```sql
SELECT id, content, meta_json,
       array_negative_inner_product(
         embedding::FLOAT[256],
         ARRAY[...]::FLOAT[256]
       ) AS distance
FROM "cds-vastdb/cds"."<collection_table>"
ORDER BY distance ASC
LIMIT 5;
```

Default similarity for the VastDB pipeline is **IP** (inner product), matching
the VastDB pipeline. Cosine / L2 are supported by setting `similarity` on the
document store.

## Environment reference

| Variable | Purpose | Example |
| --- | --- | --- |
| `VASTDB_ENDPOINT` | VastDB / Query Engine endpoint | `http://10.0.0.10` |
| `VAST_ACCESS_KEY` / `VAST_SECRET_KEY` | DB credentials | S3-style keys |
| `VASTDB_BUCKET` | VastDB bucket | `cds-vastdb` |
| `VASTDB_SCHEMA` | Schema name | `cds` |
| `VASTDB_EMBEDDING_DIM` | Must match CE1 (256) | `256` |
| `VAST_ADBC_DRIVER_PATH` | Path to `libadbc_driver_vastdb.so` | `/opt/adbc-driver/...` |
| `VAST_S3_ENDPOINT` | Vast S3 endpoint | `http://10.0.0.20` |
| `AWS_ENDPOINT_URL` | Used by CDS media fetch | same as Vast S3 |
| `ALLOWED_PIPELINES` | Enable VastDB pipeline | `cosmos_video_search_vastdb` |

You can enable **both** backends during migration:

```bash

```

## VastDB → VastDB mapping

| CDS / VastDB concept | VastDB |
| --- | --- |
| VastDB collection | VastDB table |
| `FLOAT_VECTOR` field | `list<float32>[dim]` column |
| `GPU_CAGRA` / `IVF_*` index | Native Query Engine vector search |
| Metric `IP` | `array_negative_inner_product` |
| Metric `COSINE` | `array_cosine_distance` |
| Metric `L2` | `array_distance` |
| `_collection_registry` | `_collection_registry` table |
| Vast S3 / MinIO | **Vast S3** |
| `VASTDB_ENDPOINT` | `VASTDB_ENDPOINT` |

## Optional: DataEngine serverless ingest

For large corpora, offload segment → embed → write to a VAST DataEngine
pipeline (same pattern as the [VSS blueprint](https://github.com/vast-data/vss-blueprint)).
CDS then becomes a **retrieval-only** front-end over VastDB.

See the DataEngine section in [vastdb-architecture.md](vastdb-architecture.md)
and the technical blog.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Pipeline disabled / `_MISSING_VASTDB_*` | Env vars not set before process start |
| ADBC connect fails | Driver path, VIP reachability, access keys |
| Dimension mismatch | CE1 is 256-d; keep `embedding_dim: 256` |
| Media fetch blocked | `CDS_FETCH_S3_ENDPOINTS` / `CDS_FETCH_PRIVATE_ORIGINS` must include Vast S3 |
| SDK `select()` errors on vector tables | `vastdb_patch` is imported by the document store; ensure you use this branch |

## Next steps

- [VastDB architecture & flows](vastdb-architecture.md)
- [Technical blog: CDS on VastDB](../blog/cds-on-vastdb.md)
- Stock CDS [architecture](architecture.md) (VastDB baseline)
