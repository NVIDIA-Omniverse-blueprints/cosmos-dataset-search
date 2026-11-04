# VastDB architecture for Cosmos Dataset Search

This page describes how CDS runs when the vector backend is **VastDB** and
media lives on **Vast S3**. Compare with the stock [VastDB architecture](architecture.md).

## System overview

```mermaid
flowchart LR
  subgraph Clients
    UI[Web UI / CLI]
    API_CLIENT[REST clients]
  end

  subgraph CDS["CDS Visual Search"]
    CORE[Core API<br/>collections / ingest / search]
  end

  subgraph NVIDIA
    CE1[Cosmos Embed CE1<br/>256-d embeddings]
  end

  subgraph VAST["VAST AI OS"]
    S3[Vast S3<br/>videos / frames]
    VDB[(VastDB<br/>vectors + registry)]
    QE[Query Engine<br/>ADBC vector SQL]
  end

  UI --> CORE
  API_CLIENT --> CORE
  CORE -->|embed text/video| CE1
  CORE -->|read/write media| S3
  CORE -->|insert embeddings| VDB
  CORE -->|similarity search| QE
  QE --> VDB
```

**What changed vs VastDB CDS**

- Removed: VastDB, etcd, Vast S3/MinIO (for the VastDB profile).
- Added: VastDB tables + ADBC Query Engine path.
- Replaced: object storage endpoint → Vast S3 (same S3 API, VAST-backed).
- Unchanged: CE1, CDS REST/CLI contracts, collection CRUD semantics.

## Ingest flow

```mermaid
sequenceDiagram
  participant C as Client
  participant CDS as CDS API
  participant CE1 as Cosmos Embed
  participant S3 as Vast S3
  participant VDB as VastDB

  C->>CDS: POST /collections/{id}/documents
  CDS->>S3: optional media fetch / store URI
  CDS->>CE1: embed video / frames
  CE1-->>CDS: 256-d embedding
  CDS->>VDB: insert row<br/>(id, content, meta_json, embedding)
  VDB-->>CDS: commit
  CDS-->>C: indexed documents
```

Each collection maps to one VastDB table:

| Column | Type | Role |
| --- | --- | --- |
| `id` | string | Document primary key |
| `content` | string | Optional text payload |
| `meta_json` | string | Dynamic CDS document metadata |
| `embedding` | `list<float32>[256]` | CE1 vector |

Collection records (name, pipeline, tags, …) live in `_collection_registry`.

## Search flow

```mermaid
sequenceDiagram
  participant C as Client
  participant CDS as CDS API
  participant CE1 as Cosmos Embed
  participant QE as Vast Query Engine
  participant VDB as VastDB

  C->>CDS: POST /collections/{id}/search
  CDS->>CE1: embed text or video query
  CE1-->>CDS: 256-d query vector
  CDS->>QE: ADBC SQL ORDER BY distance LIMIT k
  QE->>VDB: scan / prune vector chunks
  VDB-->>QE: top-k rows + distance
  QE-->>CDS: Arrow results
  CDS-->>C: ranked documents + scores
```

Distance functions (see
[Vector Search docs](https://kb.vastdata.com/documentation/docs/vector-search-55)):

| CDS similarity | VastDB SQL |
| --- | --- |
| `IP` / `dot_product` | `array_negative_inner_product` |
| `cosine` | `array_cosine_distance` |
| `l2` / `euclidean` | `array_distance` |

## Storage layout on VAST

```mermaid
flowchart TB
  subgraph VastS3["Vast S3 buckets"]
    MEDIA[cds-media / videos / frames]
  end

  subgraph VastDBBucket["VastDB bucket: cds-vastdb"]
    subgraph Schema["schema: cds"]
      REG[_collection_registry]
      T1[collection table A]
      T2[collection table B]
    end
  end

  MEDIA -. URIs in meta_json .-> T1
  MEDIA -. URIs in meta_json .-> T2
  REG --> T1
  REG --> T2
```

## Optional: DataEngine serverless ingest

For hyperscale or continuous video landing, push heavy ingest off the CDS API
node onto **VAST DataEngine** (same pattern as the
[VSS blueprint](https://github.com/vast-data/vss-blueprint)):

```mermaid
flowchart LR
  LAND[Video lands on Vast S3] --> SEG[DataEngine<br/>segmenter]
  SEG --> EMB[DataEngine<br/>embedder / CE1 client]
  EMB --> W[DataEngine<br/>vastdb-writer]
  W --> VDB[(VastDB)]
  VDB --> CDS[CDS retrieval API]
```

**When this helps**

- Billions of clips / long-running curation (Figure AI–style loops).
- Burst ingest without sizing the CDS API replica set for peak CPU.
- Reuse of existing VAST event triggers and identity policies.

**What stays in CDS**

- Collection registry semantics, search API, CE1 query embedding, UI/CLI.

This repository ships the **in-process VastDB connector** first. A starter
DataEngine pipeline is documented as a follow-on (see the blog).

## Component map

| CDS component | VastDB path |
| --- | --- |
| Pipeline | `cosmos_video_search_vastdb` |
| Document store | `src/haystack/components/vastdb/document_store.py` |
| Writer | `VastDocumentWriter` |
| Retriever | `VastEmbeddingRetriever` |
| Compose overlay | `deploy/standalone/docker-compose.vastdb.yml` |
| Env template | `deploy/standalone/.env.vastdb.example` |

## Further reading

- Customer how-to: [VastDB vector store](vastdb-vector-store.md)
- [VAST Vector Search announcement](https://www.vastdata.com/blog/introducing-vast-vector-search-real-time-ai-retrieval-without-limits)
- [Vector DB Quickstart](https://vast-data.github.io/data-platform-field-docs/vast_vectordb/quickstart-new.html)
- [vastdb SDK](https://github.com/vast-data/vastdb_sdk) · [ADBC driver](https://github.com/vast-data/vastdb-adbc-driver)
