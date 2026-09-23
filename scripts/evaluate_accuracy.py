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

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Dict, Iterable, List, Optional, Tuple
import zipfile
from huggingface_hub import hf_hub_download
import boto3
from tqdm import tqdm

# Local imports from repo
try:
    from src.visual_search.client import Client
    from src.visual_search.client.client import load_profile, get_headers
except Exception as exc:
    print("ERROR: Failed to import visual-search client. Did you run from repo root?", file=sys.stderr)
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("benchmark-accuracy")

DEFAULT_COMPOSE = str(Path("deploy/standalone/docker-compose.build.yml").resolve())
DEFAULT_PROFILE = "local"
DEFAULT_PIPELINE = "cosmos_video_search_milvus"
DEFAULT_BUCKET = "cosmos-test-bucket"
DEFAULT_S3_ENDPOINT = "http://localstack:4566"
RELEVANCE_CHOICES = ["instance", "class"]

# ---------------------------------------------------------------------------
# MSR-VTT helper – download MSRVTT_Videos.zip once and extract it
# ---------------------------------------------------------------------------

def ensure_msrvtt_videos(target_dir: Path, repo: str = "friedrichor/MSR-VTT", zip_name: str = "MSRVTT_Videos.zip") -> Path:
    """Ensure that *all* MSR-VTT mp4s are available locally.

    If ``target_dir`` is already populated (marker file ``.videos_extracted``)
    we return immediately.  Otherwise we download *MSRVTT_Videos.zip* from the
    given *Hugging Face* repo and extract it.

    Parameters
    ----------
    target_dir: Path
        Directory where the mp4s will be extracted.  Created if missing.
    repo: str
        Dataset repository on the HF Hub that contains the ZIP.
    zip_name: str
        Exact file name of the ZIP inside the repo.
    """

    target_dir = target_dir.expanduser().resolve()
    done_flag = target_dir / ".videos_extracted"
    if done_flag.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s from HF Hub (repo=%s) …", zip_name, repo)
    zip_path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=zip_name,
            repo_type="dataset", 
            resume_download=True,
        )
    )

    LOGGER.info("Extracting %s → %s … (this may take a while)", zip_path, target_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    done_flag.touch()
    LOGGER.info("MSR-VTT videos ready in %s", target_dir)
    return target_dir

def try_import_hf_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        LOGGER.error("HuggingFace datasets not installed. Please run: pip install datasets pandas")
        sys.exit(2)

def load_msrvtt_test_split(dataset_name: str, split: str = "test", config: str | None = None):
    from datasets import get_dataset_config_names, load_dataset  # type: ignore

    # Auto-select config when not provided
    if config is None:
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            config = configs[0]
        else:
            # heuristic: pick config containing the split keyword (e.g. "test")
            candidates = [c for c in configs if split in c]
            if not candidates:
                raise ValueError(
                    f"Config name is required for {dataset_name}. Available: {configs}."
                )
            config = candidates[0]

    ds = load_dataset(dataset_name, config, split=split)
    def extract_video_path(row):
        v = row.get("video")
        if isinstance(v, dict) and "path" in v:
            return v["path"]
        if isinstance(v, str):
            return v
        # Other possible keys
        for k in ("video_path", "filepath", "file", "path"):
            if k in row:
                return row[k]
        raise KeyError("Could not infer video file path field in dataset row")

    def extract_video_id(row):
        for k in ("video_id", "videoid", "vid", "id"):
            if k in row:
                return str(row[k])
        # fallback: derive from filename
        p = Path(extract_video_path(row)).name
        return Path(p).stem

    def extract_caption(row):
        for k in ("sentence", "text", "caption", "query"):
            if k in row:
                return str(row[k])
        raise KeyError("Could not infer caption/text field in dataset row")

    records: List[Tuple[str, str, str]] = []  # (video_id, video_path, caption)
    for r in ds:
        try:
            vid = extract_video_id(r)
            vpath = extract_video_path(r)
            cap = extract_caption(r)
            records.append((vid, vpath, cap))
        except Exception as e:
            LOGGER.warning("Skipping row due to error inferring fields: %s", e)
    return records

# ---------------------------------------------------------------------------
# Generic dataset loaders
# ---------------------------------------------------------------------------

def load_hf_dataset_records(
    dataset_name: str,
    split: str = "test",
    config: str | None = None,
    *,
    id_field: str = "video_id",
    video_field: str = "video",
    text_field: str = "sentence",
) -> List[Tuple[str, str, str]]:
    from datasets import get_dataset_config_names, load_dataset  # type: ignore

    # Auto-select config when not provided – same logic as above
    if config is None:
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            config = configs[0]
        else:
            candidates = [c for c in configs if split in c]
            if not candidates:
                raise ValueError(
                    f"Config name is required for {dataset_name}. Available: {configs}."
                )
            config = candidates[0]

    ds = load_dataset(dataset_name, config, split=split)

    def extract(col_name: str, row):
        if col_name in row:
            return row[col_name]
        return None

    records: List[Tuple[str, str, str]] = []
    for r in ds:
        try:
            vid = extract(id_field, r)
            if vid is None:
                vid = Path(str(r.get(video_field, ""))).stem
            vpath = extract(video_field, r)
            if vpath is None:
                vpath = extract("video_path", r) or extract("filepath", r) or extract("file", r) or extract("path", r)
            cap = extract(text_field, r) or extract("caption", r) or extract("text", r) or extract("sentence", r) or extract("query", r)

            if vid is None or vpath is None or cap is None:
                raise KeyError("Missing required columns in dataset row")

            records.append((str(vid), str(vpath), str(cap)))
        except Exception as e:
            LOGGER.warning("Skipping row due to error: %s", e)

    return records

def load_local_dataset(
    file_path: str,
    *,
    id_field: str = "video_id",
    video_field: str = "video",
    text_field: str = "caption",
) -> List[Tuple[str, str, str]]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    records: List[Tuple[str, str, str]] = []

    try:
        if ext in {".json", ".jsonl"}:
            import json

            with path.open() as fp:
                # Detect JSON list vs JSON lines
                first_char = fp.read(1)
                fp.seek(0)
                if first_char == "[":
                    data = json.load(fp)
                    rows = data if isinstance(data, list) else []
                else:
                    rows = [json.loads(line) for line in fp if line.strip()]

            for r in rows:
                vid = r.get(id_field)
                vpath = r.get(video_field)
                cap = r.get(text_field)
                if vid and vpath and cap:
                    records.append((str(vid), str(vpath), str(cap)))
        else:
            import csv  # pylint: disable=import-outside-toplevel

            delimiter = "\t" if ext == ".tsv" else ","
            with path.open(newline="") as fp:
                reader = csv.DictReader(fp, delimiter=delimiter)
                for r in reader:
                    vid = r.get(id_field)
                    vpath = r.get(video_field)
                    cap = r.get(text_field)
                    if vid and vpath and cap:
                        records.append((str(vid), str(vpath), str(cap)))
    except Exception as exc:  # pragma: no cover – best-effort loader
        LOGGER.error("Failed to parse dataset file %s: %s", path, exc)
        raise

    return records

def upload_unique_videos_to_s3(
    records: List[Tuple[str, str, str]],
    bucket: str,
    endpoint_url: str,
    *,
    prefix: str = "dataset",
) -> Tuple[str, Dict[str, str]]:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        # LocalStack special-case for us-east-1
        if (os.environ.get("AWS_DEFAULT_REGION", "us-east-1") or "us-east-1") == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")})

    uploaded: Dict[str, str] = {}
    seen_files: set[str] = set()

    for video_id, video_path, _ in tqdm(records, desc="Uploading videos"):
        if video_path in seen_files:
            continue
        seen_files.add(video_path)
        src = Path(video_path)
        if not src.exists():
            # Some datasets lazy-download on access; force read to trigger download if the Video feature stores remote URL
            LOGGER.warning("Video path not found locally, skipping: %s", video_path)
            continue
        key = f"{prefix}/{src.name}"
        # Skip upload when object already present – speeds up reruns with bind-mounted dataset
        try:
            s3.head_object(Bucket=bucket, Key=key)
            uploaded[key] = video_id
            continue  # object exists, no need to upload
        except s3.exceptions.ClientError as exc:  # noqa: E501  pylint: disable=broad-except
            if exc.response.get("Error", {}).get("Code") != "404":
                raise  # unexpected error

        s3.upload_file(str(src), bucket, key)
        uploaded[key] = video_id

    LOGGER.info("Uploaded %d unique videos to s3://%s/%s", len(uploaded), bucket, prefix)
    return prefix, uploaded

def build_metadata_json(tmpdir: Path, s3_mapping: Dict[str, str]) -> Path:
    mapping: Dict[str, Dict[str, str]] = {}
    for key, vid in s3_mapping.items():
        rel_name = Path(key).name
        mapping[rel_name] = {"video_id": vid}
    out = tmpdir / "msrvtt_metadata.json"
    with out.open("w") as fp:
        json.dump(mapping, fp)
    return out

@dataclass
class Metrics:
    recall_at_1: float
    precision_at_1: float
    recall_at_k: float
    precision_at_k: float
    f1_at_k: float
    mrr: float
    ndcg: float

def evaluate(
    client: Client,
    collection_id: str,
    queries: Iterable[Tuple[str, str]],
    top_k: int = 10,
    *,
    relevance_level: str = "instance",
    label_to_ids: Dict[str, set[str]] | None = None,
) -> Metrics:
    """Compute retrieval metrics given (text, video_id) pairs."""
    import math

    total = 0
    hits_at_1 = 0
    hits_at_k = 0
    sum_precision_at_k = 0.0
    sum_f1_at_k = 0.0
    sum_rr = 0.0
    sum_ndcg = 0.0

    for text, video_id in queries:
        total += 1
        resp = client.search(collection_ids=[collection_id], text_query=text, top_k=top_k, profile=DEFAULT_PROFILE)
        retrieved = resp.get("retrievals", [])

        # determine relevance set
        if relevance_level == "instance" or label_to_ids is None:
            relevant_ids = {str(video_id)}
        else:
            relevant_ids = {str(v) for v in label_to_ids.get(text, set())}

        found_rank = None
        hits_in_topk = 0
        for idx, doc in enumerate(retrieved, start=1):
            meta = doc.get("metadata") or {}
            if str(meta.get("video_id")) in relevant_ids:
                if found_rank is None:
                    found_rank = idx
                hits_in_topk += 1
        # keep earlier logic but account multiple hits
        if found_rank is not None:
            if found_rank == 1:
                hits_at_1 += 1
            if found_rank <= top_k:
                hits_at_k += 1
                precision_k = hits_in_topk / float(top_k)
                recall_k = hits_in_topk / float(len(relevant_ids))
                if precision_k + recall_k > 0:
                    f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
                else:
                    f1_k = 0.0
                sum_precision_at_k += precision_k
                sum_f1_at_k += f1_k
                sum_rr += 1.0 / float(found_rank)
                # NDCG with binary relevance and a single relevant item
                dcg = 1.0 / math.log2(found_rank + 1.0)
                # ideal DCG depends on how many relevant docs exist
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), top_k)))
                sum_ndcg += dcg / idcg
        else:
            # Not found in top_k
            sum_precision_at_k += 0.0
            sum_f1_at_k += 0.0
            sum_rr += 0.0
            sum_ndcg += 0.0

    if total == 0:
        return Metrics(0, 0, 0, 0, 0, 0, 0)

    recall_at_1 = hits_at_1 / total
    precision_at_1 = recall_at_1  # single relevant item scenario
    recall_at_k = hits_at_k / total
    precision_at_k = sum_precision_at_k / total
    f1_at_k = sum_f1_at_k / total
    mrr = sum_rr / total
    ndcg = sum_ndcg / total
    return Metrics(recall_at_1, precision_at_1, recall_at_k, precision_at_k, f1_at_k, mrr, ndcg)

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Cosmos-Embed text-to-video retrieval on MSR-VTT test split")
    # Docker-compose orchestration has been removed – the test harness is
    # supposed to start the stack externally, so we no longer expose
    # compose-file / spawn flags here.
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--dataset", default="friedrichor/MSR-VTT", help="HuggingFace dataset name (ignored when --dataset-file is used)")
    parser.add_argument("--dataset-file", default=None, help="Path to a local JSON/CSV dataset file")
    parser.add_argument("--config", default=None, help="Optional dataset config name (e.g. test_1k)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of queries for a quick run (0=all)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="LocalStack S3 bucket name")
    parser.add_argument("--s3-endpoint", default=DEFAULT_S3_ENDPOINT, help="S3 endpoint URL (LocalStack)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Client profile to use")
    parser.add_argument("--pipeline-id", default=DEFAULT_PIPELINE, help="Visual-search pipeline id to query")
    parser.add_argument("--id-field", default="video_id", help="Column name for video ids")
    parser.add_argument("--video-field", default="video", help="Column name for video file path")
    parser.add_argument("--text-field", default="caption", help="Column name for caption / query text")
    parser.add_argument("--relevance-level", choices=RELEVANCE_CHOICES, default="instance", help="Relevance definition: 'instance' (default) exact video, or 'class' any video sharing the caption label")
    parser.add_argument("--tmpdir", default=None, help="Optional directory for temporary files (defaults to system temp)")
    parser.add_argument(
        "--video-dir",
        default=None,
        help=(
            "Directory containing extracted MSR-VTT mp4s.  If omitted the script will "
            "automatically download and extract MSRVTT_Videos.zip into a cache directory."
        ),
    )
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------
    # Resolve video directory (optional)
    # ---------------------------------------------------------------------

    if args.video_dir:
        video_dir: Optional[Path] = Path(args.video_dir).expanduser().resolve()
        if not video_dir.exists():
            LOGGER.error("Provided --video-dir '%s' does not exist", video_dir)
            return 2
    else:
        # Auto-download MSR-VTT only when evaluating that dataset, otherwise leave unset
        if args.dataset_file is None and "msr-vtt" in args.dataset.lower():
            video_dir = ensure_msrvtt_videos(Path(tempfile.gettempdir()) / "msrvtt_videos")
        else:
            video_dir = None  # rely on absolute/relative paths as provided by dataset

    # Ensure HuggingFace datasets available when needed
    if args.dataset_file is None:
        try_import_hf_datasets()

    # Ensure env for LocalStack S3 (so presigned URLs use localstack host)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_ENDPOINT_URL", args.s3_endpoint)

    # We assume docker-compose services are already running. Any health
    # checks should be executed by the outer orchestration layer.

    client = Client()
    # Ensure pipeline exists
    pipes = client.pipelines.list(profile=args.profile)
    ids = {p["id"] for p in pipes["pipelines"]}
    if args.pipeline_id not in ids:
        LOGGER.error("Pipeline %s not available. Got: %s", args.pipeline_id, ids)
        return 3

    collection_id = None
    tmp_root: Optional[Path] = None
    try:
        # Create collection
        dataset_slug = Path(args.dataset_file).stem if args.dataset_file else args.dataset.split("/")[-1]
        resp = client.collections.create(pipeline=args.pipeline_id, name=f"{dataset_slug} Test", profile=args.profile)
        collection_id = resp["collection"]["id"]
        LOGGER.info("Collection created: %s", collection_id)

        # Load dataset
        if args.dataset_file:
            LOGGER.info("Loading local dataset from %s", args.dataset_file)
            records = load_local_dataset(
                args.dataset_file,
                id_field=args.id_field,
                video_field=args.video_field,
                text_field=args.text_field,
            )
        else:
            LOGGER.info("Loading HF dataset %s split=%s", args.dataset, args.split)
            records = load_hf_dataset_records(
                args.dataset,
                args.split,
                config=args.config,
                id_field=args.id_field,
                video_field=args.video_field,
                text_field=args.text_field,
            )

        # Map video filenames to actual paths – handle cases where the ZIP
        # contains nested directories such as "videos/all/".
        if video_dir is not None:
            resolved_records = []
            for vid, vpath, cap in records:
                p = Path(vpath)
                if p.is_absolute() and p.exists():
                    resolved_path = p
                else:
                    basename = p.name
                    candidate = video_dir / basename
                    if not candidate.exists():
                        found = list(video_dir.rglob(basename))
                        if found:
                            candidate = found[0]
                    resolved_path = candidate
                resolved_records.append((vid, str(resolved_path), cap))
            records = resolved_records
        if args.limit and args.limit > 0:
            # subselect queries but keep all unique videos present in limited queries
            query_records = records[: args.limit]
            limited_video_ids = set(vid for vid, _, _ in query_records)
            records_for_upload = [r for r in records if r[0] in limited_video_ids]
        else:
            query_records = records
            records_for_upload = records

        # Upload unique videos to S3
        prefix, mapping = upload_unique_videos_to_s3(
            records_for_upload,
            args.bucket,
            args.s3_endpoint,
            prefix=dataset_slug,
        )

        # Build metadata mapping JSON (keyed by full s3 key). client.ingest.files will strip base_path
        tmp_root = Path(args.tmpdir) if args.tmpdir else Path(tempfile.mkdtemp(prefix="msrvtt_acc_"))
        metadata_json = build_metadata_json(tmp_root, mapping)

        # Ingest
        s3_path = f"s3://{args.bucket}/{prefix}"
        LOGGER.info("Ingesting from %s (videos=%d)", s3_path, len(mapping))
        stats = client.ingest.files(
            directory_path=s3_path,
            collection_id=collection_id,
            num_workers=4,
            batch_size=1,
            metadata_json=metadata_json,
            strip_directory_path=True,   # use basename as key into metadata
            extensions=[".mp4", ".mov", ".mkv"],
            profile=args.profile,
            s3_profile=None,
            timeout=600,
            output_log=str(tmp_root / "ingest_log.csv"),
            existence_check="skip",
        )
        LOGGER.info("Ingest status codes: %s", dict(stats))

        # Force flush for immediate searchability
        cfg = load_profile(args.profile)
        headers = get_headers(cfg)
        import requests
        flush_resp = requests.post(
            f"{cfg.api_endpoint}/v1/admin/collections/{collection_id}/flush",
            headers=headers,
            verify=False,
        )
        flush_resp.raise_for_status()

        # Prepare queries (text, video_id)
        queries: List[Tuple[str, str]] = [(cap, vid) for (vid, _vpath, cap) in query_records]
        # Build label -> set(video_id) map for class-level relevance
        label_to_ids: Dict[str, set[str]] = defaultdict(set)
        for vid, _vp, cap in records:
            label_to_ids[cap].add(vid)

        # Evaluate
        LOGGER.info("Evaluating %d queries (top_k=%d)...", len(queries), args.top_k)
        metrics = evaluate(client, collection_id, queries, top_k=args.top_k, relevance_level=args.relevance_level, label_to_ids=label_to_ids)

        print("")
        print("=== MSR-VTT Retrieval Accuracy ===")
        print(f"Queries: {len(queries)}")
        print(f"Recall@1:      {metrics.recall_at_1:.4f}")
        print(f"Precision@1:   {metrics.precision_at_1:.4f}")
        print(f"Recall@{args.top_k}:    {metrics.recall_at_k:.4f}")
        print(f"Precision@{args.top_k}: {metrics.precision_at_k:.4f}")
        print(f"F1@{args.top_k}:       {metrics.f1_at_k:.4f}")
        print(f"MRR:           {metrics.mrr:.4f}")
        print(f"NDCG@{args.top_k}:     {metrics.ndcg:.4f}")

        # Also dump as JSON for CI artifact consumption
        result_json = {
            "queries": len(queries),
            "top_k": args.top_k,
            "recall_at_1": metrics.recall_at_1,
            "precision_at_1": metrics.precision_at_1,
            "recall_at_k": metrics.recall_at_k,
            "precision_at_k": metrics.precision_at_k,
            "f1_at_k": metrics.f1_at_k,
            "mrr": metrics.mrr,
            "ndcg_at_k": metrics.ndcg,
        }
        out_file = (tmp_root or Path.cwd()) / "msrvtt_accuracy.json"
        with open(out_file, "w") as fp:
            json.dump(result_json, fp, indent=2)
        LOGGER.info("Saved metrics JSON to %s", out_file)

    except Exception as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        return 1
    finally:
        # Cleanup collection
        try:
            if collection_id:
                client.collections.delete(collection_id=collection_id, profile=args.profile)
        except Exception:
            pass
        if tmp_root and tmp_root.exists():
            try:
                shutil.rmtree(tmp_root)
            except Exception:
                pass

    return 0

if __name__ == "__main__":
    sys.exit(main())

