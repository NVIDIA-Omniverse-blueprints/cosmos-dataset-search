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

import argparse
import csv
import json
import os
import statistics
import sys
import time
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest video URLs into a collection. Provide a single --video-url or a --csv file with 'name,url' rows."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", "http://localhost:8000"),
        help="Service base URL (default: env BASE_URL or http://localhost:8000)",
    )
    parser.add_argument("--collection-id", required=True, help="Target collection ID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-url", help="Single video URL to ingest")
    group.add_argument(
        "--csv",
        dest="csv_path",
        help="CSV file path with rows formatted as name,url (header optional)",
    )
    parser.add_argument(
        "--mime",
        default="video/mp4",
        help="Mime type to use for all items (default: video/mp4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Max documents per request (<= backend limit; default: 50)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on number of videos to process from CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output: print requests and responses, and per-URL checks",
    )
    parser.add_argument(
        "--precheck-urls",
        action="store_true",
        help="Before uploading, try a quick GET to validate each URL is reachable",
    )
    parser.add_argument(
        "--skip-bad",
        action="store_true",
        help="When prechecking URLs, skip any that fail instead of aborting",
    )
    parser.add_argument(
        "--cosmos-embed-base-url",
        dest="embed_base_url",
        default=None,
        help="Optional Cosmos Embed service base URL for direct embedding timing (e.g., http://desktop:9000)",
    )
    parser.add_argument(
        "--cosmos-embed-model",
        dest="embed_model",
        default="nvidia/cosmos-embed1",
        help="Model to use when calling the Cosmos Embed service embeddings endpoint",
    )
    parser.add_argument(
        "--measure-embed-delta",
        action="store_true",
        help="If set with --cosmos-embed-base-url, times each batch on the Cosmos Embed service directly and reports CDS overhead",
    )
    parser.add_argument(
        "--require-video-content-type",
        action="store_true",
        help="Fail or skip URLs whose Content-Type is not video/* during precheck",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write a CSV report (summary and per-batch details)",
    )
    parser.add_argument(
        "--flush-per-batch",
        action="store_true",
        help="If set, flush after each batch (default: single final flush only)",
    )
    parser.add_argument(
        "--adjust-cds-per-batch",
        action="store_true",
        help="When not flushing per batch, use average CDS time per batch for per-batch deltas",
    )
    # URL mode is the only supported approach: DocumentUploadUrl (url + mime_type)
    return parser.parse_args()


def read_csv_name_url(csv_path: str) -> List[Tuple[str, str]]:
    """Read CSV rows of [name,url]. Header 'name,url' allowed but not required."""
    rows: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 2:
                print(
                    f"Skipping invalid row (expected name,url): {row}", file=sys.stderr
                )
                continue
            name, url = row[0].strip(), row[1].strip()
            # Skip header if present
            if rows == [] and name.lower() == "name" and url.lower() == "url":
                continue
            rows.append((name, url))
    return rows


def chunked(
    iterable: List[Tuple[str, str]], size: int
) -> Iterable[List[Tuple[str, str]]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def build_payload(items: List[Tuple[str, str]], mime: str) -> list:
    payload = []
    for name, url in items:
        doc = {
            "url": url,
            "mime_type": mime,
            "metadata": {"source_name": name},
        }
        payload.append(doc)
    return payload


def post_documents(
    base_url: str, collection_id: str, payload: list, verbose: bool = False
) -> dict:
    url = f"{base_url}/v1/collections/{collection_id}/documents"
    if verbose:
        preview = payload[:3]
        print("POST", url)
        print("Payload preview (first 3 docs):")
        print(json.dumps(preview, indent=2))
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        if verbose:
            print("Request that failed:")
            print(json.dumps(payload, indent=2))
        raise
    return resp.json()


def flush_collection(base_url: str, collection_id: str, verbose: bool = False) -> dict:
    """Force Milvus flush for the collection and return response JSON."""
    url = f"{base_url}/v1/admin/collections/{collection_id}/flush"
    if verbose:
        print("POST", url, "(flush)")
    resp = requests.post(
        url, headers={"Accept": "application/json"}, verify=False, timeout=120
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"Flush failed HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        raise
    return resp.json()


def embed_batch_latency(
    embed_base_url: str,
    model: str,
    urls: List[str],
    mime: str,
    timeout_s: int = 600,
) -> float:
    """Call Cosmos Embed service embeddings for a batch of presigned URL data URIs; return latency seconds."""
    ext = mime.split("/")[-1] if "/" in mime else "mp4"
    inputs = [f"data:video/{ext};presigned_url,{u}" for u in urls]
    payload = {
        "input": inputs,
        "encoding_format": "float",
        "model": model,
        "request_type": "bulk_video",
    }
    url = f"{embed_base_url}/v1/embeddings"
    t0 = time.perf_counter()
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    t1 = time.perf_counter()
    resp.raise_for_status()
    _ = resp.json()
    return t1 - t0


def warn_if_localhost(urls: List[str]) -> None:
    flagged = []
    for u in urls:
        try:
            host = urlparse(u).hostname or ""
            if host in {"0.0.0.0", "localhost", "127.0.0.1"}:
                flagged.append(u)
        except Exception:
            continue
    if flagged:
        print(
            "Warning: some URLs point to localhost/0.0.0.0 which is often not reachable from the service.\n"
            "Consider regenerating CSV with a reachable interface IP or hostname.",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()

    # Collect items as (name, url)
    items: List[Tuple[str, str]] = []
    if args.video_url:
        items = [
            (
                os.path.basename(args.video_url.rstrip("/")) or args.video_url,
                args.video_url,
            )
        ]
    else:
        items = read_csv_name_url(args.csv_path)
        if not items:
            print("No valid rows found in CSV", file=sys.stderr)
            sys.exit(1)

    # Enforce max videos if provided (only applies to CSV mode)
    if args.csv_path and args.max_videos is not None:
        items = items[: max(0, args.max_videos)]

    # Quick warning for localhost/0.0.0.0 URLs
    warn_if_localhost([u for _, u in items])

    # Optional precheck that URLs are reachable
    if args.precheck_urls:
        ok_items: List[Tuple[str, str]] = []
        bad_items: List[Tuple[str, str]] = []
        for name, url in items:
            try:
                # Prefer HEAD; some servers may not support it; fallback to GET
                r = requests.head(url, timeout=5, allow_redirects=True)
                if r.status_code >= 400 or not r.headers:
                    r = requests.get(url, timeout=5, stream=True)
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if args.require_video_content_type and not ctype.startswith("video/"):
                    raise ValueError(f"Non-video Content-Type: {ctype}")
                ok_items.append((name, url))
                if args.verbose:
                    clen = r.headers.get("Content-Length", "?")
                    print(
                        f"OK {url} (Content-Type={ctype or '?'}, Content-Length={clen})"
                    )
            except Exception as e:
                bad_items.append((name, url))
                print(f"BAD {url}: {e}", file=sys.stderr)
        if bad_items and not args.skip_bad:
            print(
                f"Aborting due to {len(bad_items)} unreachable URL(s). Use --skip-bad to skip them.",
                file=sys.stderr,
            )
            sys.exit(2)
        if bad_items and args.skip_bad:
            print(
                f"Skipping {len(bad_items)} unreachable URL(s); continuing with {len(ok_items)}"
            )
            items = ok_items

    # Prepare batches
    batches: List[List[Tuple[str, str]]] = list(chunked(items, max(1, args.batch_size)))
    total_sent = 0
    results = []
    batch_latencies: List[float] = []  # measured CDS per-batch latency
    batch_sizes: List[int] = []
    embed_batch_latencies: List[float] = []
    cds_minus_embed_deltas: List[float] = []
    per_batch_details = (
        []
    )  # (idx, size, cds_latency_used, embed_latency or nan, delta or nan)

    # Run CDS ingestion for all batches first
    cds_start = time.perf_counter() if batches else None
    final_flush_time_sec: float | None = None
    for batch_idx, batch in enumerate(batches, start=1):
        payload = build_payload(batch, args.mime)
        try:
            t0 = time.perf_counter()  # start CDS batch timing
            result = post_documents(
                args.base_url, args.collection_id, payload, verbose=args.verbose
            )
            if args.flush_per_batch:
                try:
                    _ = flush_collection(
                        args.base_url, args.collection_id, verbose=args.verbose
                    )
                except Exception as e:
                    if args.verbose:
                        print(f"Flush error (batch {batch_idx}): {e}", file=sys.stderr)
                    sys.exit(1)
            t_end = time.perf_counter()
            cds_latency = t_end - t0
            batch_latencies.append(cds_latency)
            batch_sizes.append(len(payload))
            results.append(result)
            total_sent += len(payload)
            print(
                f"Uploaded batch {batch_idx} of {len(payload)} document(s) in {cds_latency:.3f}s. Total so far: {total_sent}"
            )
        except Exception as e:
            if args.verbose:
                print(
                    f"Upload failed for batch of size {len(payload)}: {e}",
                    file=sys.stderr,
                )
            sys.exit(1)

    # Final flush (if not per-batch)
    if cds_start is not None and not args.flush_per_batch:
        try:
            t_flush0 = time.perf_counter()
            _ = flush_collection(
                args.base_url, args.collection_id, verbose=args.verbose
            )
            t_flush1 = time.perf_counter()
            final_flush_time_sec = t_flush1 - t_flush0
        except Exception as e:
            if args.verbose:
                print(f"Final flush error: {e}", file=sys.stderr)
            sys.exit(1)
    cds_end = time.perf_counter() if cds_start is not None else None

    # Now measure Cosmos Embed service direct embedding for same batches (optional)
    if args.measure_embed_delta and args.embed_base_url:
        for batch_idx, batch in enumerate(batches, start=1):
            urls_this_batch = [u for _, u in batch]
            try:
                embed_t = embed_batch_latency(
                    embed_base_url=args.embed_base_url,
                    model=args.embed_model,
                    urls=urls_this_batch,
                    mime=args.mime,
                )
                embed_batch_latencies.append(embed_t)
            except Exception as e:
                if args.verbose:
                    print(
                        f"Cosmos Embed service embed measurement failed for batch {batch_idx}: {e}",
                        file=sys.stderr,
                    )
                embed_batch_latencies.append(float("nan"))

    # Compute CDS per-batch latencies to use for delta
    cds_latencies_used: List[float] = list(batch_latencies)
    if (
        batches
        and not args.flush_per_batch
        and args.adjust_cds_per_batch
        and cds_start is not None
        and cds_end is not None
    ):
        avg_per_batch = (cds_end - cds_start) / len(batches)
        cds_latencies_used = [avg_per_batch for _ in batches]

    # Build per-batch details and deltas
    for i in range(len(batches)):
        cds_t = cds_latencies_used[i]
        embed_t = (
            embed_batch_latencies[i] if i < len(embed_batch_latencies) else float("nan")
        )
        delta = (cds_t - embed_t) if not (embed_t != embed_t) else float("nan")
        cds_minus_embed_deltas.append(delta if not (delta != delta) else float("nan"))
        per_batch_details.append((i + 1, batch_sizes[i], cds_t, embed_t, delta))

    # Print the last response for convenience; or aggregate if many
    if results:
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps({"batches": results[-1]}, indent=2))

    # Performance report
    if batch_latencies:
        total_batches = len(batch_latencies)
        min_latency = min(batch_latencies)
        max_latency = max(batch_latencies)
        avg_latency = statistics.mean(batch_latencies)
        total_videos = sum(batch_sizes)
        # CDS-only time: if we have overall start/end (with final flush), prefer that for throughput
        total_batch_time = (
            (cds_end - cds_start)
            if (cds_start is not None and cds_end is not None)
            else sum(batch_latencies)
        )
        per_video_time = (total_batch_time / total_videos) if total_videos > 0 else 0.0
        # Throughput derived strictly from CDS ingestion time
        throughput = (total_videos / total_batch_time) if total_batch_time > 0 else 0.0

        print("\nPerformance report:")
        print(f"  Total batches processed: {total_batches}")
        print(
            f"  Batch latency (s): min={min_latency:.3f}, max={max_latency:.3f}, avg={avg_latency:.3f}"
        )
        print(f"  Per-video processing time (s): {per_video_time:.3f}")
        print(f"  Overall throughput (videos/sec): {throughput:.3f}")
        print(f"  Overall CDS time incl. final flush (s): {total_batch_time:.3f}")
        if final_flush_time_sec is not None:
            print(f"  Final flush time (s): {final_flush_time_sec:.3f}")

        # Optional Cosmos Embed service vs CDS comparison
        if (
            args.measure_embed_delta
            and args.embed_base_url
            and any(not (t != t) for t in embed_batch_latencies)
        ):
            embed_min = min(embed_batch_latencies)
            embed_max = max(embed_batch_latencies)
            embed_avg = statistics.fmean(
                t for t in embed_batch_latencies if not (t != t)
            )
            delta_avg = (
                statistics.fmean(d for d in cds_minus_embed_deltas if not (d != d))
                if cds_minus_embed_deltas
                else float("nan")
            )
            print(
                "  Cosmos Embed service batch latency (s): min={:.3f}, max={:.3f}, avg={:.3f}".format(
                    embed_min, embed_max, embed_avg
                )
            )
            print(
                "  (cds - cosmos-embed) delta per batch (s): avg={:.3f}".format(
                    delta_avg
                )
            )

        # CSV report
        if args.csv_out:
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["section", "metric", "value"])
                writer.writerow(["cds", "total_batches", total_batches])
                writer.writerow(["cds", "min_latency_sec", f"{min_latency:.6f}"])
                writer.writerow(["cds", "avg_latency_sec", f"{avg_latency:.6f}"])
                writer.writerow(["cds", "max_latency_sec", f"{max_latency:.6f}"])
                writer.writerow(["cds", "per_video_time_sec", f"{per_video_time:.6f}"])
                writer.writerow(["cds", "throughput_vps", f"{throughput:.6f}"])
                writer.writerow(["cds", "overall_time_sec", f"{total_batch_time:.6f}"])
                if final_flush_time_sec is not None:
                    writer.writerow(
                        ["cds", "final_flush_time_sec", f"{final_flush_time_sec:.6f}"]
                    )
                writer.writerow(["cds", "flush_per_batch", args.flush_per_batch])
                writer.writerow(
                    ["cds", "adjust_cds_per_batch", args.adjust_cds_per_batch]
                )
                if (
                    args.measure_embed_delta
                    and args.embed_base_url
                    and any(not (t != t) for t in embed_batch_latencies)
                ):
                    embed_min = min(embed_batch_latencies)
                    embed_max = max(embed_batch_latencies)
                    embed_avg = statistics.fmean(
                        t for t in embed_batch_latencies if not (t != t)
                    )
                    delta_avg = (
                        statistics.fmean(
                            d for d in cds_minus_embed_deltas if not (d != d)
                        )
                        if cds_minus_embed_deltas
                        else float("nan")
                    )
                    writer.writerow(
                        ["cosmos_embed", "min_latency_sec", f"{embed_min:.6f}"]
                    )
                    writer.writerow(
                        ["cosmos_embed", "avg_latency_sec", f"{embed_avg:.6f}"]
                    )
                    writer.writerow(
                        ["cosmos_embed", "max_latency_sec", f"{embed_max:.6f}"]
                    )
                    writer.writerow(
                        ["delta", "cds_minus_embed_avg_sec", f"{delta_avg:.6f}"]
                    )

                # Per-batch details
                writer.writerow([])
                writer.writerow(
                    [
                        "per_batch",
                        "index",
                        "batch_size",
                        "cds_latency_sec_used",
                        "embed_latency_sec",
                        "delta_sec",
                    ]
                )
                for idx, size, cds_t, embed_t, d in per_batch_details:
                    writer.writerow(
                        [
                            "per_batch",
                            idx,
                            size,
                            f"{cds_t:.6f}",
                            (
                                "{:.6f}".format(embed_t)
                                if not (embed_t != embed_t)
                                else "nan"
                            ),
                            ("{:.6f}".format(d) if not (d != d) else "nan"),
                        ]
                    )


if __name__ == "__main__":
    main()
