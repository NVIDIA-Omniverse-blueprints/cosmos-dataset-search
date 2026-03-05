#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Prepare datasets for CDS ingestion.

Supports:
1. HuggingFace datasets with video ZIPs (e.g., MSR-VTT)
   - Requires: video_zip_filename in config
   - Downloads ZIP from HF repo, extracts, and copies videos locally
   
2. HuggingFace datasets with local videos
   - Video files must exist locally
   - Script loads metadata from HF and links to local files
   
3. Local datasets
   - Videos and metadata both local
   - No HuggingFace dependency

Example config (MSR-VTT):
    source: hf
    hf_repo: friedrichor/MSR-VTT
    hf_config: test_1k
    video_zip_filename: MSRVTT_Videos.zip  # Optional - downloads videos from HF
    max_records: 100
    copy_videos_to: ~/datasets/msrvtt/videos
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def download_and_extract_videos(target_dir: Path, repo: str, zip_filename: str) -> Path:
    """Download and extract video ZIP from HuggingFace if needed."""
    target_dir = target_dir.expanduser().resolve()
    done_flag = target_dir / ".videos_extracted"
    
    if done_flag.exists():
        print(f"Videos already extracted in {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {zip_filename} from HuggingFace ({repo})...")
    
    try:
        zip_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=zip_filename,
                repo_type="dataset", 
                resume_download=True,
            )
        )
        print(f"Extracting {zip_path} → {target_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)   
        done_flag.touch()
        print(f"Videos ready in {target_dir}")
        return target_dir
    except Exception as e:
        print(f"Warning: Could not download/extract videos from HuggingFace: {e}")
        print("Continuing without video download - files must be provided locally")
        return target_dir

def copy_if_needed(src: str, dest_dir: Path, video_source_dir: Path = None) -> str:
    """Copy or link video file to destination."""
    src_path = Path(src).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    
    if dest_path.exists():
        return str(dest_path)
    
    # If source doesn't exist, try to find it in video_source_dir
    if not src_path.exists() and video_source_dir:
        # MSR-VTT videos are in a subdirectory called 'video'
        for subdir in [video_source_dir, video_source_dir / "video"]:
            potential_path = subdir / src_path.name
            if potential_path.exists():
                src_path = potential_path
                break
    
    if src_path.exists():
        try:
            # try hard-link first, fall back to copy
            dest_path.hardlink_to(src_path)
        except Exception:
            shutil.copy2(src_path, dest_path)
        return str(dest_path)
    else:
        print(f"Warning: Source file {src_path.name} not found, skipping")
        return str(src_path)

def encode_h265_to_h264(unzip_path_h265: str, unzip_path_h264: str) -> str:
    """Encode H265 to H264."""
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(unzip_path_h264, exist_ok=True)

    mp4_files = [f for f in os.listdir(unzip_path_h265) if f.endswith(".mp4")]
    n_workers = os.cpu_count() or 4
    print(f"Encoding {len(mp4_files)} videos to H264 using {n_workers} workers")

    def convert_one(file: str) -> None:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", f"{unzip_path_h265}/{file}",
                "-c:v", "libx264", "-profile:v", "baseline",
                "-movflags", "+faststart", f"{unzip_path_h264}/{file}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(convert_one, mp4_files),
            total=len(mp4_files),
            desc="H.264 re-encode",
            unit="video",
        ))

    shutil.rmtree(unzip_path_h265)
    return unzip_path_h264

def load_pai_dataset(text_field: str, max_records: int) -> List[Tuple[str, str, str]]:
    """Load PAI dataset from local file."""
    import os
    from physical_ai_av import PhysicalAIAVDatasetInterface
    CACHE_DIR = os.path.join(os.environ["HOME"], ".cache")
    avdi = PhysicalAIAVDatasetInterface(
        revision="ncore_test",
        token=os.environ["HF_TOKEN"],
        cache_dir=CACHE_DIR,
    )

    chunk_ids = list(range(max_records))
    features = ["camera_front_wide_120fov"]
    files_to_download = []
    for cid in chunk_ids:
        for feat in features:
            if avdi.chunk_sensor_presence.at[cid, feat]:
                files_to_download.append(
                    avdi.features.get_chunk_feature_filename(cid, feat)
                )

    local_paths = avdi.download_files(files_to_download, force_download=True)

    unzip_path_h265 = "/".join([i for i in local_paths[0].split("/")[:-1]]) + "_mp4"
    os.system(f"mkdir -p {unzip_path_h265}")
    print(unzip_path_h265)
    for path in local_paths:
        if not str(path).endswith(".zip"):
            continue
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(unzip_path_h265)
        os.system(f"rm {unzip_path_h265}/*parquet")


    ## Temporary fix for CDS to ingest videos with codec H264
    unzip_path_h264 = "/".join([i for i in local_paths[0].split("/")[:-1]]) + "_mp4_h264"
    unzip_path_h264 = encode_h265_to_h264(unzip_path_h265, unzip_path_h264)

    records = []
    for file in os.listdir(unzip_path_h264):
        if not file.endswith(".mp4"):
            continue
        records.append((file.split(".")[0], f"{unzip_path_h264}/{file}", f"{text_field}"))
    return records


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for CDS ingestion")
    parser.add_argument("config", help="Path to YAML or JSON config file")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    with cfg_path.open() as fp:
        cfg: Dict[str, Any] = yaml.safe_load(fp)

    source = cfg.get("source", "hf")
    max_records = int(cfg.get("max_records", 0))
    id_field = cfg.get("id_field", "video_id")
    video_field = cfg.get("video_field", "video")
    text_field = cfg.get("text_field", "caption")

    video_source_dir = None
    
    # For HuggingFace datasets with video ZIPs
    if source == "hf":
        repo = cfg["hf_repo"]
        
        # Check if there's a video ZIP to download
        video_zip = cfg.get("video_zip_filename")
        if video_zip:
            # Download and extract videos
            dataset_name = repo.split('/')[-1].lower().replace('-', '_')
            cache_dir = Path(tempfile.gettempdir()) / f"{dataset_name}_videos"
            video_source_dir = download_and_extract_videos(cache_dir, repo, video_zip)
        
        # Load metadata from HuggingFace dataset
        from datasets import load_dataset
        split = cfg.get("split", "test")
        hf_config = cfg.get("hf_config")
        
        ds = load_dataset(repo, hf_config, split=split) if hf_config else load_dataset(repo, split=split)
        records = [(item[id_field], item[video_field], item[text_field]) for item in ds]
    if source == "pai":
        records = load_pai_dataset(text_field, max_records)
    elif source == "local":
        # Load from local file
        from scripts.evaluate_accuracy import load_local_dataset
        records = load_local_dataset(
            cfg["dataset_file"],
            id_field=id_field,
            video_field=video_field,
            text_field=text_field,
        )
    else:
        raise ValueError(f"Unsupported source {source}")

    if max_records: # and source != "pai":
        records = records[: max_records]

    copy_dir = cfg.get("copy_videos_to")
    if copy_dir:
        dest = Path(copy_dir).expanduser().resolve()
        new_records = []
        for vid, vpath, cap in tqdm(records, desc="Copying videos"):
            new_path = copy_if_needed(vpath, dest, video_source_dir)
            new_records.append((vid, new_path, cap))
        records = new_records

    out_path = Path(cfg.get("output_jsonl", "dataset.jsonl")).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        for vid, vpath, cap in records:
            json.dump({"video_id": vid, "video": vpath, "caption": cap}, fp)
            fp.write("\n")
    print(f"Wrote {len(records)} records -> {out_path}")

if __name__ == "__main__":
    main()
