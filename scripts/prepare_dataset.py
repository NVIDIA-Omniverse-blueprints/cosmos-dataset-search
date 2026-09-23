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

    if max_records:
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
