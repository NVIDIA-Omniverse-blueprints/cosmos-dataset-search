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

import os
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger("dataset-loader")

class DatasetRecord:
    
    def __init__(self, video_id: str, video_path: str, caption: str):
        self.video_id = video_id
        self.video_path = video_path
        self.caption = caption
    
    def __iter__(self):
        return iter((self.video_id, self.video_path, self.caption))
    
    def __repr__(self):
        return f"DatasetRecord(id={self.video_id}, path={self.video_path[:50]}..., caption={self.caption[:50]}...)"

class MSRVTTLoader:
    def __init__(self, repo: str = "friedrichor/MSR-VTT", zip_name: str = "MSRVTT_Videos.zip"):
        self.repo = repo
        self.zip_name = zip_name
    
    def ensure_videos(self, target_dir: Path) -> Path:
        target_dir = target_dir.expanduser().resolve()
        done_flag = target_dir / ".videos_extracted"
        if done_flag.exists():
            LOGGER.info("MSR-VTT videos already extracted in %s", target_dir)
            return target_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading %s from HF Hub (repo=%s) …", self.zip_name, self.repo)
        
        zip_path = Path(
            hf_hub_download(
                repo_id=self.repo,
                filename=self.zip_name,
                repo_type="dataset", 
                resume_download=True,
            )
        )
        LOGGER.info("Extracting %s → %s", zip_path, target_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)   
        done_flag.touch()
        LOGGER.info("MSR-VTT videos ready in %s", target_dir)
        return target_dir

class HuggingFaceDatasetLoader:
    
    @staticmethod
    def _try_import_datasets():
        try:
            import datasets
        except ImportError:
            LOGGER.error("HuggingFace datasets not installed. Please run: pip install datasets pandas")
            raise ImportError("datasets library required for HuggingFace dataset loading")
    
    @staticmethod
    def _auto_select_config(dataset_name: str, split: str, config: Optional[str]) -> str:
        from datasets import get_dataset_config_names
        if config is not None:
            return config
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            return configs[0]
        candidates = [c for c in configs if split in c]
        if not candidates:
            raise ValueError(
                f"Config name is required for {dataset_name}. Available: {configs}."
            )
        return candidates[0]
    
    @classmethod
    def load_msrvtt(cls, split: str = "test", config: Optional[str] = None) -> List[DatasetRecord]:
        cls._try_import_datasets()
        from datasets import load_dataset
        dataset_name = "friedrichor/MSR-VTT"
        config = cls._auto_select_config(dataset_name, split, config)
        ds = load_dataset(dataset_name, config, split=split)
        def extract_video_path(row):
            v = row.get("video")
            if isinstance(v, dict) and "path" in v:
                return v["path"]
            if isinstance(v, str):
                return v
            for k in ("video_path", "filepath", "file", "path"):
                if k in row:
                    return row[k]
            raise KeyError("Could not infer video file path field in dataset row")

        def extract_video_id(row):
            for k in ("video_id", "videoid", "vid", "id"):
                if k in row:
                    return str(row[k])
            p = Path(extract_video_path(row)).name
            return Path(p).stem

        def extract_caption(row):
            for k in ("sentence", "text", "caption", "query"):
                if k in row:
                    return str(row[k])
            raise KeyError("Could not infer caption/text field in dataset row")

        records = []
        for r in ds:
            try:
                vid = extract_video_id(r)
                vpath = extract_video_path(r)
                cap = extract_caption(r)
                records.append(DatasetRecord(vid, vpath, cap))
            except Exception as e:
                LOGGER.warning("Skipping row due to error inferring fields: %s", e)
                
        LOGGER.info("Loaded %d records from MSR-VTT %s split", len(records), split)
        return records

    @classmethod
    def load_generic(
        cls,
        dataset_name: str,
        split: str = "test",
        config: Optional[str] = None,
        *,
        id_field: str = "video_id",
        video_field: str = "video",
        text_field: str = "sentence",
    ) -> List[DatasetRecord]:
        cls._try_import_datasets()
        from datasets import load_dataset
        
        config = cls._auto_select_config(dataset_name, split, config)
        ds = load_dataset(dataset_name, config, split=split)

        def extract(col_name: str, row):
            if col_name in row:
                return row[col_name]
            return None
        
        LOGGER.info("Loading dataset %s %s split", dataset_name, split)
        records = []
        for r in ds:
            try:
                vid = extract(id_field, r)
                if vid is None:
                    vid = Path(str(r.get(video_field, ""))).stem
                LOGGER.info("Extracted video ID: %s", vid)
                vpath = extract(video_field, r)
                if vpath is None:
                    for alt_field in ["video_path", "filepath", "file", "path"]:
                        vpath = extract(alt_field, r)
                        if vpath is not None:
                            break
                LOGGER.info("Extracted video path: %s", vpath)
                cap = extract(text_field, r)
                if cap is None:
                    for alt_field in ["caption", "text", "sentence", "query"]:
                        cap = extract(alt_field, r)
                        if cap is not None:
                            break
                if vid is None or vpath is None or cap is None:
                    raise KeyError("Missing required columns in dataset row")
                records.append(DatasetRecord(str(vid), str(vpath), str(cap)))
            except Exception as e:
                LOGGER.warning("Skipping row due to error: %s", e)
        LOGGER.info("Loaded %d records from %s %s split", len(records), dataset_name, split)
        return records

class LocalDatasetLoader:
    
    @staticmethod
    def load(
        file_path: str,
        *,
        id_field: str = "video_id",
        video_field: str = "video",
        text_field: str = "caption",
    ) -> List[DatasetRecord]:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        ext = path.suffix.lower()
        records = []

        try:
            if ext in {".json", ".jsonl"}:
                records = LocalDatasetLoader._load_json(path, id_field, video_field, text_field)
            else:
                records = LocalDatasetLoader._load_csv(path, ext, id_field, video_field, text_field)
        except Exception as exc:
            LOGGER.error("Failed to parse dataset file %s: %s", path, exc)
            raise

        LOGGER.info("Loaded %d records from %s", len(records), path)
        return records
    
    @staticmethod
    def _load_json(
        path: Path, 
        id_field: str, 
        video_field: str, 
        text_field: str
    ) -> List[DatasetRecord]:
        with path.open() as fp:

            first_char = fp.read(1)
            fp.seek(0)
            
            if first_char == "[":
                data = json.load(fp)
                rows = data if isinstance(data, list) else []
            else:
                rows = [json.loads(line) for line in fp if line.strip()]

        records = []
        for r in rows:
            vid = r.get(id_field)
            vpath = r.get(video_field)
            cap = r.get(text_field)
            if vid and vpath and cap:
                records.append(DatasetRecord(str(vid), str(vpath), str(cap)))
                
        return records
    
    @staticmethod
    def _load_csv(
        path: Path, 
        ext: str, 
        id_field: str, 
        video_field: str, 
        text_field: str
    ) -> List[DatasetRecord]:
        import csv
        
        delimiter = "\t" if ext == ".tsv" else ","
        records = []
        
        with path.open(newline="") as fp:
            reader = csv.DictReader(fp, delimiter=delimiter)
            for r in reader:
                vid = r.get(id_field)
                vpath = r.get(video_field)
                cap = r.get(text_field)
                if vid and vpath and cap:
                    records.append(DatasetRecord(str(vid), str(vpath), str(cap)))
                    
        return records

class DatasetFactory:
    
    @staticmethod
    def load_dataset(
        dataset_name: Optional[str] = None,
        dataset_file: Optional[str] = None,
        split: str = "test",
        config: Optional[str] = None,
        *,
        id_field: str = "video_id",
        video_field: str = "video", 
        text_field: str = "caption",
        video_dir: Optional[Path] = None,
    ) -> Tuple[List[DatasetRecord], Optional[Path]]:

        if dataset_file:
            LOGGER.info("Loading local dataset from %s", dataset_file)
            records = LocalDatasetLoader.load(
                dataset_file,
                id_field=id_field,
                video_field=video_field,
                text_field=text_field,
            )
            resolved_video_dir = None
        else:
            if dataset_name and "msr-vtt" in dataset_name.lower():
                LOGGER.info("Loading MSR-VTT dataset %s split=%s", dataset_name, split)
                records = HuggingFaceDatasetLoader.load_msrvtt(split, config)
                

                if video_dir is None:
                    msrvtt_loader = MSRVTTLoader()
                    resolved_video_dir = msrvtt_loader.ensure_videos(
                        Path(tempfile.gettempdir()) / "msrvtt_videos"
                    )
                else:
                    resolved_video_dir = video_dir
            elif os.path.exists(video_dir):
                resolved_video_dir = Path(video_dir)
                records = [
                    DatasetRecord(
                        video_id=file.split(".")[0],
                        video_path=file,
                        caption="",
                    )
                    for file in os.listdir(resolved_video_dir)
                ]
                LOGGER.info("Loading local dataset from %s", resolved_video_dir)
            else:
                LOGGER.info("Loading HF dataset %s split=%s", dataset_name, split)
                records = HuggingFaceDatasetLoader.load_generic(
                    dataset_name,
                    split,
                    config,
                    id_field=id_field,
                    video_field=video_field,
                    text_field=text_field,
                )
                resolved_video_dir = video_dir
        

        if resolved_video_dir is not None:
            records = DatasetFactory._resolve_video_paths(records, resolved_video_dir)
            
        return records, resolved_video_dir
    
    @staticmethod
    def _resolve_video_paths(records: List[DatasetRecord], video_dir: Path) -> List[DatasetRecord]:
        resolved_records = []
        for record in records:
            vid, vpath, cap = record
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
                    else:
                        LOGGER.warning("Video file not found: %s", basename)
                resolved_path = candidate
                
            resolved_records.append(DatasetRecord(vid, str(resolved_path), cap))
            
        return resolved_records