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
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import DatasetFactory
from video_manager import create_s3_manager, VideoIngestionPipeline
from metrics_calculator import create_evaluator
from collection_manager import create_collection_lifecycle_manager

try:
    from visual_search.client import Client
except ImportError as e:
    print(f"ERROR: Failed to import CVDS client: {e}", file=sys.stderr)
    print("Make sure you're running from the repo root directory.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger("evaluate-accuracy")

DEFAULT_PIPELINE = "cosmos_video_search_milvus"
DEFAULT_BUCKET = "cosmos-test-bucket"
DEFAULT_S3_ENDPOINT = "http://localstack:4566"
DEFAULT_PROFILE = "local"
RELEVANCE_CHOICES = ["instance", "class"]

class EvaluationConfig:
    
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.dataset_file = args.dataset_file
        self.split = args.split
        self.config = args.config
        self.video_dir = Path(args.video_dir) if args.video_dir else None
        

        self.id_field = args.id_field
        self.video_field = args.video_field
        self.text_field = args.text_field
        

        self.top_k = args.top_k
        self.limit = args.limit if args.limit > 0 else None
        self.relevance_level = args.relevance_level
        

        self.pipeline_id = args.pipeline_id
        self.profile = args.profile
        self.bucket = args.bucket
        self.s3_endpoint = args.s3_endpoint
        

        self.tmpdir = Path(args.tmpdir) if args.tmpdir else None
        self.output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else None
    
    def validate(self) -> None:
        if not self.dataset_name and not self.dataset_file:
            raise ValueError("Either --dataset or --dataset-file must be specified")
        
        if self.dataset_name and self.dataset_file:
            raise ValueError("Cannot specify both --dataset and --dataset-file")
        
        if self.relevance_level not in RELEVANCE_CHOICES:
            raise ValueError(f"Invalid relevance level: {self.relevance_level}")
        
        if self.video_dir and not self.video_dir.exists():
            raise ValueError(f"Video directory does not exist: {self.video_dir}")


class AccuracyEvaluator:
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        

        self.client = Client()
        self.s3_manager = create_s3_manager(config.bucket, config.s3_endpoint)
        self.video_pipeline = VideoIngestionPipeline(self.s3_manager)
        self.evaluator = create_evaluator(config.relevance_level)
        

        (self.collection_manager, 
         self.ingestion_manager, 
         self.search_manager, 
         self.validator) = create_collection_lifecycle_manager(self.client, config.profile)
        

        self.work_dir = config.tmpdir or Path(tempfile.mkdtemp(prefix="cvds_accuracy_"))
    
    def run_evaluation(self) -> dict:
        LOGGER.info("Starting CVDS accuracy evaluation")
        LOGGER.info("Configuration: dataset=%s, pipeline=%s, top_k=%d", 
                   self.config.dataset_name or self.config.dataset_file,
                   self.config.pipeline_id, self.config.top_k)
        try:
            if not self.validator.validate_pipeline(self.config.pipeline_id):
                available = self.validator.list_available_pipelines()
                raise ValueError(f"Pipeline '{self.config.pipeline_id}' not available. "
                               f"Available pipelines: {available}")
            LOGGER.info("Loading dataset...")
            records, video_dir = self._load_dataset()
            LOGGER.info("Loaded %d records", len(records))
            LOGGER.info("Preparing videos for ingestion...")
            dataset_slug = self._get_dataset_slug()
            s3_path, metadata_json, query_records = self.video_pipeline.prepare_for_ingestion(
                records, dataset_slug, self.work_dir, limit=self.config.limit
            )
            LOGGER.info("Creating collection and ingesting videos...")
            collection_id = self._create_and_ingest_collection(s3_path, metadata_json, dataset_slug)
            LOGGER.info("Running evaluation...")
            queries = [(record.caption, record.video_id) for record in query_records]
            search_function = self.search_manager.create_search_function(collection_id)            
            metrics = self.evaluator.evaluate_queries(
                queries, search_function, self.config.top_k
            )
            results = self._generate_results(metrics, query_records, dataset_slug)
            LOGGER.info("Evaluation completed successfully!")
            return results
        except Exception as e:
            LOGGER.error("Evaluation failed: %s", e)
            raise
        finally:
            LOGGER.info("Cleaning up...")
            self._cleanup()
    
    def _load_dataset(self) -> Tuple[List, Optional[Path]]:
        return DatasetFactory.load_dataset(
            dataset_name=self.config.dataset_name,
            dataset_file=self.config.dataset_file,
            split=self.config.split,
            config=self.config.config,
            id_field=self.config.id_field,
            video_field=self.config.video_field,
            text_field=self.config.text_field,
            video_dir=self.config.video_dir
        )
    
    def _get_dataset_slug(self) -> str:
        if self.config.dataset_file:
            return Path(self.config.dataset_file).stem
        else:
            return self.config.dataset_name.split("/")[-1]
    
    def _create_and_ingest_collection(self, s3_path: str, metadata_json: Path, dataset_slug: str) -> str:

        collection_id = self.collection_manager.create_collection(
            self.config.pipeline_id,
            dataset_slug
        )
        stats = self.ingestion_manager.ingest_videos(
            collection_id, s3_path, metadata_json
        )
        LOGGER.info("Ingestion stats: %s", stats)
        self.ingestion_manager.flush_collection(collection_id)
        LOGGER.info("Collection created and ingested")
        return collection_id
    
    def _generate_results(self, metrics, query_records, dataset_slug) -> dict:
        results = {
            "dataset": dataset_slug,
            "pipeline": self.config.pipeline_id,
            "queries": len(query_records),
            "metrics": metrics.to_dict(),
            "config": {
                "top_k": self.config.top_k,
                "relevance_level": self.config.relevance_level,
                "limit": self.config.limit
            }
        }
        
        LOGGER.info("Printing metrics summary...")
        metrics.print_summary()
        output_file = self.work_dir / f"{dataset_slug}_accuracy_results.json"
        metrics.save_json(output_file)
        LOGGER.info("Results saved to: %s", output_file)
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            output_copy = self.config.output_dir / f"{dataset_slug}_accuracy_results.json"
            shutil.copy2(output_file, output_copy)
            LOGGER.info("Results saved to: %s", output_copy)
        
        return results
    
    def _cleanup(self):
        try:

            pass
        except Exception as e:
            LOGGER.warning("Cleanup warning: %s", e)


def setup_environment():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CVDS text-to-video retrieval accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset", help="HuggingFace dataset name (e.g., friedrichor/MSR-VTT)"
    )
    dataset_group.add_argument(
        "--dataset-file", help="Path to local dataset file (JSON/CSV/JSONL)"
    )
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--config", help="Dataset config name (auto-detected if not provided)")
    parser.add_argument("--video-dir", help="Directory containing video files")
    parser.add_argument("--id-field", default="video_id", help="Field name for video IDs")
    parser.add_argument("--video-field", default="video", help="Field name for video paths")
    parser.add_argument("--text-field", default="caption", help="Field name for text queries")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries (0=all)")
    parser.add_argument(
        "--relevance-level", choices=RELEVANCE_CHOICES, default="instance",
        help="Relevance definition: 'instance' for exact video match, 'class' for same caption"
    )
    parser.add_argument("--pipeline-id", default=DEFAULT_PIPELINE, help="CVDS pipeline ID")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="CVDS client profile")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    parser.add_argument("--s3-endpoint", default=DEFAULT_S3_ENDPOINT, help="S3 endpoint URL")
    parser.add_argument("--tmpdir", help="Temporary directory for working files")
    parser.add_argument("--output-dir", help="Output directory for results")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    LOGGER.info("Starting evaluation...")
    try:
        setup_environment()
        config = EvaluationConfig(args)
        config.validate()
        with create_collection_lifecycle_manager(Client(), config.profile)[0]:
            evaluator = AccuracyEvaluator(config)
            results = evaluator.run_evaluation()
        print(f"\nEvaluation completed successfully!")
        print(f"Final Results: Recall@1={results['metrics']['recall_at_1']:.3f}, "
              f"Recall@{config.top_k}={results['metrics']['recall_at_k']:.3f}, "
              f"MRR={results['metrics']['mrr']:.3f}")
        return 0
        
    except KeyboardInterrupt:
        LOGGER.info("Evaluation interrupted by user")
        return 130
    except Exception as e:
        LOGGER.error("Evaluation failed: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
