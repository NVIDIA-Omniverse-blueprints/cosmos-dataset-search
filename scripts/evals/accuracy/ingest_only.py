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

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import DatasetFactory
from video_manager import create_s3_manager, VideoIngestionPipeline
from collection_manager import create_collection_lifecycle_manager

try:
    from visual_search.client import Client
except ImportError as e:
    print(f"ERROR: Failed to import CVDS client: {e}", file=sys.stderr)
    print("Make sure you're running from the repo root directory.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger("ingest-collection")

DEFAULT_PIPELINE = "cosmos_video_search_milvus"
DEFAULT_BUCKET = "cosmos-test-bucket"
DEFAULT_S3_ENDPOINT = "http://localstack:4566"
DEFAULT_PROFILE = "default"
DEFAULT_NUM_WORKERS = 4

class IngestionConfig:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.dataset_file = args.dataset_file
        self.split = args.split
        self.config = args.config
        self.video_dir = Path(args.video_dir) if args.video_dir else None
        
        self.id_field = args.id_field
        self.video_field = args.video_field
        self.text_field = args.text_field
        
        self.limit = args.limit if args.limit > 0 else None
        self.collection_name = args.collection_name
        
        self.pipeline_id = args.pipeline_id
        self.profile = args.profile
        self.num_workers = args.num_workers

        self.bucket = args.bucket
        self.s3_endpoint = args.s3_endpoint
        
        self.tmpdir = Path(args.tmpdir) if args.tmpdir else None
    
    def validate(self) -> None:
        if not self.dataset_name and not self.dataset_file and not self.video_dir:
            raise ValueError("Either --dataset or --dataset-file or --video-dir must be specified")
        
        if self.dataset_name and self.dataset_file:
            raise ValueError("Cannot specify both --dataset and --dataset-file")
        
        if self.video_dir and not self.video_dir.exists():
            raise ValueError(f"Video directory does not exist: {self.video_dir}")


class CollectionIngestor:
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        self.client = Client()
        self.s3_manager = create_s3_manager(config.bucket, config.s3_endpoint)
        self.video_pipeline = VideoIngestionPipeline(self.s3_manager)
        
        (self.collection_manager, 
         self.ingestion_manager, 
         self.search_manager, 
         self.validator) = create_collection_lifecycle_manager(self.client, config.profile)
        
        self.work_dir = config.tmpdir or Path(tempfile.mkdtemp(prefix="cvds_ingest_"))
    
    def ingest_collection(self) -> str:
        """Create collection and ingest videos, return collection ID."""
        LOGGER.info("Starting CVDS collection ingestion")
        LOGGER.info("Configuration: dataset=%s, pipeline=%s", 
                   self.config.dataset_name or self.config.dataset_file,
                   self.config.pipeline_id)
        
        try:
            # Validate pipeline
            if not self.validator.validate_pipeline(self.config.pipeline_id):
                available = self.validator.list_available_pipelines()
                raise ValueError(f"Pipeline '{self.config.pipeline_id}' not available. "
                               f"Available pipelines: {available}")
            
            # Load dataset
            LOGGER.info("Loading dataset...")
            records, video_dir = self._load_dataset()
            LOGGER.info("Loaded %d records", len(records))
            
            # Prepare videos for ingestion
            LOGGER.info("Preparing videos for ingestion...")
            dataset_slug = self._get_dataset_slug()
            s3_path, metadata_json, query_records = self.video_pipeline.prepare_for_ingestion(
                records, dataset_slug, self.work_dir, limit=self.config.limit
            )
            
            # Create collection and ingest
            LOGGER.info("Creating collection and ingesting videos...")
            collection_id = self._create_and_ingest_collection(s3_path, metadata_json, dataset_slug)
            
            LOGGER.info("Collection ingestion completed successfully!")
            LOGGER.info("Collection ID: %s", collection_id)
            LOGGER.info("Collection Name: %s", self.config.collection_name or dataset_slug)
            LOGGER.info("Total videos ingested: %d", len(query_records))
            
            return collection_id
            
        except Exception as e:
            LOGGER.error("Ingestion failed: %s", e)
            raise
        finally:
            LOGGER.info("Cleaning up temporary files...")
            # Note: We don't clean up the collection, only temporary files
            if self.work_dir.exists() and not self.config.tmpdir:
                import shutil
                shutil.rmtree(self.work_dir)
    
    def _load_dataset(self) -> Tuple[List, Optional[Path]]:
        """Load dataset records."""
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
        """Generate dataset slug for collection naming."""
        if self.config.collection_name:
            return self.config.collection_name
        elif self.config.dataset_name:
            return self.config.dataset_name.split('/')[-1]
        else:
            return Path(self.config.dataset_file).stem
    
    def _create_and_ingest_collection(self, s3_path: str, metadata_json: Path, dataset_slug: str) -> str:
        """Create collection and ingest videos."""
        collection_id = self.collection_manager.create_collection(
            self.config.pipeline_id,
            dataset_slug
        )
        
        stats = self.ingestion_manager.ingest_videos(
            collection_id, s3_path, metadata_json, num_workers=self.config.num_workers
        )
        LOGGER.info("Ingestion stats: %s", stats)
        
        self.ingestion_manager.flush_collection(collection_id)
        LOGGER.info("Collection created and ingested")
        
        return collection_id


def create_parser():
    parser = argparse.ArgumentParser(
        description="Ingest a dataset into a CVDS collection without running evaluation"
    )
    
    # Dataset options
    dataset_group = parser.add_mutually_exclusive_group(required=False)
    dataset_group.add_argument(
        "--dataset", 
        help="HuggingFace dataset name (e.g., 'friedrichor/MSR-VTT')"
    )
    dataset_group.add_argument(
        "--dataset-file", 
        help="Path to local dataset file (JSON/CSV/Parquet)"
    )
    
    parser.add_argument(
        "--split", 
        default="test", 
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--config", 
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--video-dir", 
        help="Directory containing video files (if not using URLs)"
    )
    
    # Field mapping for custom datasets
    parser.add_argument(
        "--id-field", 
        default="video_id", 
        help="Field name for video ID (default: video_id)"
    )
    parser.add_argument(
        "--video-field", 
        default="video", 
        help="Field name for video path/URL (default: video)"
    )
    parser.add_argument(
        "--text-field", 
        default="caption", 
        help="Field name for text description (default: caption)"
    )
    
    # Collection options
    parser.add_argument(
        "--collection-name", 
        help="Custom name for the collection (default: derived from dataset)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0, 
        help="Limit number of videos to ingest (0 = no limit)"
    )
    
    # Pipeline options
    parser.add_argument(
        "--pipeline-id", 
        default=DEFAULT_PIPELINE, 
        help=f"Pipeline to use (default: {DEFAULT_PIPELINE})"
    )
    parser.add_argument(
        "--profile", 
        default=DEFAULT_PROFILE, 
        help=f"Configuration profile (default: {DEFAULT_PROFILE})"
    )

    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=DEFAULT_NUM_WORKERS, 
        help=f"Number of parallel ingest workers (default: {DEFAULT_NUM_WORKERS})"
    )

    # Storage options
    parser.add_argument(
        "--bucket", 
        default=DEFAULT_BUCKET, 
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})"
    )
    parser.add_argument(
        "--s3-endpoint", 
        default=DEFAULT_S3_ENDPOINT, 
        help=f"S3 endpoint URL (default: {DEFAULT_S3_ENDPOINT})"
    )
    
    # Working directory
    parser.add_argument(
        "--tmpdir", 
        help="Temporary directory for processing (default: auto-generated)"
    )
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    #try:
    config = IngestionConfig(args)
    config.validate()
    
    ingestor = CollectionIngestor(config)
    collection_id = ingestor.ingest_collection()
    
    print(f"\nCollection ingestion completed successfully!")
    print(f"Collection ID: {collection_id}")
    print(f"Collection Name: {config.collection_name or ingestor._get_dataset_slug()}")
    print(f"Pipeline: {config.pipeline_id}")
    print(f"\nTest in UI: http://localhost:8080/cosmos-dataset-search")
        
    # except Exception as e:
    #     LOGGER.error("Failed to ingest collection: %s", e)
    #     sys.exit(1)


if __name__ == "__main__":
    main()
