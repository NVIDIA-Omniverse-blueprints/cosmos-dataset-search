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
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import DatasetFactory
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
LOGGER = logging.getLogger("accuracy-only")

DEFAULT_PROFILE = "local"
RELEVANCE_CHOICES = ["instance", "class"]


class AccuracyConfig:
    def __init__(self, args):
        self.collection_id = args.collection_id
        self.collection_name = args.collection_name
        
        # Dataset for queries
        self.dataset_name = args.dataset
        self.dataset_file = args.dataset_file
        self.split = args.split
        self.config = args.config
        self.video_dir = Path(args.video_dir) if args.video_dir else None
        
        # Field mapping
        self.id_field = args.id_field
        self.video_field = args.video_field
        self.text_field = args.text_field
        
        # Evaluation parameters
        self.top_k = args.top_k
        self.limit = args.limit if args.limit > 0 else None
        self.relevance_level = args.relevance_level
        
        # Client config
        self.profile = args.profile
        
        # Output
        self.output_dir = Path(args.output_dir) if args.output_dir else None
    
    def validate(self) -> None:
        if not self.collection_id and not self.collection_name:
            raise ValueError("Either --collection-id or --collection-name must be specified")
        
        if self.collection_id and self.collection_name:
            raise ValueError("Cannot specify both --collection-id and --collection-name")
        
        if not self.dataset_name and not self.dataset_file:
            raise ValueError("Either --dataset or --dataset-file must be specified")
        
        if self.dataset_name and self.dataset_file:
            raise ValueError("Cannot specify both --dataset and --dataset-file")
        
        if self.relevance_level not in RELEVANCE_CHOICES:
            raise ValueError(f"Invalid relevance level: {self.relevance_level}")
        
        if self.video_dir and not self.video_dir.exists():
            raise ValueError(f"Video directory does not exist: {self.video_dir}")


class AccuracyOnlyEvaluator:
    def __init__(self, config: AccuracyConfig):
        self.config = config
        
        self.client = Client()
        self.evaluator = create_evaluator(config.relevance_level)
        
        (self.collection_manager, 
         self.ingestion_manager, 
         self.search_manager, 
         self.validator) = create_collection_lifecycle_manager(self.client, config.profile)
    
    def run_accuracy_evaluation(self) -> dict:
        """Run accuracy evaluation on existing collection."""
        LOGGER.info("Starting accuracy-only evaluation")
        LOGGER.info("Configuration: dataset=%s, top_k=%d", 
                   self.config.dataset_name or self.config.dataset_file,
                   self.config.top_k)
        
        try:
            # Find collection
            collection_id = self._resolve_collection_id()
            LOGGER.info("Using collection ID: %s", collection_id)
            
            # Load dataset for queries
            LOGGER.info("Loading dataset for queries...")
            records, video_dir = self._load_dataset()
            LOGGER.info("Loaded %d records", len(records))
            
            # Limit queries if specified
            if self.config.limit:
                records = records[:self.config.limit]
                LOGGER.info("Limited to %d queries", len(records))
            
            # Run evaluation
            LOGGER.info("Running evaluation...")
            queries = [(record.caption, record.video_id) for record in records]
            search_function = self.search_manager.create_search_function(collection_id)
            
            metrics = self.evaluator.evaluate_queries(
                queries, search_function, self.config.top_k
            )
            
            # Generate results
            results = self._generate_results(metrics, records)
            
            LOGGER.info("Accuracy evaluation completed successfully!")
            return results
            
        except Exception as e:
            LOGGER.error("Evaluation failed: %s", e)
            raise
    
    def _resolve_collection_id(self) -> str:
        """Resolve collection ID from name or use provided ID."""
        if self.config.collection_id:
            return self.config.collection_id
        
        # Find collection by name
        try:
            response = self.client.collections.list(profile=self.config.profile)
            collections = response.get("collections", [])
            
            for collection in collections:
                if collection.get("name") == self.config.collection_name:
                    collection_id = collection.get("id")
                    LOGGER.info("Found collection '%s' with ID: %s", 
                               self.config.collection_name, collection_id)
                    return collection_id
            
            raise ValueError(f"Collection '{self.config.collection_name}' not found. "
                           f"Available collections: {[c.get('name') for c in collections]}")
            
        except Exception as e:
            LOGGER.error("Failed to resolve collection: %s", e)
            raise
    
    def _load_dataset(self) -> Tuple[List, Optional[Path]]:
        """Load dataset records for queries."""
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
    
    def _generate_results(self, metrics, query_records) -> dict:
        """Generate and save results."""
        dataset_slug = self._get_dataset_slug()
        
        results = {
            "dataset": dataset_slug,
            "collection_id": self.config.collection_id or "resolved",
            "collection_name": self.config.collection_name or "unknown",
            "queries": len(query_records),
            "metrics": metrics.to_dict(),
            "config": {
                "top_k": self.config.top_k,
                "relevance_level": self.config.relevance_level,
                "limit": self.config.limit
            }
        }
        
        # Print summary
        LOGGER.info("Printing metrics summary...")
        metrics.print_summary()
        
        # Save results
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.config.output_dir / f"{dataset_slug}_accuracy_only_results.json"
            metrics.save_json(output_file)
            LOGGER.info("Results saved to: %s", output_file)
        
        return results
    
    def _get_dataset_slug(self) -> str:
        """Generate dataset slug for naming."""
        if self.config.dataset_file:
            return Path(self.config.dataset_file).stem
        else:
            return self.config.dataset_name.split("/")[-1]


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run accuracy evaluation on existing CVDS collection"
    )
    
    # Collection identification
    collection_group = parser.add_mutually_exclusive_group(required=True)
    collection_group.add_argument(
        "--collection-id", 
        help="Collection ID to evaluate"
    )
    collection_group.add_argument(
        "--collection-name", 
        help="Collection name to evaluate"
    )
    
    # Dataset for queries
    dataset_group = parser.add_mutually_exclusive_group(required=True)
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
    
    # Evaluation parameters
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=10, 
        help="Top-K for retrieval evaluation (default: 10)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0, 
        help="Limit number of queries to evaluate (0 = no limit)"
    )
    parser.add_argument(
        "--relevance-level", 
        choices=RELEVANCE_CHOICES, 
        default="instance",
        help="Relevance definition: 'instance' for exact video match, 'class' for same caption"
    )
    
    # Client options
    parser.add_argument(
        "--profile", 
        default=DEFAULT_PROFILE, 
        help=f"Configuration profile (default: {DEFAULT_PROFILE})"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        help="Output directory for results (default: no output file)"
    )
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        config = AccuracyConfig(args)
        config.validate()
        
        evaluator = AccuracyOnlyEvaluator(config)
        results = evaluator.run_accuracy_evaluation()
        
        print(f"\nAccuracy evaluation completed successfully!")
        print(f"Collection: {config.collection_name or config.collection_id}")
        print(f"Dataset: {config.dataset_name or config.dataset_file}")
        print(f"Queries: {results['queries']}")
        print(f"Recall@1: {results['metrics']['recall_at_1']:.3f}")
        print(f"Recall@{config.top_k}: {results['metrics']['recall_at_k']:.3f}")
        print(f"MRR: {results['metrics']['mrr']:.3f}")
        
    except Exception as e:
        LOGGER.error("Failed to run accuracy evaluation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
