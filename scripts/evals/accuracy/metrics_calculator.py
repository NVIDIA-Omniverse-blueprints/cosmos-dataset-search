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

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

LOGGER = logging.getLogger("metrics-calculator")

@dataclass
class RetrievalMetrics:
    recall_at_1: float
    precision_at_1: float
    recall_at_k: float
    precision_at_k: float
    f1_at_k: float
    mrr: float
    ndcg: float
    total_queries: int
    top_k: int
    relevance_level: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save_json(self, output_path: Path) -> None:
        with output_path.open("w") as fp:
            json.dump(self.to_dict(), fp, indent=2)
        LOGGER.info("Saved metrics to %s", output_path)
    
    def print_summary(self) -> None:
        print(f"\n=== Retrieval Accuracy Metrics ===")
        print(f"Total Queries:    {self.total_queries}")
        print(f"Top-K:           {self.top_k}")
        print(f"Relevance Level: {self.relevance_level}")
        print(f"Recall@1:        {self.recall_at_1:.4f}")
        print(f"Precision@1:     {self.precision_at_1:.4f}")
        print(f"Recall@{self.top_k}:      {self.recall_at_k:.4f}")
        print(f"Precision@{self.top_k}:   {self.precision_at_k:.4f}")
        print(f"F1@{self.top_k}:         {self.f1_at_k:.4f}")
        print(f"MRR:             {self.mrr:.4f}")
        print(f"NDCG@{self.top_k}:       {self.ndcg:.4f}")

class RelevanceManager:
    
    def __init__(self, relevance_level: str = "instance"):
        if relevance_level not in ["instance", "class"]:
            raise ValueError(f"Invalid relevance_level: {relevance_level}. Must be 'instance' or 'class'")
        
        self.relevance_level = relevance_level
        self._label_to_ids: Dict[str, Set[str]] = defaultdict(set)
    
    def build_label_mapping(self, queries: List[Tuple[str, str]]) -> None:
        for text_query, video_id in queries:
            self._label_to_ids[text_query].add(video_id)
        
        LOGGER.info("Built label mapping for %d unique queries", len(self._label_to_ids))
    
    def get_relevant_ids(self, text_query: str, target_video_id: str) -> Set[str]:
        if self.relevance_level == "instance":
            return {str(target_video_id)}
        else:
            return {str(vid) for vid in self._label_to_ids.get(text_query, set())}

class MetricsCalculator:
    
    def __init__(self, relevance_manager: RelevanceManager):
        self.relevance_manager = relevance_manager
    
    def evaluate_batch(
        self,
        results: List[Tuple[str, str, List[Dict]]],
        top_k: int = 10
    ) -> RetrievalMetrics:
        total_queries = len(results)
        if total_queries == 0:
            return RetrievalMetrics(0, 0, 0, 0, 0, 0, 0, 0, top_k, self.relevance_manager.relevance_level)
        hits_at_1 = 0
        hits_at_k = 0
        sum_precision_at_k = 0.0
        sum_f1_at_k = 0.0
        sum_rr = 0.0
        sum_ndcg = 0.0
        
        for text_query, video_id, retrieved_docs in results:
            relevant_ids = self.relevance_manager.get_relevant_ids(text_query, video_id)
            found_rank = None
            hits_in_topk = 0
            
            for idx, doc in enumerate(retrieved_docs[:top_k], start=1):
                doc_video_id = self._extract_video_id(doc)
                if doc_video_id in relevant_ids:
                    if found_rank is None:
                        found_rank = idx
                    hits_in_topk += 1
            if found_rank is not None:
                if found_rank == 1:
                    hits_at_1 += 1
                
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
                dcg = 1.0 / math.log2(found_rank + 1.0)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), top_k)))
                sum_ndcg += dcg / idcg if idcg > 0 else 0.0
            else:
                sum_precision_at_k += 0.0
                sum_f1_at_k += 0.0
                sum_rr += 0.0
                sum_ndcg += 0.0
        recall_at_1 = hits_at_1 / total_queries
        precision_at_1 = recall_at_1
        recall_at_k = hits_at_k / total_queries
        precision_at_k = sum_precision_at_k / total_queries
        f1_at_k = sum_f1_at_k / total_queries
        mrr = sum_rr / total_queries
        ndcg = sum_ndcg / total_queries
        
        return RetrievalMetrics(
            recall_at_1=recall_at_1,
            precision_at_1=precision_at_1,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            f1_at_k=f1_at_k,
            mrr=mrr,
            ndcg=ndcg,
            total_queries=total_queries,
            top_k=top_k,
            relevance_level=self.relevance_manager.relevance_level
        )
    
    @staticmethod
    def _extract_video_id(document: Dict) -> str:
        metadata = document.get("metadata", {})
        video_id = metadata.get("video_id")
        return str(video_id) if video_id is not None else ""

class BatchEvaluator:
    
    def __init__(self, relevance_level: str = "instance"):
        self.relevance_manager = RelevanceManager(relevance_level)
        self.calculator = MetricsCalculator(self.relevance_manager)
    
    def evaluate_queries(
        self,
        queries: List[Tuple[str, str]],
        search_function,
        top_k: int = 10,
        *,
        progress_callback=None
    ) -> RetrievalMetrics:
        if self.relevance_manager.relevance_level == "class":
            self.relevance_manager.build_label_mapping(queries)
        results = []
        total_queries = len(queries)
        
        LOGGER.info("Evaluating %d queries with top_k=%d", total_queries, top_k)
        
        for i, (text_query, video_id) in enumerate(queries):
            retrieved_docs = search_function(text_query, top_k)
            results.append((text_query, video_id, retrieved_docs))
            if progress_callback:
                progress_callback(i + 1, total_queries)
            if (i + 1) % 100 == 0 or (i + 1) == total_queries:
                LOGGER.info("Processed %d/%d queries", i + 1, total_queries)
        metrics = self.calculator.evaluate_batch(results, top_k)
        
        LOGGER.info("Evaluation complete: Recall@1=%.3f, Recall@%d=%.3f, MRR=%.3f", 
                   metrics.recall_at_1, top_k, metrics.recall_at_k, metrics.mrr)
        
        return metrics

def create_evaluator(relevance_level: str = "instance") -> BatchEvaluator:
    return BatchEvaluator(relevance_level)

def evaluate_single_query(
    retrieved_docs: List[Dict],
    relevant_video_ids: Set[str],
    top_k: int = 10
) -> Dict[str, float]:
    found_rank = None
    hits_in_topk = 0
    
    for idx, doc in enumerate(retrieved_docs[:top_k], start=1):
        metadata = doc.get("metadata", {})
        doc_video_id = str(metadata.get("video_id", ""))
        
        if doc_video_id in relevant_video_ids:
            if found_rank is None:
                found_rank = idx
            hits_in_topk += 1
    if found_rank is not None:
        recall_at_1 = 1.0 if found_rank == 1 else 0.0
        recall_at_k = 1.0
        precision_at_k = hits_in_topk / float(top_k)
        recall_k = hits_in_topk / float(len(relevant_video_ids))
        
        if precision_at_k + recall_k > 0:
            f1_at_k = 2 * precision_at_k * recall_k / (precision_at_k + recall_k)
        else:
            f1_at_k = 0.0
            
        mrr = 1.0 / float(found_rank)
        dcg = 1.0 / math.log2(found_rank + 1.0)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_video_ids), top_k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
    else:
        recall_at_1 = 0.0
        recall_at_k = 0.0
        precision_at_k = 0.0
        f1_at_k = 0.0
        mrr = 0.0
        ndcg = 0.0
    
    return {
        "recall_at_1": recall_at_1,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "f1_at_k": f1_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
        "found_rank": found_rank,
        "hits_in_topk": hits_in_topk
    }