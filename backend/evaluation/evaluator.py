"""
Evaluation Pipeline — retrieval quality metrics using LangSmith.
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.utils.datatypes import RetrievalResult
from backend.utils.config import CONFIG
from backend.utils.logger import logger


class RAGEvaluator:
    """Evaluate RAG retrieval and generation quality."""

    def __init__(self):
        self.results_log: List[Dict[str, Any]] = []

        # Initialize LangSmith tracing if configured
        if CONFIG.langsmith.tracing_enabled and CONFIG.langsmith.api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = CONFIG.langsmith.api_key
            os.environ["LANGCHAIN_PROJECT"] = CONFIG.langsmith.project_name
            logger.info(f"LangSmith tracing enabled for project: {CONFIG.langsmith.project_name}")
        else:
            logger.info("LangSmith tracing not configured, using local evaluation only")

    def precision_at_k(
        self, retrieved: List[RetrievalResult], relevant_ids: List[str], k: int = 5
    ) -> float:
        """Compute precision@k: fraction of top-k that are relevant."""
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        relevant_count = sum(1 for r in top_k if r.chunk.id in relevant_ids)
        return relevant_count / len(top_k)

    def recall_at_k(
        self, retrieved: List[RetrievalResult], relevant_ids: List[str], k: int = 5
    ) -> float:
        """Compute recall@k: fraction of relevant docs found in top-k."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved[:k]
        relevant_found = sum(1 for r in top_k if r.chunk.id in relevant_ids)
        return relevant_found / len(relevant_ids)

    def mean_reciprocal_rank(
        self, retrieved: List[RetrievalResult], relevant_ids: List[str]
    ) -> float:
        """Compute MRR: 1/rank of the first relevant result."""
        for i, result in enumerate(retrieved):
            if result.chunk.id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def context_relevance(
        self, query: str, retrieved: List[RetrievalResult]
    ) -> float:
        """
        Estimate context relevance using retrieval scores.
        Higher average score → more relevant context.
        """
        if not retrieved:
            return 0.0
        scores = [r.score for r in retrieved]
        return sum(scores) / len(scores)

    def evaluate_query(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        relevant_ids: Optional[List[str]] = None,
        response: str = "",
    ) -> Dict[str, Any]:
        """Run full evaluation for a single query."""
        relevant_ids = relevant_ids or []

        metrics = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "num_retrieved": len(retrieved),
            "context_relevance": round(self.context_relevance(query, retrieved), 4),
            "retrieval_methods": list(set(r.retrieval_method for r in retrieved)),
        }

        if relevant_ids:
            for k in [1, 3, 5]:
                metrics[f"precision@{k}"] = round(
                    self.precision_at_k(retrieved, relevant_ids, k), 4
                )
                metrics[f"recall@{k}"] = round(
                    self.recall_at_k(retrieved, relevant_ids, k), 4
                )
            metrics["mrr"] = round(
                self.mean_reciprocal_rank(retrieved, relevant_ids), 4
            )

        if response:
            metrics["response_length"] = len(response)
            metrics["response_word_count"] = len(response.split())

        self.results_log.append(metrics)
        return metrics

    def evaluate_batch(
        self, test_cases: List[Dict[str, Any]], retriever, llm_client=None
    ) -> Dict[str, Any]:
        """
        Run evaluation on a batch of test cases.
        Each test case: {"query": str, "relevant_ids": [str], ...}
        """
        all_metrics = []

        for case in test_cases:
            query = case["query"]
            relevant_ids = case.get("relevant_ids", [])

            results = retriever.retrieve(query)
            response = ""
            if llm_client:
                response = llm_client.generate(query, results)

            metrics = self.evaluate_query(query, results, relevant_ids, response)
            all_metrics.append(metrics)

        # Aggregate
        summary = {
            "total_queries": len(all_metrics),
            "avg_context_relevance": round(
                sum(m["context_relevance"] for m in all_metrics) / len(all_metrics), 4
            ) if all_metrics else 0,
            "avg_num_retrieved": round(
                sum(m["num_retrieved"] for m in all_metrics) / len(all_metrics), 1
            ) if all_metrics else 0,
        }

        if all(m.get("mrr") is not None for m in all_metrics):
            summary["avg_mrr"] = round(
                sum(m["mrr"] for m in all_metrics) / len(all_metrics), 4
            )
            for k in [1, 3, 5]:
                key = f"precision@{k}"
                if all(key in m for m in all_metrics):
                    summary[f"avg_{key}"] = round(
                        sum(m[key] for m in all_metrics) / len(all_metrics), 4
                    )

        summary["individual_results"] = all_metrics
        return summary

    def get_results(self) -> List[Dict[str, Any]]:
        """Return all logged evaluation results."""
        return self.results_log
