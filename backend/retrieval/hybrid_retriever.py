"""
Hybrid Retriever — combines dense vector + BM25 lexical search
with Reciprocal Rank Fusion (RRF).
"""
from typing import List, Dict
from collections import defaultdict

from backend.datatypes import RetrievalResult
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.config import CONFIG
from backend.logger import logger


class HybridRetriever:
    """Hybrid retrieval combining dense + BM25 with RRF fusion."""

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = CONFIG.retrieval.rrf_k

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Retrieve using both methods and fuse with RRF."""
        top_k = top_k or CONFIG.retrieval.top_k
        fetch_k = top_k * 2  # Fetch more for fusion

        # Parallel retrieval
        dense_results = self.dense.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)

        logger.info(
            f"Hybrid retrieval: {len(dense_results)} dense, {len(bm25_results)} BM25"
        )

        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            [dense_results, bm25_results], top_k=top_k
        )

        return fused

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.
        RRF score = sum( 1 / (k + rank_i) ) for each result list.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievalResult] = {}
        methods_map: Dict[str, List[str]] = defaultdict(list)

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                chunk_id = result.chunk.id
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)
                methods_map[chunk_id].append(result.retrieval_method)

                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for chunk_id in sorted_ids[:top_k]:
            result = chunk_map[chunk_id]
            methods = list(set(methods_map[chunk_id]))
            fused_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=rrf_scores[chunk_id],
                    retrieval_method="+".join(methods),
                )
            )

        logger.info(f"RRF fusion produced {len(fused_results)} results")
        return fused_results
