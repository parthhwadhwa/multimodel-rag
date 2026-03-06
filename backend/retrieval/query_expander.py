"""
Query Expander — implements HyDE and MultiQuery retrieval strategies.
Uses the LLM to generate alternative queries for improved recall.
"""
import json
from typing import List
from collections import defaultdict

import requests

from backend.datatypes import RetrievalResult
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.config import CONFIG
from backend.logger import logger


class QueryExpander:
    """Expand queries using HyDE and MultiQuery strategies."""

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever
        self.ollama_url = CONFIG.ollama.base_url
        self.model = CONFIG.ollama.model

    def hyde_retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        HyDE: Generate a hypothetical answer, embed it, and search.
        This finds documents similar to what a good answer would look like.
        """
        top_k = top_k or CONFIG.retrieval.top_k

        hypothetical = self._generate_hypothetical_answer(query)
        if not hypothetical:
            logger.warning("HyDE: failed to generate hypothetical, falling back to direct")
            return self.retriever.retrieve(query, top_k=top_k)

        logger.info(f"HyDE: generated hypothetical answer ({len(hypothetical)} chars)")

        # Retrieve using the hypothetical answer as the search query
        results = self.retriever.retrieve(hypothetical, top_k=top_k)
        for r in results:
            r.retrieval_method = f"hyde+{r.retrieval_method}"

        return results

    def multi_query_retrieve(
        self, query: str, top_k: int = None
    ) -> List[RetrievalResult]:
        """
        MultiQuery: Generate alternative query formulations and merge results.
        """
        top_k = top_k or CONFIG.retrieval.top_k
        num_queries = CONFIG.retrieval.multi_query_count

        alt_queries = self._generate_alternative_queries(query, num_queries)
        if not alt_queries:
            logger.warning("MultiQuery: failed to generate alternatives, using original")
            return self.retriever.retrieve(query, top_k=top_k)

        logger.info(f"MultiQuery: generated {len(alt_queries)} alternative queries")

        # Retrieve for each query
        all_results = [self.retriever.retrieve(query, top_k=top_k)]
        for aq in alt_queries:
            results = self.retriever.retrieve(aq, top_k=top_k)
            all_results.append(results)

        # Fuse all result lists using RRF
        fused = self._fuse_results(all_results, top_k)
        for r in fused:
            r.retrieval_method = f"multiquery+{r.retrieval_method}"

        return fused

    def _generate_hypothetical_answer(self, query: str) -> str:
        """Use LLM to generate a hypothetical document passage that answers the query."""
        prompt = (
            f"You are a medical knowledge assistant. Write a brief, factual passage "
            f"that would answer the following question about medications. "
            f"Write in the style of a drug information document.\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )
        return self._call_ollama(prompt, max_tokens=200)

    def _generate_alternative_queries(self, query: str, count: int = 3) -> List[str]:
        """Use LLM to generate alternative query formulations."""
        prompt = (
            f"You are a helpful assistant that generates alternative search queries. "
            f"Given the following question about medications, generate {count} "
            f"alternative ways to ask the same question. Focus on different aspects "
            f"and use different terminology.\n\n"
            f"Original question: {query}\n\n"
            f"Return ONLY the alternative queries, one per line, numbered 1-{count}:"
        )
        response = self._call_ollama(prompt, max_tokens=300)
        if not response:
            return []

        # Parse numbered lines
        lines = response.strip().split("\n")
        queries = []
        for line in lines:
            line = line.strip()
            # Remove numbering (1. or 1) or -)
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned and len(cleaned) > 10:
                queries.append(cleaned)

        return queries[:count]

    def _call_ollama(self, prompt: str, max_tokens: int = 200) -> str:
        """Call Ollama for text generation."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_tokens,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""

    def _fuse_results(
        self, result_lists: List[List[RetrievalResult]], top_k: int
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion across multiple result lists."""
        k = CONFIG.retrieval.rrf_k
        scores = defaultdict(float)
        chunk_map = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                cid = result.chunk.id
                scores[cid] += 1.0 / (k + rank + 1)
                if cid not in chunk_map:
                    chunk_map[cid] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            RetrievalResult(
                chunk=chunk_map[cid].chunk,
                score=scores[cid],
                retrieval_method=chunk_map[cid].retrieval_method,
            )
            for cid in sorted_ids[:top_k]
        ]
