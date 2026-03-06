"""
BM25 Retriever — lexical/keyword search using rank_bm25.
"""
from typing import List, Optional
import re

from rank_bm25 import BM25Okapi

from backend.datatypes import DocumentChunk, RetrievalResult
from backend.vectorstore.chroma_store import ChromaStore
from backend.config import CONFIG
from backend.logger import logger


class BM25Retriever:
    """BM25 lexical search over document chunks."""

    def __init__(self, store: ChromaStore):
        self.store = store
        self._index: Optional[BM25Okapi] = None
        self._chunks: List[DocumentChunk] = []
        self._tokenized_corpus: List[List[str]] = []

    def build_index(self):
        """Build the BM25 index from all chunks in the store."""
        self._chunks = self.store.get_all_chunks()
        if not self._chunks:
            logger.warning("No chunks found for BM25 indexing")
            return

        self._tokenized_corpus = [
            self._tokenize(chunk.text) for chunk in self._chunks
        ]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(self._chunks)} documents")

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Retrieve top-k chunks using BM25 scoring."""
        top_k = top_k or CONFIG.retrieval.top_k

        if self._index is None:
            self.build_index()

        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in scored_indices:
            if scores[idx] > 0:
                results.append(
                    RetrievalResult(
                        chunk=self._chunks[idx],
                        score=float(scores[idx]),
                        retrieval_method="bm25",
                    )
                )

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing and punctuation removal."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove stopwords (minimal set for drug domain)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "and",
                      "or", "but", "in", "on", "at", "to", "for", "of", "with",
                      "by", "it", "its", "this", "that", "from"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]
