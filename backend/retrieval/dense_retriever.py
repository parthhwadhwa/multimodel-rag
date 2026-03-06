"""
Dense Retriever — vector similarity search through ChromaDB.
"""
from typing import List, Optional, Dict, Any

from backend.utils.datatypes import RetrievalResult
from backend.vectorstore.chroma_store import ChromaStore
from backend.embeddings.embeddings import EmbeddingEngine
from backend.utils.config import CONFIG


class DenseRetriever:
    """Retrieve documents using dense vector similarity."""

    def __init__(self, store: ChromaStore, embedder: EmbeddingEngine):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k similar chunks for a query."""
        top_k = top_k or CONFIG.retrieval.top_k
        query_embedding = self.embedder.embed_text(query)
        results = self.store.search(query_embedding, top_k=top_k, where=filters)

        for r in results:
            r.retrieval_method = "dense"

        return results
