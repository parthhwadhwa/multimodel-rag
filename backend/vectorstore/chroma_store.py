"""
ChromaDB Vector Store — persistent vector database for document chunks.
Stores embeddings with metadata (document name, page number, section title).
"""
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from backend.utils.datatypes import DocumentChunk, RetrievalResult
from backend.utils.config import CONFIG
from backend.utils.logger import logger


class ChromaStore:
    """ChromaDB persistent vector store for document intelligence system."""

    def __init__(self, collection_name: str = None, persist_directory: str = None):
        self.collection_name = collection_name or CONFIG.chroma.collection_name
        self.persist_directory = persist_directory or CONFIG.chroma.persist_directory
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": CONFIG.chroma.distance_metric},
            )
            logger.info(
                f"Collection '{self.collection_name}' ready "
                f"({self._collection.count()} documents)"
            )
        return self._collection

    def add_documents(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ):
        """Add document chunks with embeddings to the collection."""
        if not chunks or not embeddings:
            return

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = {k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
            if chunk.parent_id:
                meta["parent_id"] = chunk.parent_id
            metadatas.append(meta)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(f"Added {len(ids)} chunks to ChromaDB")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Search for similar documents by embedding."""
        top_k = top_k or CONFIG.retrieval.top_k

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        retrieval_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk = DocumentChunk(
                    id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                # ChromaDB returns distances; for cosine, distance = 1 - similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 - distance  # Convert to similarity
                retrieval_results.append(
                    RetrievalResult(chunk=chunk, score=score, retrieval_method="dense")
                )

        return retrieval_results

    def get_all_documents(self) -> List[str]:
        """Get all unique document texts (for BM25 indexing)."""
        result = self.collection.get(include=["documents"])
        return result.get("documents", [])

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks with metadata."""
        result = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for i in range(len(result["ids"])):
            chunks.append(
                DocumentChunk(
                    id=result["ids"][i],
                    text=result["documents"][i],
                    metadata=result["metadatas"][i] if result["metadatas"] else {},
                )
            )
        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID (for parent-child expansion)."""
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if result["ids"]:
            return DocumentChunk(
                id=result["ids"][0],
                text=result["documents"][0],
                metadata=result["metadatas"][0] if result["metadatas"] else {},
            )
        return None

    def get_parent_chunk(self, parent_id: str) -> Optional[DocumentChunk]:
        """Retrieve the parent chunk for context expansion."""
        return self.get_chunk_by_id(parent_id)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")

    def list_documents(self) -> List[str]:
        """Get list of unique document names in the store."""
        result = self.collection.get(include=["metadatas"])
        doc_names = set()
        for meta in result.get("metadatas", []):
            if meta and "document_name" in meta:
                doc_names.add(meta["document_name"])
        return sorted(doc_names)
