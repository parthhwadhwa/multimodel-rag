import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any

from utils.datatypes import DocumentChunk, QueryResult
from utils.logger import logger

class HealthcareVectorStore:
    def __init__(self, index_path: str = "data/faiss_index.index", metadata_path: str = "data/metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = 384 # all-MiniLM-L6-v2 dimension
        self.max_chunks = 1000
        
        self.index = None # Lazy load
        self.metadata_store: Dict[str, dict] = {}
        self.current_id = 0
        
    def _initialize_index(self):
        if self.index is None:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded healthcare index with {self.index.ntotal} vectors.")
            else:
                self.index = faiss.IndexFlatIP(self.dim)
                logger.info("Initialized new Healthcare IndexFlatIP.")
                
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.current_id = data.get("current_id", 0)
                    self.metadata_store = data.get("store", {})
                logger.info(f"Loaded healthcare metadata for {len(self.metadata_store)} chunks.")

    def add_documents(self, chunks: List[DocumentChunk]):
        self._initialize_index()
        
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
                
            if self.current_id >= self.max_chunks:
                logger.warning(f"Maximum chunks ({self.max_chunks}) reached for Healthcare RAG.")
                break
                
            embeddings.append(chunk.embedding)
            
            chunk_dict = chunk.model_dump()
            chunk_dict['index_type'] = 'text'
            self.metadata_store[f"id_{self.current_id}"] = chunk_dict
            self.current_id += 1
                
        if embeddings:
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors) # Required for inner product to be cosine similarity
            self.index.add(vectors)
            logger.info(f"Added {len(embeddings)} healthcare chunks to FAISS.")
            
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        self._initialize_index()
        
        if self.index is None or self.index.ntotal == 0:
            return []
            
        vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            key = f"id_{idx}"
            if key in self.metadata_store:
                chunk_data = self.metadata_store[key]
                chunk = DocumentChunk(**chunk_data)
                results.append(QueryResult(chunk=chunk, score=float(dist)))
        return results

    def save(self):
        if self.index is not None:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    "current_id": self.current_id,
                    "store": self.metadata_store
                }, f)
            logger.info("Healthcare vector store saved.")
