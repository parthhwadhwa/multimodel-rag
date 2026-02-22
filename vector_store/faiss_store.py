import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any

from utils.datatypes import DocumentChunk, Modality, QueryResult
from utils.config import CONFIG
from utils.logger import logger

class VectorStore:
    def __init__(self, index_path: str = None, metadata_path: str = None):
        self.index_path_text = (index_path or CONFIG.vector_store_path) + "_text.index"
        self.index_path_image = (index_path or CONFIG.vector_store_path) + "_image.index"
        self.metadata_path = metadata_path or CONFIG.metadata_store_path
        
        self.text_dim = CONFIG.text_embedding.dimension
        self.image_dim = CONFIG.image_embedding.dimension
        

        self.text_index = faiss.IndexFlatL2(self.text_dim)
        self.image_index = faiss.IndexFlatL2(self.image_dim)
        
        self.metadata_store: Dict[int, dict] = {}
        self.current_text_id = 0
        self.current_image_id = 0
        
        self.load()

    def add_documents(self, chunks: List[DocumentChunk]):
        text_embeddings = []
        text_ids = []
        
        image_embeddings = []
        image_ids = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            
            if chunk.modality == Modality.TEXT:
                text_embeddings.append(chunk.embedding)
                text_ids.append(self.current_text_id)
                

                chunk_dict = chunk.model_dump()
                chunk_dict['index_type'] = 'text'
                self.metadata_store[f"text_{self.current_text_id}"] = chunk_dict
                self.current_text_id += 1
                
            elif chunk.modality == Modality.IMAGE:
                image_embeddings.append(chunk.embedding)
                image_ids.append(self.current_image_id)
                

                chunk_dict = chunk.model_dump()
                chunk_dict['index_type'] = 'image'
                self.metadata_store[f"image_{self.current_image_id}"] = chunk_dict
                self.current_image_id += 1
                
        if text_embeddings:
            vectors = np.array(text_embeddings, dtype=np.float32)
            self.text_index.add(vectors)
            logger.info(f"Added {len(text_embeddings)} text chunks to FAISS.")
            
        if image_embeddings:
            vectors = np.array(image_embeddings, dtype=np.float32)
            self.image_index.add(vectors)
            logger.info(f"Added {len(image_embeddings)} image chunks to FAISS.")
            
        self.save()

    def search_text(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        if self.text_index.ntotal == 0:
            return []
            
        vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.text_index.search(vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            key = f"text_{idx}"
            if key in self.metadata_store:
                chunk_data = self.metadata_store[key]
                chunk = DocumentChunk(**chunk_data)
                results.append(QueryResult(chunk=chunk, score=float(dist)))
        return results

    def search_image(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        if self.image_index.ntotal == 0:
            return []
            
        vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.image_index.search(vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            key = f"image_{idx}"
            if key in self.metadata_store:
                chunk_data = self.metadata_store[key]
                chunk = DocumentChunk(**chunk_data)
                results.append(QueryResult(chunk=chunk, score=float(dist)))
        return results

    def save(self):
        faiss.write_index(self.text_index, self.index_path_text)
        faiss.write_index(self.image_index, self.index_path_image)
        with open(self.metadata_path, 'w') as f:
            json.dump({
                "current_text_id": self.current_text_id,
                "current_image_id": self.current_image_id,
                "store": self.metadata_store
            }, f)
        logger.info("Vector store and metadata saved to disk.")

    def load(self):
        if os.path.exists(self.index_path_text):
            self.text_index = faiss.read_index(self.index_path_text)
            logger.info(f"Loaded text index with {self.text_index.ntotal} vectors.")
            
        if os.path.exists(self.index_path_image):
            self.image_index = faiss.read_index(self.index_path_image)
            logger.info(f"Loaded image index with {self.image_index.ntotal} vectors.")
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.current_text_id = data.get("current_text_id", 0)
                self.current_image_id = data.get("current_image_id", 0)
                self.metadata_store = data.get("store", {})
            logger.info(f"Loaded metadata for {len(self.metadata_store)} total chunks.")
