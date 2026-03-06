"""
Embedding Module — generates embeddings using sentence-transformers (all-MiniLM-L6-v2).
"""
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.utils.config import CONFIG
from backend.utils.logger import logger


class EmbeddingEngine:
    """Generate embeddings using all-MiniLM-L6-v2 via sentence-transformers."""

    def __init__(self):
        self.model_name = CONFIG.embedding.model_name
        self.batch_size = CONFIG.embedding.batch_size
        self.device = CONFIG.embedding.device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return CONFIG.embedding.dimension

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        embedding = self.model.encode(
            text, convert_to_numpy=True, device=self.device, normalize_embeddings=True
        )
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batches."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
