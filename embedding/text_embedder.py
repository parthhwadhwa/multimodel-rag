import torch
from typing import List
from sentence_transformers import SentenceTransformer
from utils.logger import logger

class HealthcareEmbedder:
    """
    Optimized Embeddings for 8GB RAM MacBook:
    - all-MiniLM-L6-v2 (small size, 384 dims)
    - CPU mode only
    - Batch size limited to 16
    """
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.batch_size = 16 # Small batch size to avoid RAM spikes
        
        logger.info(f"Loading HealthcareEmbedder model: {self.model_name} on CPU")
        try:
            self.model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:
            logger.error(f"Failed to load HealthcareEmbedder model {self.model_name}: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True, device="cpu")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size, 
                convert_to_numpy=True,
                device="cpu"
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed multiple healthcare texts: {e}")
            raise
