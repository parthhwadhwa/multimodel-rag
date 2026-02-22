from typing import List
from sentence_transformers import SentenceTransformer
from utils.config import CONFIG
from utils.logger import logger

class TextEmbedder:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = CONFIG.text_embedding.model_name
        self.model_name = model_name
        logger.info(f"Loading TextEmbedder model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load TextEmbedder model {self.model_name}: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        try:

            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed multiple texts: {e}")
            raise
