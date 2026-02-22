import torch
from PIL import Image
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel
from utils.config import CONFIG
from utils.logger import logger

class ImageEmbedder:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = CONFIG.image_embedding.model_name
        self.model_name = model_name
        logger.info(f"Loading ImageEmbedder model: {self.model_name}")
        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"ImageEmbedder loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ImageEmbedder model {self.model_name}: {e}")
            raise

    def embed_image(self, image_path: str) -> List[float]:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to embed image {image_path}: {e}")
            raise

    def embed_text_query(self, text: str) -> List[float]:
        """
        Embeds a text query using the same CLIP model to enable multimodal search
        (text searching for images).
        """
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to embed text query with CLIP: {e}")
            raise
