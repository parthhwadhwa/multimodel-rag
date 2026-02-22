import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

class TextEmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384

class ImageEmbeddingConfig(BaseModel):
    model_name: str = "openai/clip-vit-base-patch32"
    dimension: int = 512

class RetrievalConfig(BaseModel):
    top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

class GenerationConfig(BaseModel):
    ollama_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.7

class GeminiConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "gemini-2.5-flash"

class AppConfig(BaseModel):
    text_embedding: TextEmbeddingConfig = TextEmbeddingConfig()
    image_embedding: ImageEmbeddingConfig = ImageEmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    gemini: GeminiConfig = GeminiConfig()
    vector_store_path: str = "faiss_index"
    metadata_store_path: str = "metadata.json"

def load_config(config_path: str = "config.yaml") -> AppConfig:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        return AppConfig(**config_dict)
    return AppConfig()


CONFIG = load_config()
