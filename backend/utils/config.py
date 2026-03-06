import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents" / "drugs"
VECTOR_STORE_DIR = BASE_DIR / "multimodal_rag" / "vector_store"
DATA_DIR = BASE_DIR / "data"


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"


class ChunkingConfig(BaseModel):
    default_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_threshold: float = 0.75
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256


class ChromaConfig(BaseModel):
    collection_name: str = "drug_documents"
    persist_directory: str = str(VECTOR_STORE_DIR / "chroma_db")
    distance_metric: str = "cosine"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "phi3:mini"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024


class RetrievalConfig(BaseModel):
    top_k: int = 5
    rrf_k: int = 60
    hyde_enabled: bool = True
    multi_query_count: int = 3
    bm25_weight: float = 0.4
    dense_weight: float = 0.6


class LangSmithConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    project_name: str = "doc-intelligence"
    tracing_enabled: bool = Field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    chroma: ChromaConfig = ChromaConfig()
    ollama: OllamaConfig = OllamaConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    langsmith: LangSmithConfig = LangSmithConfig()
    documents_dir: str = str(DOCUMENTS_DIR)
    data_dir: str = str(DATA_DIR)


CONFIG = AppConfig()
