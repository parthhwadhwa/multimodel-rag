from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_file: str
    modality: Modality
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class QueryResult(BaseModel):
    chunk: DocumentChunk
    score: float
