from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    TOKEN = "token"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"


class DocumentPage(BaseModel):
    page_number: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    title: str
    content: str
    level: int = 1
    page_number: int = 0


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None

    @property
    def document_name(self) -> str:
        return self.metadata.get("document_name", "unknown")

    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", 0)

    @property
    def section_title(self) -> str:
        return self.metadata.get("section_title", "")


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    retrieval_method: str = "dense"


class QueryState(BaseModel):
    query: str
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    response: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    safety_passed: bool = True
    safety_message: str = ""
    needs_expansion: bool = False
