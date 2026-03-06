"""
Parent-Child Chunker — creates large parent chunks and small child chunks.
Child chunks link back to parents via metadata for context expansion during retrieval.
"""
from typing import List, Tuple
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.datatypes import DocumentChunk, DocumentSection
from backend.utils.config import CONFIG


class ParentChildChunker:
    """Create hierarchical parent-child chunks for context-aware retrieval."""

    def __init__(
        self,
        parent_chunk_size: int = None,
        child_chunk_size: int = None,
        child_overlap: int = 30,
    ):
        self.parent_chunk_size = parent_chunk_size or CONFIG.chunking.parent_chunk_size
        self.child_chunk_size = child_chunk_size or CONFIG.chunking.child_chunk_size
        self.child_overlap = child_overlap

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(
        self, sections: List[DocumentSection], document_name: str = ""
    ) -> List[DocumentChunk]:
        """Returns child chunks (for embedding/retrieval) with parent_id references."""
        all_chunks = []

        for section in sections:
            if not section.content.strip():
                continue

            text = f"{section.title}\n\n{section.content}" if section.title else section.content
            parent_texts = self.parent_splitter.split_text(text)

            for p_idx, parent_text in enumerate(parent_texts):
                parent_id = str(uuid.uuid4())

                # Store parent chunk (marked as parent in metadata)
                parent_chunk = DocumentChunk(
                    id=parent_id,
                    text=parent_text,
                    metadata={
                        "document_name": document_name,
                        "page_number": section.page_number,
                        "section_title": section.title,
                        "chunk_index": p_idx,
                        "chunking_strategy": "parent_child",
                        "is_parent": True,
                        "parent_chunk_size": self.parent_chunk_size,
                    },
                )
                all_chunks.append(parent_chunk)

                # Create child chunks
                child_texts = self.child_splitter.split_text(parent_text)
                for c_idx, child_text in enumerate(child_texts):
                    child_chunk = DocumentChunk(
                        text=child_text,
                        parent_id=parent_id,
                        metadata={
                            "document_name": document_name,
                            "page_number": section.page_number,
                            "section_title": section.title,
                            "chunk_index": c_idx,
                            "chunking_strategy": "parent_child",
                            "is_parent": False,
                            "parent_id": parent_id,
                            "child_chunk_size": self.child_chunk_size,
                        },
                    )
                    all_chunks.append(child_chunk)

        return all_chunks
