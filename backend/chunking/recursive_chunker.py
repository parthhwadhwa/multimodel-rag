"""
Recursive Character Text Splitter — wraps LangChain's RecursiveCharacterTextSplitter.
"""
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.datatypes import DocumentChunk, DocumentSection
from backend.utils.config import CONFIG


class RecursiveChunker:
    """Split text by recursively trying different separators."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or CONFIG.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or CONFIG.chunking.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(
        self, sections: List[DocumentSection], document_name: str = ""
    ) -> List[DocumentChunk]:
        chunks = []
        for section in sections:
            if not section.content.strip():
                continue
            text_to_split = f"{section.title}\n\n{section.content}" if section.title else section.content
            split_texts = self.splitter.split_text(text_to_split)

            for i, text in enumerate(split_texts):
                chunks.append(
                    DocumentChunk(
                        text=text,
                        metadata={
                            "document_name": document_name,
                            "page_number": section.page_number,
                            "section_title": section.title,
                            "chunk_index": i,
                            "chunking_strategy": "recursive",
                            "chunk_size": self.chunk_size,
                        },
                    )
                )
        return chunks
