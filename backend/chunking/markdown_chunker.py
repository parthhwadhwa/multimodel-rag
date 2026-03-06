"""
Markdown Header Text Splitter — splits on heading levels to preserve semantic sections.
"""
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter as LCMarkdownSplitter

from backend.datatypes import DocumentChunk, DocumentSection


class MarkdownChunker:
    """Split Markdown-formatted text on heading boundaries."""

    def __init__(self):
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        self.splitter = LCMarkdownSplitter(
            headers_to_split_on=self.headers_to_split_on,
        )

    def chunk(
        self, sections: List[DocumentSection], document_name: str = ""
    ) -> List[DocumentChunk]:
        chunks = []
        for section in sections:
            if not section.content.strip():
                continue

            full_text = f"# {section.title}\n\n{section.content}" if section.title else section.content
            try:
                split_docs = self.splitter.split_text(full_text)
            except Exception:
                # Fallback: treat entire section as one chunk
                chunks.append(
                    DocumentChunk(
                        text=section.content,
                        metadata={
                            "document_name": document_name,
                            "page_number": section.page_number,
                            "section_title": section.title,
                            "chunk_index": 0,
                            "chunking_strategy": "markdown",
                        },
                    )
                )
                continue

            for i, doc in enumerate(split_docs):
                header_title = ""
                if hasattr(doc, "metadata"):
                    header_title = doc.metadata.get("h1", doc.metadata.get("h2", doc.metadata.get("h3", "")))
                text = doc.page_content if hasattr(doc, "page_content") else str(doc)

                chunks.append(
                    DocumentChunk(
                        text=text,
                        metadata={
                            "document_name": document_name,
                            "page_number": section.page_number,
                            "section_title": header_title or section.title,
                            "chunk_index": i,
                            "chunking_strategy": "markdown",
                        },
                    )
                )
        return chunks
