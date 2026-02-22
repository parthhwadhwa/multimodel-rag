from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.datatypes import DocumentChunk, Modality
from utils.logger import logger

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_document(self, text: str, source_file: str, base_metadata: dict = None) -> List[DocumentChunk]:
        if base_metadata is None:
            base_metadata = {}
        
        try:

            text_chunks = self.splitter.split_text(text)
            

            doc_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                metadata = base_metadata.copy()
                metadata["chunk_index"] = i
                
                doc_chunk = DocumentChunk(
                    source_file=source_file,
                    modality=Modality.TEXT,
                    text_content=chunk_text,
                    metadata=metadata
                )
                doc_chunks.append(doc_chunk)
            return doc_chunks
        except Exception as e:
            logger.error(f"Error chunking document {source_file}: {e}")
            return []
