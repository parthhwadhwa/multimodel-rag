import os
import fitz
from typing import List
from pathlib import Path
from utils.datatypes import DocumentChunk, Modality
from utils.logger import logger
from ingestion.chunker import TextChunker

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_pdf(self, file_path: str) -> List[DocumentChunk]:
        logger.info(f"Loading PDF: {file_path}")
        chunks = []
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for i, page in enumerate(doc):
                text = page.get_text()
                full_text += text + "\n"
            
            chunks = self.chunker.chunk_document(
                text=full_text,
                source_file=file_path,
                base_metadata={"file_type": "pdf"}
            )
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
        return chunks

    def load_text(self, file_path: str) -> List[DocumentChunk]:
        logger.info(f"Loading Text file: {file_path}")
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            chunks = self.chunker.chunk_document(
                text=full_text,
                source_file=file_path,
                base_metadata={"file_type": "txt"}
            )
        except Exception as e:
            logger.error(f"Failed to load Text {file_path}: {e}")
        return chunks

    def load_image(self, file_path: str) -> List[DocumentChunk]:
        logger.info(f"Loading Image file: {file_path}")
        try:

            chunk = DocumentChunk(
                source_file=file_path,
                modality=Modality.IMAGE,
                image_path=file_path,
                metadata={"file_type": Path(file_path).suffix[1:]}
            )
            return [chunk]
        except Exception as e:
            logger.error(f"Failed to load Image {file_path}: {e}")
            return []

    def load_directory(self, directory_path: str) -> List[DocumentChunk]:
        all_chunks = []
        logger.info(f"Ingesting directory: {directory_path}")
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = file.lower().split('.')[-1]
                
                if ext == "pdf":
                    all_chunks.extend(self.load_pdf(file_path))
                elif ext in ["txt", "md"]:
                    all_chunks.extend(self.load_text(file_path))
                elif ext in ["png", "jpg", "jpeg"]:
                    all_chunks.extend(self.load_image(file_path))
                else:
                    logger.debug(f"Skipping unsupported file type: {file_path}")

        logger.info(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks
