"""
Semantic Chunker — groups sentences by embedding similarity.
Splits when cosine similarity between consecutive sentences drops below a threshold.
"""
import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.utils.datatypes import DocumentChunk, DocumentSection
from backend.utils.config import CONFIG
from backend.utils.logger import logger


class SemanticChunker:
    """Split text at semantic boundaries using embedding similarity."""

    def __init__(self, threshold: float = None, model: SentenceTransformer = None):
        self.threshold = threshold or CONFIG.chunking.semantic_threshold
        self._model = model

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                CONFIG.embedding.model_name, device=CONFIG.embedding.device
            )
        return self._model

    def chunk(
        self, sections: List[DocumentSection], document_name: str = ""
    ) -> List[DocumentChunk]:
        chunks = []
        for section in sections:
            if not section.content.strip():
                continue
            section_chunks = self._semantic_split(section, document_name)
            chunks.extend(section_chunks)
        return chunks

    def _semantic_split(
        self, section: DocumentSection, document_name: str
    ) -> List[DocumentChunk]:
        text = section.content
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [
                DocumentChunk(
                    text=text,
                    metadata={
                        "document_name": document_name,
                        "page_number": section.page_number,
                        "section_title": section.title,
                        "chunk_index": 0,
                        "chunking_strategy": "semantic",
                    },
                )
            ]

        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-10
            )
            similarities.append(float(sim))

        # Find split points where similarity drops
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))

        chunks = []
        for idx in range(len(split_indices) - 1):
            start = split_indices[idx]
            end = split_indices[idx + 1]
            chunk_text = " ".join(sentences[start:end])

            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        metadata={
                            "document_name": document_name,
                            "page_number": section.page_number,
                            "section_title": section.title,
                            "chunk_index": idx,
                            "chunking_strategy": "semantic",
                        },
                    )
                )
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
