"""
Chunking Manager — unified interface to select and run any chunking strategy.
"""
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from backend.datatypes import DocumentChunk, DocumentSection, ChunkingStrategy
from backend.config import CONFIG
from backend.logger import logger
from backend.chunking.recursive_chunker import RecursiveChunker
from backend.chunking.token_chunker import TokenChunker
from backend.chunking.markdown_chunker import MarkdownChunker
from backend.chunking.semantic_chunker import SemanticChunker
from backend.chunking.parent_child_chunker import ParentChildChunker


class ChunkingManager:
    """Factory and manager for all chunking strategies."""

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self._embedding_model = embedding_model

    def get_chunker(self, strategy: ChunkingStrategy):
        """Return the appropriate chunker for the given strategy."""
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker()
        elif strategy == ChunkingStrategy.TOKEN:
            return TokenChunker()
        elif strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownChunker()
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(model=self._embedding_model)
        elif strategy == ChunkingStrategy.PARENT_CHILD:
            return ParentChildChunker()
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to recursive")
            return RecursiveChunker()

    def chunk_sections(
        self,
        sections: List[DocumentSection],
        document_name: str = "",
        strategy: ChunkingStrategy = None,
    ) -> List[DocumentChunk]:
        """Chunk sections using the specified or default strategy."""
        strategy = strategy or ChunkingStrategy(CONFIG.chunking.default_strategy)
        chunker = self.get_chunker(strategy)
        logger.info(f"Chunking {len(sections)} sections with strategy: {strategy.value}")
        chunks = chunker.chunk(sections, document_name)
        logger.info(f"Generated {len(chunks)} chunks")
        return chunks

    @staticmethod
    def available_strategies() -> List[str]:
        """Return list of available chunking strategy names."""
        return [s.value for s in ChunkingStrategy]
