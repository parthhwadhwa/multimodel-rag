#!/usr/bin/env python3
"""
ingest_documents.py

Script to automatically load, preprocess, chunk, and embed all PDFs in
`documents/drugs` into the ChromaDB vector store.
"""

import argparse
import os
import sys

# Ensure the backend module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.pdf_loader import PDFLoader
from backend.preprocessing.preprocessor import TextPreprocessor
from backend.preprocessing.structure_detector import StructureDetector
from backend.chunking.chunking_manager import ChunkingManager
from backend.utils.datatypes import ChunkingStrategy
from backend.embeddings.embeddings import EmbeddingEngine
from backend.vectorstore.chroma_store import ChromaStore
from backend.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Ingest drug PDFs into the vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing ChromaDB collection before indexing",
    )
    args = parser.parse_args()

    # 1. Initialize components
    logger.info("Initializing ingestion components...")
    pdf_loader = PDFLoader()
    preprocessor = TextPreprocessor()
    structure_detector = StructureDetector()
    embedder = EmbeddingEngine()
    chunking_manager = ChunkingManager(embedding_model=embedder.model)
    chroma_store = ChromaStore()

    if args.rebuild:
        logger.warning("Rebuild flag detected. Deleting existing Chroma collection...")
        chroma_store.delete_collection()
        # Re-initialize collection
        chroma_store = ChromaStore()

    documents_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "documents", "drugs"
    )

    if not os.path.exists(documents_dir):
        logger.error(f"Documents directory not found: {documents_dir}")
        sys.exit(1)

    # 2. Extract and Process
    logger.info(f"Loading PDFs from: {documents_dir}")
    pages = pdf_loader.load_directory(documents_dir)
    
    if not pages:
        logger.error("No pages loaded. Exiting.")
        sys.exit(1)

    # Preprocess text and apply formatting annotations
    cleaned_pages = preprocessor.process(pages)

    # Identify structural boundaries (headings, lists, paragraphs)
    sections = structure_detector.detect_sections(cleaned_pages)

    # 3. Chunking
    # Use PARENT_CHILD chunking strategy to map small chunks to larger context
    chunks = chunking_manager.chunk_sections(
        sections,
        document_name="bulk_ingest", # We can also group by individual document
        strategy=ChunkingStrategy.PARENT_CHILD
    )

    # 4. Embeddings
    # We only need to embed child chunks for retrieval. Parent chunks are for context.
    child_chunks = [c for c in chunks if not c.metadata.get("is_parent", False)]
    parent_chunks = [c for c in chunks if c.metadata.get("is_parent", False)]
    
    logger.info(f"Generated {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks.")

    logger.info("Generating embeddings for child chunks...")
    texts_to_embed = [chunk.text for chunk in child_chunks]
    embeddings = embedder.embed_texts(texts_to_embed)

    # Parent chunks don't need dense embeddings for search, but Chroma expects them.
    # We'll use a dummy vector for parents or embed them depending on Chroma requirements.
    # Actually, we can just embed all chunks to keep it simple, or assign zeros if not searched.
    # Let's just embed the parent chunks as well to avoid schema mismatches if requested.
    logger.info("Generating embeddings for parent chunks...")
    parent_texts_to_embed = [chunk.text for chunk in parent_chunks]
    parent_embeddings = embedder.embed_texts(parent_texts_to_embed)

    # 5. Store in ChromaDB
    logger.info("Storing chunks in ChromaDB...")
    
    # Add parents first
    if parent_chunks:
        chroma_store.add_documents(parent_chunks, parent_embeddings)
        
    # Add children
    if child_chunks:
        chroma_store.add_documents(child_chunks, embeddings)

    logger.info(f"Ingestion complete. Total documents in Chroma: {chroma_store.count()}")

if __name__ == "__main__":
    main()
