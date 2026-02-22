import argparse
import sys
from typing import List

from utils.logger import logger
from utils.datatypes import Modality
from ingestion.loader import DocumentLoader
from embedding.text_embedder import TextEmbedder
from embedding.image_embedder import ImageEmbedder
from vector_store.faiss_store import VectorStore
from retrieval.retriever import Retriever
from generation.ollama_client import OllamaClient
from generation.gemini_client import GeminiClient

def ingest_data(data_dir: str):
    logger.info("Initializing embedding models...")
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    
    logger.info(f"Starting ingestion from {data_dir}...")
    loader = DocumentLoader()
    chunks = loader.load_directory(data_dir)
    
    if not chunks:
        logger.warning("No documents loaded. Exiting.")
        return
        
    logger.info("Generating embeddings...")
    for chunk in chunks:
        if chunk.modality == Modality.TEXT:
            if chunk.text_content:
                chunk.embedding = text_embedder.embed_text(chunk.text_content)
        elif chunk.modality == Modality.IMAGE:
            if chunk.image_path:
                chunk.embedding = image_embedder.embed_image(chunk.image_path)
                
    logger.info("Storing embeddings in FAISS...")
    vstore = VectorStore()
    vstore.add_documents(chunks)
    logger.info("Ingestion complete.")

def query_system(query: str, use_gemini: bool = False):
    logger.info("Initializing models for querying...")
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    vstore = VectorStore()
    retriever = Retriever(vstore, text_embedder, image_embedder)
    

    results = retriever.retrieve(query)
    
    if not results:
        logger.warning("No context retrieved. System may be empty or query matches nothing.")
        
    print("\n" + "="*50)
    print("RETRIEVED CONTEXT SUMMARY:")
    for i, res in enumerate(results):
        mod = res.chunk.modality.value.upper()
        src = res.chunk.source_file
        score = f"{res.score:.4f}"
        print(f"[{i+1}] {mod} | {src} | Score: {score}")
    print("="*50 + "\n")
    

    if use_gemini:
        llm_client = GeminiClient()
        print("Using GEMINI API for final reasoning...\n")
    else:
        llm_client = OllamaClient()
        print(f"Using OLLAMA (Local) for final reasoning...\n")
        
    print("Answer: ", end="", flush=True)
    for chunk_text in llm_client.generate_stream(query, results):
        print(chunk_text, end="", flush=True)
    print("\n")

def interactive_loop(use_gemini: bool = False):
    print("\n" + "="*50)
    print("Welcome to Multimodal RAG Interactive Mode!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            if not user_input:
                continue
            query_system(user_input, use_gemini)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG System")
    parser.add_argument("--ingest", type=str, help="Directory containing PDF, TXT, and Image files to ingest.")
    parser.add_argument("--query", type=str, help="The question you want to ask the system.")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini API instead of local Ollama for generation.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop.")
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest_data(args.ingest)
    elif args.query:
        query_system(args.query, args.use_gemini)
    elif args.interactive or (not args.ingest and not args.query):
        interactive_loop(args.use_gemini)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
