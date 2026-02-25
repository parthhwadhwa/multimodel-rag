import argparse
import os
import glob

from utils.logger import logger
from ingestion.chunker import HealthcareChunker
from embedding.text_embedder import HealthcareEmbedder
from vector_store.faiss_store import HealthcareVectorStore
from retrieval.retriever import HealthcareRetriever
from generation.ollama_client import HealthcareOllamaClient
from generation.gemini_client import HealthcareGeminiClient

def print_memory_usage():
    logger.info("Memory tracking disabled (psutil not installed).")

def ingest_data(data_dir: str):
    logger.info("Initializing Healthcare embedding model (CPU mode)...")
    embedder = HealthcareEmbedder()
    chunker = HealthcareChunker()
    vstore = HealthcareVectorStore()
    
    logger.info(f"Starting lightweight JSON ingestion from {data_dir}...")
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    all_chunks = []
    for jf in json_files:
        if jf.endswith('metadata.json'):
            continue 
        chunks = chunker.chunk_json_file(jf)
        all_chunks.extend(chunks)
        
    if not all_chunks:
        logger.warning("No valid healthcare drugs loaded. Exiting.")
        return
        
    logger.info(f"Generated {len(all_chunks)} lightweight chunks. embedding in batches of 16...")
    print_memory_usage()
    
    texts = [c.text_content for c in all_chunks]
    embeddings = embedder.embed_texts(texts)
    
    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb
                
    logger.info("Storing embeddings in memory-efficient FAISS IndexFlatIP...")
    vstore.add_documents(all_chunks)
    logger.info("Ingestion complete.")
    print_memory_usage()

def query_system(query: str, use_gemini: bool = False):
    logger.info("Initializing domain-specific models for querying...")
    embedder = HealthcareEmbedder()
    vstore = HealthcareVectorStore()
    retriever = HealthcareRetriever(vstore, embedder)
    
    results = retriever.retrieve(query)
    
    if not results:
        logger.warning("No context retrieved. System may be empty or query matches nothing.")
        
    print("\n" + "="*50)
    print("RETRIEVED HEALTHCARE CONTEXT:")
    for i, res in enumerate(results):
        mod = res.chunk.modality.value.upper()
        src = os.path.basename(res.chunk.source_file)
        section = res.chunk.metadata.get('section', 'unknown')
        score = f"{res.score:.4f}"
        print(f"[{i+1}] {mod} | {src} ({section}) | Score: {score}")
    print("="*50 + "\n")
    
    if use_gemini:
        llm_client = HealthcareGeminiClient()
        print("Using Google GEMINI API with Safety Constraints...\n")
    else:
        llm_client = HealthcareOllamaClient()
        print("Using OLLAMA (mistral) with Safety Constraints...\n")
        
    print("Answer: ", end="", flush=True)
    for chunk_text in llm_client.generate_stream(query, results):
        print(chunk_text, end="", flush=True)
    print("\n")
    print_memory_usage()

def interactive_loop(use_gemini: bool = False):
    print("\n" + "="*50)
    print("Welcome to Healthcare RAG")
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
    parser = argparse.ArgumentParser(description="Healthcare RAG System")
    parser.add_argument("--ingest", type=str, help="Directory containing JSON files to ingest.")
    parser.add_argument("--query", type=str, help="The question you want to ask the system.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop.")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini API instead of local Ollama for generation.")
    
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
