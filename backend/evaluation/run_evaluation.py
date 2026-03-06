import os
import json
from pathlib import Path

from backend.embeddings.embeddings import EmbeddingEngine
from backend.vectorstore.chroma_store import ChromaStore
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.evaluation.metrics import evaluate_retrieval
from backend.utils.logger import logger

def main():
    # Load evaluation queries
    queries_path = Path(__file__).parent / "evaluation_queries.json"
    if not queries_path.exists():
        logger.error(f"Cannot find {queries_path}")
        return

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info("Initializing Retrieval Components...")
    
    # Init system identically to API
    embedding_engine = EmbeddingEngine()
    chroma_store = ChromaStore()
    dense_retriever = DenseRetriever(chroma_store, embedding_engine)
    bm25_retriever = BM25Retriever(chroma_store)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

    # Note: Evaluator assumes ChromaDB has already been populated with documents and BM25 index built.
    
    k = 5
    logger.info(f"Running evaluation on {len(queries)} queries (top_{k})...")
    results = evaluate_retrieval(hybrid_retriever, queries, k=k)

    print("\n## Retrieval Evaluation Results\n")
    print(f"Precision : {results.get('Precision', 0):.2f}")
    print(f"Recall    : {results.get('Recall', 0):.2f}")
    print(f"MRR       : {results.get('MRR', 0):.2f}")

if __name__ == "__main__":
    main()
