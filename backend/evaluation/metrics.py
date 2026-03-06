from typing import List, Dict, Any
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.utils.logger import logger

def calculate_precision_at_k(retrieved_docs: List[str], expected_doc: str, k: int) -> float:
    """Calculate Precision@k: relevant retrieved / k."""
    top_k = retrieved_docs[:k]
    relevant = sum(1 for doc in top_k if doc == expected_doc)
    return relevant / k if k > 0 else 0.0

def calculate_recall_at_k(retrieved_docs: List[str], expected_doc: str, k: int) -> float:
    """
    Calculate Recall@k: relevant retrieved / total_relevant_documents.
    Here total relevant documents for a query is assumed to be 1.
    """
    top_k = retrieved_docs[:k]
    return 1.0 if expected_doc in top_k else 0.0

def calculate_mrr(retrieved_docs: List[str], expected_doc: str) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1 / rank_of_first_relevant_doc."""
    for i, doc in enumerate(retrieved_docs):
        if doc == expected_doc:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_retrieval(retriever: HybridRetriever, queries: List[Dict[str, str]], k: int = 5) -> Dict[str, float]:
    """
    Evaluate the retrieval system on a set of queries.
    Expects queries as [{"query": "...", "relevant_doc": "..."}].
    Returns aggregated metrics: Precision@k, Recall@k, and MRR.
    """
    total_queries = len(queries)
    if total_queries == 0:
        return {"precision@k": 0.0, "recall@k": 0.0, "MRR": 0.0}

    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    logger.info(f"Starting evaluation on {total_queries} queries (k={k})...")

    for q in queries:
        query_text = q["query"]
        expected_doc = q["relevant_doc"]

        # Run retriever
        results = retriever.retrieve(query_text, top_k=k)
        
        # Extract document names uniquely preserving rank order
        retrieved_docs = []
        seen = set()
        for r in results:
            doc_name = r.chunk.metadata.get("document_name", "Unknown")
            if doc_name not in seen:
                seen.add(doc_name)
                retrieved_docs.append(doc_name)

        precision = calculate_precision_at_k(retrieved_docs, expected_doc, k)
        recall = calculate_recall_at_k(retrieved_docs, expected_doc, k)
        mrr = calculate_mrr(retrieved_docs, expected_doc)

        total_precision += precision
        total_recall += recall
        total_mrr += mrr

    return {
        "Precision": round(total_precision / total_queries, 4),
        "Recall": round(total_recall / total_queries, 4),
        "MRR": round(total_mrr / total_queries, 4)
    }
