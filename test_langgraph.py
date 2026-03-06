"""
Test script for the new LangGraph generic orchestration
"""
import asyncio
from backend.langchain_integration.tracing import setup_langsmith_tracing

# Initialize tracing BEFORE anything else
setup_langsmith_tracing()

from backend.embeddings.embeddings import EmbeddingEngine
from backend.vectorstore.chroma_store import ChromaStore
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.query_expander import QueryExpander
from backend.agents.llm_client import LLMClient
from backend.agents.safety_guard import SafetyGuard
from backend.langchain_integration.graph import build_orchestrator, OrchestratorState

def run_test():
    print("Initializing isolated components...")
    embedding_engine = EmbeddingEngine()
    chroma_store = ChromaStore()
    dense_retriever = DenseRetriever(chroma_store, embedding_engine)
    bm25_retriever = BM25Retriever(chroma_store)
    
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
    query_expander = QueryExpander(hybrid_retriever)
    llm_client = LLMClient()
    safety_guard = SafetyGuard()
    
    # Build explicit graph
    workflow = build_orchestrator(hybrid_retriever, query_expander, llm_client, safety_guard)
    
    initial_state: OrchestratorState = {
        "query": "What are the common side effects of amoxicillin?",
        "extracted_entities": [],
        "needs_expansion": False,
        "retrieved_chunks": [],
        "final_response": "",
        "citations": [],
        "safety_passed": True,
        "safety_message": "",
    }
    
    print("\n--- Invoking LangGraph Orchestrator ---")
    result = workflow.invoke(initial_state)
    
    print("\n[Extracted Entities Step]:", result.get("extracted_entities"))
    print(f"[Chunks Retrieved]: {len(result.get('retrieved_chunks', []))} chunks")
    print(f"[Safety Passed]: {result.get('safety_passed')}")
    print("\n[Final Generation]:")
    print(result.get("final_response"))
    print("\n[Citations Count]:", len(result.get("citations", [])))

if __name__ == "__main__":
    run_test()
