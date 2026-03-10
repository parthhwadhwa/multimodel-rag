"""
RAG Agent — LangGraph state-machine agent for document intelligence.
Orchestrates: safety_check → classify → retrieve → [expand] → generate.
"""
from typing import TypedDict, List, Dict, Any, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END

from backend.utils.datatypes import RetrievalResult, QueryState
from backend.rag.llm_client import LLMClient
from backend.rag.safety_guard import SafetyGuard
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.query_expander import QueryExpander
from backend.utils.config import CONFIG
from backend.utils.logger import logger


class AgentState(TypedDict):
    query: str
    expanded_queries: List[str]
    retrieved_chunks: List[dict]
    response: str
    citations: List[Dict[str, Any]]
    safety_passed: bool
    safety_message: str
    needs_expansion: bool
    retrieval_method: str


def build_rag_agent(
    hybrid_retriever: HybridRetriever,
    query_expander: QueryExpander,
    llm_client: LLMClient,
    safety_guard: SafetyGuard,
):
    """Build and return a compiled LangGraph RAG agent."""

    def safety_check(state: AgentState) -> AgentState:
        """Validate the input query for safety."""
        is_safe, message = safety_guard.check(state["query"])
        return {
            **state,
            "safety_passed": is_safe,
            "safety_message": message,
        }

    def should_proceed(state: AgentState) -> Literal["retrieve", "refuse"]:
        """Route based on safety check."""
        if state["safety_passed"]:
            return "retrieve"
        return "refuse"

    def refuse(state: AgentState) -> AgentState:
        """Return safety refusal message."""
        return {
            **state,
            "response": state["safety_message"],
            "citations": [],
        }

    def retrieve(state: AgentState) -> AgentState:
        """Run hybrid retrieval."""
        results = hybrid_retriever.retrieve(state["query"], top_k=CONFIG.retrieval.top_k)

        # Check if retrieval quality is sufficient
        needs_expansion = False
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            if avg_score < 0.01:  # RRF scores are typically small
                needs_expansion = True
        else:
            needs_expansion = True

        chunks_data = [
            {
                "id": r.chunk.id,
                "text": r.chunk.text,
                "metadata": r.chunk.metadata,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
            }
            for r in results
        ]

        return {
            **state,
            "retrieved_chunks": chunks_data,
            "needs_expansion": needs_expansion,
            "retrieval_method": "hybrid",
        }

    def should_expand(state: AgentState) -> Literal["expand", "generate"]:
        """Decide if query expansion is needed."""
        if state["needs_expansion"] and CONFIG.retrieval.hyde_enabled:
            return "expand"
        return "generate"

    def expand_query(state: AgentState) -> AgentState:
        """Expand query using HyDE or MultiQuery."""
        logger.info("Expanding query with HyDE...")
        results = query_expander.hyde_retrieve(state["query"], top_k=CONFIG.retrieval.top_k)

        chunks_data = [
            {
                "id": r.chunk.id,
                "text": r.chunk.text,
                "metadata": r.chunk.metadata,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
            }
            for r in results
        ]

        return {
            **state,
            "retrieved_chunks": chunks_data,
            "retrieval_method": "hyde",
        }

    def generate(state: AgentState) -> AgentState:
        """Generate response using retrieved context."""
        results = [
            RetrievalResult(
                chunk=__import__("backend.utils.datatypes", fromlist=["DocumentChunk"]).DocumentChunk(
                    id=c["id"],
                    text=c["text"],
                    metadata=c["metadata"],
                ),
                score=c["score"],
                retrieval_method=c.get("retrieval_method", "hybrid"),
            )
            for c in state["retrieved_chunks"]
        ]

        response = llm_client.generate(state["query"], results)
        response = safety_guard.sanitize_output(response)

        citations = [
            {
                "document": c["metadata"].get("document_name", "Unknown"),
                "page": c["metadata"].get("page_number", 0),
                "section": c["metadata"].get("section_title", ""),
                "score": round(c["score"], 4),
                "retrieval_method": c.get("retrieval_method", ""),
                "text_preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            }
            for c in state["retrieved_chunks"]
        ]

        return {
            **state,
            "response": response,
            "citations": citations,
        }

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("safety_check", safety_check)
    workflow.add_node("refuse", refuse)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("expand_query", expand_query)
    workflow.add_node("generate", generate)

    # Set entry point
    workflow.set_entry_point("safety_check")

    # Add edges
    workflow.add_conditional_edges(
        "safety_check",
        should_proceed,
        {"retrieve": "retrieve", "refuse": "refuse"},
    )
    workflow.add_conditional_edges(
        "retrieve",
        should_expand,
        {"expand": "expand_query", "generate": "generate"},
    )
    workflow.add_edge("expand_query", "generate")
    workflow.add_edge("refuse", END)
    workflow.add_edge("generate", END)

    return workflow.compile()


class RAGAgent:
    """High-level interface to the LangGraph RAG agent."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        query_expander: QueryExpander,
        llm_client: LLMClient,
    ):
        self.llm_client = llm_client
        self.safety_guard = SafetyGuard()
        self.hybrid_retriever = hybrid_retriever
        self.query_expander = query_expander
        self.agent = build_rag_agent(
            hybrid_retriever, query_expander, llm_client, self.safety_guard
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the full agent pipeline."""
        initial_state: AgentState = {
            "query": question,
            "expanded_queries": [],
            "retrieved_chunks": [],
            "response": "",
            "citations": [],
            "safety_passed": True,
            "safety_message": "",
            "needs_expansion": False,
            "retrieval_method": "",
        }

        result = self.agent.invoke(initial_state)

        return {
            "answer": result.get("response", ""),
            "citations": result.get("citations", []),
            "safety_passed": result.get("safety_passed", True),
            "retrieval_method": result.get("retrieval_method", ""),
        }

    def query_stream(self, question: str):
        """
        Run safety + retrieval, then stream the generation.
        Returns (citations, generator) or (citations, error_string).
        """
        is_safe, message = self.safety_guard.check(question)
        if not is_safe:
            return [], message

        # Retrieve
        results = self.hybrid_retriever.retrieve(question, top_k=CONFIG.retrieval.top_k)

        # Check if expansion needed
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            if avg_score < 0.01 and CONFIG.retrieval.hyde_enabled:
                logger.info("Low retrieval scores, expanding with HyDE...")
                results = self.query_expander.hyde_retrieve(question, top_k=CONFIG.retrieval.top_k)

        citations = [
            {
                "document": r.chunk.metadata.get("document_name", "Unknown"),
                "page": r.chunk.metadata.get("page_number", 0),
                "section": r.chunk.metadata.get("section_title", ""),
                "score": round(r.score, 4),
                "retrieval_method": r.retrieval_method,
                "text_preview": r.chunk.text[:200] + "..." if len(r.chunk.text) > 200 else r.chunk.text,
            }
            for r in results
        ]

        # Return streaming generator
        def stream():
            for token in self.llm_client.generate_stream(question, results):
                sanitized = self.safety_guard.sanitize_output(token)
                yield sanitized

        return citations, stream()
