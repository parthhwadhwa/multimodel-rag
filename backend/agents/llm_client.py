"""
LLM Client — Phi-3 Mini (3.8B) via Ollama with streaming support.
"""
import json
from typing import Generator, List, Dict, Any

import requests

from backend.utils.datatypes import RetrievalResult
from backend.utils.config import CONFIG
from backend.utils.logger import logger


SYSTEM_PROMPT = """You are a medical information assistant.

Use ONLY the provided context to answer the question.

Rules:

* Do not invent information.
* Do NOT include citation markers like (Source 1), [1], or similar.
* Do NOT list sources inside the answer.
* The system will display citations separately.
* Write a clear and concise medical explanation.
* If the answer cannot be found in the context, say so.

Return ONLY the answer text."""


class LLMClient:
    """Phi-3 Mini LLM client via Ollama with streaming and RAG prompt assembly."""

    def __init__(self):
        self.base_url = CONFIG.ollama.base_url
        self.model = CONFIG.ollama.model
        self.temperature = CONFIG.ollama.temperature
        self.top_p = CONFIG.ollama.top_p
        self.max_tokens = CONFIG.ollama.max_tokens

    def build_rag_prompt(
        self, query: str, results: List[RetrievalResult]
    ) -> str:
        """Build a RAG prompt with context and citations."""
        context_blocks = []
        for i, result in enumerate(results, 1):
            doc = result.chunk.metadata.get("document_name", "Unknown")
            page = result.chunk.metadata.get("page_number", "?")
            section = result.chunk.metadata.get("section_title", "General")
            method = result.retrieval_method
            score = f"{result.score:.3f}"

            context_blocks.append(
                f"[Source {i}] Document: {doc} | Page: {page} | Section: {section} | "
                f"Retrieval: {method} | Relevance: {score}\n"
                f"{result.chunk.text}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER:"""
        return prompt

    def _calculate_confidence_and_sources(self, results: List[RetrievalResult]) -> str:
        """Calculate confidence score and extract unique sources."""
        if not results:
            return "\n\nConfidence: 0.00\n\nSources:\nNone"
            
        avg_score = sum(r.score for r in results) / len(results)
        
        seen = set()
        sources = []
        for r in results:
            doc = r.chunk.metadata.get("document_name", "Unknown")
            page = r.chunk.metadata.get("page_number", "?")
            src_str = f"* {doc} (page {page})"
            if src_str not in seen:
                seen.add(src_str)
                sources.append(src_str)
                
        sources_str = "\n".join(sources)
        return ""

    def generate(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate a complete response (non-streaming)."""
        prompt = self.build_rag_prompt(query, results)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            base_answer = response.json().get("response", "")
            return base_answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: Unable to generate response. {e}"

    def generate_stream(
        self, query: str, results: List[RetrievalResult]
    ) -> Generator[str, None, None]:
        """Stream response tokens for real-time display."""
        prompt = self.build_rag_prompt(query, results)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                    },
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"\n[Error: {e}]"

    def raw_generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Raw generation without RAG template (for HyDE/MultiQuery)."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_tokens,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Raw LLM call failed: {e}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "model": self.model,
                "family": data.get("details", {}).get("family", "unknown"),
                "parameter_size": data.get("details", {}).get("parameter_size", "unknown"),
                "quantization": data.get("details", {}).get("quantization_level", "unknown"),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        except Exception:
            return {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "status": "unable to fetch details",
            }
