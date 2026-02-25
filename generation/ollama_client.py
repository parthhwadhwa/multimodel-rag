import requests
import json
from typing import Generator, List
from utils.logger import logger
from utils.datatypes import QueryResult

class HealthcareOllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "mistral"
        self.temperature = 0.2
        self.top_p = 0.8
        self.max_tokens = 400

    def build_prompt(self, query: str, context_results: List[QueryResult]) -> str:
        context_texts = []
        for res in context_results:
            context_texts.append(res.chunk.text_content)
                
        context_str = "\n\n".join(context_texts)
        
        prompt = f"""System: Provide general educational information only. Avoid diagnosis. Avoid personalized dosage advice. If unsure, respond: "The information is not available in the provided documents." Recommend consulting healthcare professionals.

Context:
{context_str}

Question:
{query}

Answer:"""
        return prompt

    def generate_stream(self, query: str, context_results: List[QueryResult]) -> Generator[str, None, None]:
        prompt = self.build_prompt(query, context_results)
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"\n[Error connecting to Ollama: {e}]"
