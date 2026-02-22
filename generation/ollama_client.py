import requests
import json
from typing import Generator, List

from utils.config import CONFIG
from utils.logger import logger
from utils.datatypes import QueryResult, Modality

class OllamaClient:
    def __init__(self):
        self.base_url = CONFIG.generation.ollama_url
        self.model = CONFIG.generation.model
        self.temperature = CONFIG.generation.temperature

    def build_prompt(self, query: str, context_results: List[QueryResult]) -> str:
        context_texts = []
        image_paths = []
        
        for res in context_results:
            if res.chunk.modality == Modality.TEXT:
                context_texts.append(res.chunk.text_content)
            elif res.chunk.modality == Modality.IMAGE:
                image_paths.append(res.chunk.image_path)
                
        context_str = "\n\n".join(context_texts)
        
        prompt = f"""System:
You are a helpful research assistant. Answer the question based ONLY on the provided context. If the context does not contain enough information, say so. Do not make up information.

Context:
---------
{context_str}

---------
Images Referenced (for your awareness, if any): {', '.join(image_paths) if image_paths else 'None'}
---------

Question:
---------
{query}

Answer:"""
        return prompt

    def generate(self, query: str, context_results: List[QueryResult]) -> str:
        prompt = self.build_prompt(query, context_results)
        logger.info(f"Sending prompt to Ollama ({self.model})...")
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error connecting to Ollama: {e}"

    def generate_stream(self, query: str, context_results: List[QueryResult]) -> Generator[str, None, None]:
        prompt = self.build_prompt(query, context_results)
        logger.info(f"Streaming from Ollama ({self.model})...")
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature
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
