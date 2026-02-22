import google.generativeai as genai
from typing import Generator, List

from utils.config import CONFIG
from utils.logger import logger
from utils.datatypes import QueryResult, Modality

class GeminiClient:
    def __init__(self):
        self.api_key = CONFIG.gemini.api_key
        self.model_name = CONFIG.gemini.model
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini integration will fail.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def build_prompt(self, query: str, context_results: List[QueryResult]) -> str:
        context_texts = []
        for res in context_results:
            if res.chunk.modality == Modality.TEXT:
                context_texts.append(res.chunk.text_content)
                
        context_str = "\n\n".join(context_texts)
        
        prompt = f"""System:
You are a helpful research assistant. Answer the question based ONLY on the provided context. If the context does not contain enough information, say so. Do not make up information.
        
Context from Vector DB:
---------
{context_str}

---------
Original User Question:
---------
{query}

Answer:"""
        return prompt

    def generate(self, query: str, context_results: List[QueryResult]) -> str:
        if not self.api_key:
            return "Error: GEMINI_API_KEY is not configured."
            
        prompt = self.build_prompt(query, context_results)
        logger.info(f"Sending prompt to Gemini API ({self.model_name})...")
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error connecting to Gemini API: {e}"

    def generate_stream(self, query: str, context_results: List[QueryResult]) -> Generator[str, None, None]:
        if not self.api_key:
            yield "Error: GEMINI_API_KEY is not configured."
            return
            
        prompt = self.build_prompt(query, context_results)
        
        logger.info(f"Streaming from Gemini API ({self.model_name})...")
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield f"\n[Error connecting to Gemini API: {e}]"
