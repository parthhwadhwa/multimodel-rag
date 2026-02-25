import os
from typing import Generator, List
import google.generativeai as genai

from utils.logger import logger
from utils.datatypes import QueryResult
from dotenv import load_dotenv

class HealthcareGeminiClient:
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini generation may fail.")
            
        genai.configure(api_key=api_key)
        
        # Using gemini-2.5-flash for speed and lower cost in RAG
        self.model_name = "models/gemini-2.5-flash"
        
        # Strict constraints to mirror our memory-optimized design
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=400,
        )
        
        self.system_instruction = (
            "You are a helpful research assistant. Provide general educational information only. "
            "Avoid diagnosis. Avoid personalized dosage advice. If unsure, respond: "
            "'The information is not available in the provided documents.' "
            "Recommend consulting healthcare professionals."
        )
        
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction,
                generation_config=self.generation_config
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def build_prompt(self, query: str, context_results: List[QueryResult]) -> str:
        context_texts = []
        for res in context_results:
            context_texts.append(res.chunk.text_content)
                
        context_str = "\n\n".join(context_texts)
        
        prompt = f"""
Context:
{context_str}

Question:
{query}

Answer:"""
        return prompt

    def generate_stream(self, query: str, context_results: List[QueryResult]) -> Generator[str, None, None]:
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
