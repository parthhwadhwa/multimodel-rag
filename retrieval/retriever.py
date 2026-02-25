import re
from typing import List
from vector_store.faiss_store import HealthcareVectorStore
from embedding.text_embedder import HealthcareEmbedder
from utils.logger import logger
from utils.datatypes import QueryResult

class HealthcareRetriever:
    def __init__(self, vector_store: HealthcareVectorStore, embedder: HealthcareEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> List[QueryResult]:
        """
        Retrieves top chunks (default 3 to keep under 1200 tokens).
        Applies score boosting if query contains specific keywords.
        """
        logger.info(f"Retrieving healthcare info for query: '{query}'")
        
        query_embedding = self.embedder.embed_text(query)
        initial_k = top_k * 2
        results = self.vector_store.search(query_embedding, top_k=initial_k)
        
        boost_keywords = {
            "side effect": ["common_side_effects", "serious_side_effects"],
            "adverse": ["common_side_effects", "serious_side_effects", "warnings"],
            "contraindication": ["contraindications"],
            "warning": ["warnings"],
            "dosage": ["dosage_info"],
            "dose": ["dosage_info"],
            "how much": ["dosage_info"],
            "use": ["uses"],
            "treat": ["uses"]
        }
        
        query_lower = query.lower()
        active_boost_sections = set()
        for kw, sections in boost_keywords.items():
            if re.search(r'\b' + re.escape(kw) + r's?\b', query_lower):
                active_boost_sections.update(sections)
                
        if active_boost_sections:
            logger.info(f"Applying keyword boost for sections: {active_boost_sections}")
            for res in results:
                chunk_section = res.chunk.metadata.get("section", "")
                if chunk_section in active_boost_sections:
                    res.score += 0.2  # Boost exact section matches
                    
        # Sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        final_results = results[:top_k]
        
        total_words = sum(len((res.chunk.text_content or "").split()) for res in final_results)
        logger.info(f"Retrieved {len(final_results)} chunks (approx {int(total_words * 1.3)} tokens total space)")
        
        return final_results
