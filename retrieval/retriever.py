from typing import List
from sentence_transformers import CrossEncoder

from utils.config import CONFIG
from utils.logger import logger
from utils.datatypes import QueryResult, Modality
from embedding.text_embedder import TextEmbedder
from embedding.image_embedder import ImageEmbedder
from vector_store.faiss_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore, text_embedder: TextEmbedder, image_embedder: ImageEmbedder):
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        
        self.use_reranker = CONFIG.retrieval.use_reranker
        if self.use_reranker:
            model_name = CONFIG.retrieval.reranker_model
            logger.info(f"Loading CrossEncoder reranker: {model_name}")
            try:
                self.reranker = CrossEncoder(model_name)
            except Exception as e:
                logger.error(f"Failed to load reranker {model_name}: {e}")
                self.use_reranker = False

    def retrieve(self, query: str, top_k: int = None, modalities: List[Modality] = None) -> List[QueryResult]:
        if top_k is None:
            top_k = CONFIG.retrieval.top_k
            
        if modalities is None:
            modalities = [Modality.TEXT, Modality.IMAGE]
            
        logger.info(f"Retrieving for query: '{query}' with top_k={top_k}, modalities={modalities}")
        
        all_results = []
        

        if Modality.TEXT in modalities:
            try:
                text_query_embedding = self.text_embedder.embed_text(query)

                fetch_k = top_k * 3 if self.use_reranker else top_k
                text_results = self.vector_store.search_text(text_query_embedding, top_k=fetch_k)
                
                if self.use_reranker and text_results:
                    logger.info("Re-ranking text results...")
                    pairs = [[query, res.chunk.text_content] for res in text_results if res.chunk.text_content]
                    if pairs:
                        scores = self.reranker.predict(pairs)
                        for res, score in zip(text_results, scores):
                            res.score = float(score)
                        

                        text_results.sort(key=lambda x: x.score, reverse=True)
                else:

                    text_results.sort(key=lambda x: x.score, reverse=False)
                    
                all_results.extend(text_results[:top_k])
            except Exception as e:
                logger.error(f"Text retrieval failed: {e}")


        if Modality.IMAGE in modalities:
            try:
                img_query_embedding = self.image_embedder.embed_text_query(query)
                image_results = self.vector_store.search_image(img_query_embedding, top_k=top_k)

                image_results.sort(key=lambda x: x.score, reverse=False)
                all_results.extend(image_results)
            except Exception as e:
                logger.error(f"Image retrieval failed: {e}")
                
        return all_results
