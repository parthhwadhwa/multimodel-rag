from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from utils.logger import logger
from embedding.text_embedder import HealthcareEmbedder
from vector_store.faiss_store import HealthcareVectorStore
from retrieval.retriever import HealthcareRetriever
from generation.ollama_client import HealthcareOllamaClient
from generation.gemini_client import HealthcareGeminiClient

app = FastAPI(title="MediRAG API", description="Domain-Specific Healthcare Drug Information Assistant")

# Configure CORS for Next.js frontend (default port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons for efficiency
try:
    logger.info("Initializing persistent API models...")
    embedder = HealthcareEmbedder()
    vstore = HealthcareVectorStore()
    retriever = HealthcareRetriever(vstore, embedder)
    ollama_client = HealthcareOllamaClient()
    gemini_client = HealthcareGeminiClient()
except Exception as e:
    logger.error(f"Failed to initialize models on startup: {e}")

class QueryRequest(BaseModel):
    question: str
    model: str = "ollama"  # "ollama" or "gemini"

class QueryResponse(BaseModel):
    answer: str
    model_used: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        results = retriever.retrieve(req.question)
        
        # Select appropriate client
        client = gemini_client if req.model == "gemini" else ollama_client
        
        # We accumulate the streamed chunks for a single JSON response
        # Future enhancement: stream directly via FastAPI StreamingResponse
        response_text = ""
        for chunk_text in client.generate_stream(req.question, results):
            response_text += chunk_text
            
        return QueryResponse(answer=response_text, model_used=req.model)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
