"""
FastAPI Application — Document Intelligence System API.
Endpoints: query (streaming), upload, ingest, documents, health.
"""
import os
import json
import shutil
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import CONFIG, DOCUMENTS_DIR, VECTOR_STORE_DIR
from backend.logger import logger
from backend.datatypes import ChunkingStrategy

# Pipeline imports
from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.preprocessor import TextPreprocessor
from backend.ingestion.structure_detector import StructureDetector
from backend.chunking.chunking_manager import ChunkingManager
from backend.embeddings.embeddings import EmbeddingEngine
from backend.vectorstore.chroma_store import ChromaStore
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.query_expander import QueryExpander
from backend.agents.llm_client import LLMClient
from backend.agents.rag_agent import RAGAgent
from backend.evaluation.evaluator import RAGEvaluator

# --- App ---
app = FastAPI(
    title="Document Intelligence System",
    description="Production-grade RAG system with PDF ingestion, hybrid retrieval, and Phi-3 Mini",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singleton Components ---
pdf_loader = PDFLoader()
preprocessor = TextPreprocessor()
structure_detector = StructureDetector()
embedding_engine = EmbeddingEngine()
chroma_store = ChromaStore()
chunking_manager = ChunkingManager(embedding_model=embedding_engine.model)
dense_retriever = DenseRetriever(chroma_store, embedding_engine)
bm25_retriever = BM25Retriever(chroma_store)
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
query_expander = QueryExpander(hybrid_retriever)
llm_client = LLMClient()
rag_agent = RAGAgent(hybrid_retriever, query_expander, llm_client)
evaluator = RAGEvaluator()

logger.info("All components initialized successfully")


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str
    chunking_strategy: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list
    model: str
    retrieval_method: str


class IngestRequest(BaseModel):
    directory: Optional[str] = None
    chunking_strategy: str = "recursive"


class DocumentInfo(BaseModel):
    name: str
    path: str
    size_kb: float


# --- Endpoints ---
@app.get("/health")
async def health_check():
    """System health check."""
    try:
        doc_count = chroma_store.count()
        model_info = llm_client.get_model_info()
        return {
            "status": "healthy",
            "documents_indexed": doc_count,
            "model": model_info,
            "embedding_model": CONFIG.embedding.model_name,
            "vector_store": "ChromaDB",
            "chunking_strategies": ChunkingManager.available_strategies(),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """Query the RAG system (non-streaming)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_agent.query(req.question)
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            model=CONFIG.ollama.model,
            retrieval_method=result.get("retrieval_method", "hybrid"),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    """Query with streaming response (SSE)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        citations, stream_or_error = rag_agent.query_stream(req.question)

        if isinstance(stream_or_error, str):
            # Safety refusal
            async def error_stream():
                yield f"data: {json.dumps({'type': 'citations', 'data': []})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': stream_or_error})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        async def event_stream():
            # Send citations first
            yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
            # Stream tokens
            for token in stream_or_error:
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Stream query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(str(DOCUMENTS_DIR), exist_ok=True)
    file_path = os.path.join(str(DOCUMENTS_DIR), file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Auto-ingest the uploaded file
        pages = pdf_loader.load(file_path)
        pages = preprocessor.process(pages)
        sections = structure_detector.detect_sections(pages)
        chunks = chunking_manager.chunk_sections(
            sections, document_name=file.filename
        )

        if chunks:
            embeddings = embedding_engine.embed_texts([c.text for c in chunks])
            chroma_store.add_documents(chunks, embeddings)
            bm25_retriever.build_index()  # Rebuild BM25 index

        return {
            "message": f"Uploaded and ingested {file.filename}",
            "pages": len(pages),
            "chunks": len(chunks),
            "file_path": file_path,
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_documents(req: IngestRequest):
    """Ingest all PDFs from the documents directory."""
    doc_dir = req.directory or str(DOCUMENTS_DIR)

    if not os.path.isdir(doc_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {doc_dir}")

    try:
        strategy = ChunkingStrategy(req.chunking_strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunking strategy. Choose from: {ChunkingManager.available_strategies()}",
        )

    try:
        # Clear existing data
        chroma_store.delete_collection()

        pages = pdf_loader.load_directory(doc_dir)
        pages = preprocessor.process(pages)
        sections = structure_detector.detect_sections(pages)

        all_chunks = []
        # Group sections by document for proper naming
        doc_sections = {}
        for section in sections:
            doc_name = section.title  # Will be overridden by page metadata
            doc_sections.setdefault("all", []).append(section)

        # Process pages to get document names
        page_doc_map = {}
        for page in pages:
            doc_name = page.metadata.get("document_name", "unknown")
            if doc_name not in page_doc_map:
                page_doc_map[doc_name] = []

        # Chunk all sections
        chunks = chunking_manager.chunk_sections(
            sections, document_name="drug_knowledge_base", strategy=strategy
        )

        # Re-attribute document names from page metadata
        for chunk in chunks:
            if "document_name" not in chunk.metadata or chunk.metadata["document_name"] == "drug_knowledge_base":
                # Try to find from sections
                for page in pages:
                    page_doc_name = page.metadata.get("document_name", "")
                    if page_doc_name and chunk.text[:50] in page.text:
                        chunk.metadata["document_name"] = page_doc_name
                        break

        if chunks:
            embeddings = embedding_engine.embed_texts([c.text for c in chunks])
            chroma_store.add_documents(chunks, embeddings)
            bm25_retriever.build_index()

        return {
            "message": "Ingestion complete",
            "documents_processed": len(page_doc_map),
            "pages_loaded": len(pages),
            "sections_detected": len(sections),
            "chunks_created": len(chunks),
            "chunking_strategy": strategy.value,
            "vector_store_count": chroma_store.count(),
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the documents directory."""
    os.makedirs(str(DOCUMENTS_DIR), exist_ok=True)
    docs = []
    for f in sorted(os.listdir(str(DOCUMENTS_DIR))):
        if f.lower().endswith(".pdf"):
            path = os.path.join(str(DOCUMENTS_DIR), f)
            size = os.path.getsize(path) / 1024
            docs.append(DocumentInfo(name=f, path=path, size_kb=round(size, 1)))

    indexed_docs = chroma_store.list_documents()
    return {
        "documents": [d.model_dump() for d in docs],
        "indexed_documents": indexed_docs,
        "total_chunks": chroma_store.count(),
    }


@app.get("/model-info")
async def model_info():
    """Get model and system configuration."""
    return {
        "llm": llm_client.get_model_info(),
        "embedding": {
            "model": CONFIG.embedding.model_name,
            "dimension": CONFIG.embedding.dimension,
        },
        "chunking": {
            "default_strategy": CONFIG.chunking.default_strategy,
            "available": ChunkingManager.available_strategies(),
            "chunk_size": CONFIG.chunking.chunk_size,
        },
        "retrieval": {
            "top_k": CONFIG.retrieval.top_k,
            "hyde_enabled": CONFIG.retrieval.hyde_enabled,
            "rrf_k": CONFIG.retrieval.rrf_k,
        },
        "vector_store": {
            "type": "ChromaDB",
            "total_documents": chroma_store.count(),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
