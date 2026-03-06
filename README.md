# DocIntel — Document Intelligence System

A production-grade RAG system for medical document search and analysis. Processes PDF documents through a complete ingestion pipeline, indexes them in ChromaDB with hybrid retrieval (dense vectors + BM25 + RRF), and generates grounded answers using Phi-3 Mini via Ollama.

## Architecture

```
┌─────────────┐    ┌──────────────────────────────────────────────┐
│   React UI  │───▶│  FastAPI Backend                             │
│  (Vite)     │    │  ┌─────────┐  ┌────────┐  ┌──────────────┐  │
│  • Chat     │    │  │ Safety  │─▶│ Agent  │─▶│  Phi-3 Mini  │  │
│  • Upload   │    │  │ Guard   │  │(Graph) │  │  (Ollama)    │  │
│  • Sources  │    │  └─────────┘  └───┬────┘  └──────────────┘  │
│  • Model    │    │                   │                          │
│    Info     │    │  ┌────────────────▼──────────────────────┐   │
└─────────────┘    │  │     Hybrid Retriever                  │   │
                   │  │  ┌─────────┐  ┌───────┐  ┌────────┐  │   │
                   │  │  │ Dense   │  │ BM25  │  │  RRF   │  │   │
                   │  │  │ Search  │  │Search │  │ Fusion │  │   │
                   │  │  └────┬────┘  └───┬───┘  └────────┘  │   │
                   │  └───────┼───────────┼──────────────────┘   │
                   │          │           │                       │
                   │  ┌───────▼───────────▼──────────────────┐   │
                   │  │  ChromaDB + all-MiniLM-L6-v2         │   │
                   │  └──────────────────────────────────────┘   │
                   └──────────────────────────────────────────────┘
```

## Features

| Component | Details |
|-----------|---------|
| **PDF Ingestion** | PyMuPDF extraction → preprocessing → structure detection |
| **Chunking** | 5 strategies: recursive, token, markdown, semantic, parent-child |
| **Embeddings** | `all-MiniLM-L6-v2` (384 dims) via sentence-transformers |
| **Vector DB** | ChromaDB with cosine similarity |
| **Retrieval** | Dense + BM25 hybrid with Reciprocal Rank Fusion |
| **Query Expansion** | HyDE + MultiQuery via LLM |
| **LLM** | Phi-3 Mini (3.8B) via Ollama, temperature 0.2 |
| **Agent** | LangGraph state machine: safety → retrieve → [expand] → generate |
| **Security** | Jailbreak detection, prompt injection prevention |
| **Evaluation** | Precision@k, Recall@k, MRR, context relevance |
| **Frontend** | React (Vite) with SSE streaming, citations, document upload |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed

### 1. Pull the LLM model

```bash
ollama pull phi3:mini
```

### 2. Install backend dependencies

```bash
cd multimodal_rag
pip install -r requirements.txt
```

### 3. Convert drug data to PDFs

```bash
python scripts/convert_json_to_pdf.py
```

### 4. Start the backend

```bash
python -m backend.api
```

The API starts at `http://localhost:8000`. Visit `/docs` for the Swagger UI.

### 5. Ingest documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"chunking_strategy": "recursive"}'
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Project Structure

```
multimodal_rag/
├── backend/
│   ├── config.py              # Central configuration
│   ├── datatypes.py           # Data models
│   ├── logger.py              # Logging
│   ├── api.py                 # FastAPI application
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PyMuPDF PDF extraction
│   │   ├── preprocessor.py    # Text cleaning & formatting
│   │   └── structure_detector.py  # Section detection
│   ├── chunking/
│   │   ├── chunking_manager.py    # Strategy factory
│   │   ├── recursive_chunker.py
│   │   ├── token_chunker.py
│   │   ├── markdown_chunker.py
│   │   ├── semantic_chunker.py
│   │   └── parent_child_chunker.py
│   ├── embeddings/
│   │   └── embeddings.py      # all-MiniLM-L6-v2
│   ├── vectorstore/
│   │   └── chroma_store.py    # ChromaDB persistent store
│   ├── retrieval/
│   │   ├── dense_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py    # RRF fusion
│   │   └── query_expander.py      # HyDE + MultiQuery
│   ├── agents/
│   │   ├── llm_client.py      # Phi-3 Mini via Ollama
│   │   ├── safety_guard.py    # Jailbreak prevention
│   │   └── rag_agent.py       # LangGraph agent
│   └── evaluation/
│       └── evaluator.py       # Metrics pipeline
├── scripts/
│   └── convert_json_to_pdf.py # JSON → PDF converter
├── documents/                 # PDF knowledge base
├── data/                      # Original JSON drug files
├── vector_store/              # ChromaDB persistence
├── frontend/                  # React (Vite) app
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── ChatArea.jsx
│   │       ├── InputBar.jsx
│   │       └── Sidebar.jsx
│   └── package.json
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/query` | Query (non-streaming) |
| `POST` | `/query/stream` | Query with SSE streaming |
| `POST` | `/upload` | Upload PDF (auto-ingested) |
| `POST` | `/ingest` | Batch ingest all PDFs |
| `GET` | `/documents` | List documents |
| `GET` | `/model-info` | Model & system config |

## Configuration

Key settings in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `ollama.model` | `phi3:mini` | LLM model |
| `ollama.temperature` | `0.2` | Generation temperature |
| `embedding.model_name` | `all-MiniLM-L6-v2` | Embedding model |
| `chunking.default_strategy` | `recursive` | Default chunking |
| `chunking.chunk_size` | `512` | Chunk size |
| `retrieval.top_k` | `5` | Results per query |
| `retrieval.hyde_enabled` | `true` | Enable HyDE expansion |

## Environment Variables

Create a `.env` file:

```env
# Optional: LangSmith tracing
LANGSMITH_API_KEY=your_key
LANGSMITH_TRACING=true
```
