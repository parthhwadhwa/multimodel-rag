# Domain-Specific Healthcare RAG

A lightweight, memory-optimized Retrieval-Augmented Generation application strictly tuned to analyze drug and medication schema structures on restricted hardware (like an 8GB RAM MacBook Air).

## Features
- **Strict Pydantic JSON Ingestion**: Parses highly structured medication schemas.
- **Section-Based Chunking**: Minimizes token overlap and forces data context explicitly (no sliding window token waste).
- **CPU Optimized**: Forces CPU processing and limits vector batch sizes to prevent memory-spikes.
- **Local Inference**: Uses `Ollama` for reasoning with generation restricted to strict safety boundaries.
- **Fast Vector Storage**: Powered by FAISS with a lazy-loading IndexFlatIP index.

## Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com/) installed and running locally.
3. Pull the specific Mistral model: `ollama pull mistral`.

## Installation

1. Clone or navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `pydantic` installed)*

## Usage

### Ingesting Data
Place your structured `.json` medical drug files directly into the `data/` directory. Then run:

```bash
python main.py --ingest data/
```

### Querying System

You can run an interactive query session to chat against the ingested medications:

```bash
python main.py --interactive
```
