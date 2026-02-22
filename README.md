# Multimodal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system supporting both text and image documents, utilizing local LLMs via Ollama, open-source embeddings, and FAISS.

## Features
- **Multimodal Ingestion**: Load and process PDFs, TXTs, and Images (PNG/JPG).
- **Hybrid Search Ready**: Extensible architecture base.
- **Local Inference**: Uses `Ollama` for generation (e.g., Llama 3, Mistral).
- **Advanced Retrieval**: Uses `SentenceTransformers` for text, `CLIP` for images, and `CrossEncoder` for re-ranking.
- **Optional Gemini Integration**: Modular support for Google Gemini API as a final reasoning step.
- **Fast Vector Storage**: Powered by FAISS.

## Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com/) installed and running locally.
3. Pull an Ollama model, e.g., `ollama pull mistral` or `ollama pull llama3`.

## Installation

1. Clone or navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (if using Gemini):
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

## Usage

### Ingesting Data
Place your PDF, text, and image files in a directory (e.g., `data/`). Then run:

```bash
python main.py --ingest data/
```

### Querying
Ask questions across your text and image context:

```bash
python main.py --query "What does the architecture diagram in the PDF show?"
```

To enable Gemini reasoning:
```bash
python main.py --query "Summarize the findings" --use-gemini
```
