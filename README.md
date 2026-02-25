# MediRAG: Domain-Specific Medical Retrieval-Augmented Generation

## Problem Statement
General-purpose Large Language Models (LLMs) are prone to hallucination, which poses a severe risk in the healthcare and pharmacology domains. When users search for drug information, side effects, or contraindications, accuracy and context-grounding are critical. MediRAG addresses this by strictly grounding its generations in a verified, structured dataset of pharmaceutical information, ensuring that AI-driven responses are accurate, safe, and explicitly constrained from providing personalized medical advice or diagnoses.

## System Architecture

```text
[ User Query ] 
      │
      ▼
[ Embedding Model ] (all-MiniLM-L6-v2) ──► [ FAISS Vector Store ]
                                                  │
                                                  ▼
[ Context Retrieval ] ◄── (Top-3 relevant chunks with metadata)
      │
      ▼
[ Generation Guardrails ] (Safety Prompt Injection)
      │
      ▼
[ LLM Generation ] (Ollama Mistral 7B / Gemini)
      │
      ▼
[ Final Response ]
```

## Tech Stack
*   **Language:** Python
*   **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dimensional)
*   **Vector Database:** FAISS (`IndexFlatIP`, CPU-bound)
*   **LLM Inference:** Ollama (Mistral 7B quantized) / Google Gemini API (Optional)
*   **Data Models:** Pydantic

## Dataset Design
The system relies on a curated dataset of over 50 common drugs. Rather than ingesting unstructured text, the data is modeled using strict JSON schemas. 
Each drug record contains specific, isolated fields:
*   `uses`
*   `dosage_info`
*   `common_side_effects`
*   `serious_side_effects`
*   `contraindications`
*   `warnings`

## Retrieval & Generation Pipeline

### Section-Based Chunking
Instead of using fixed-token sliding windows which often split semantic context or merge irrelevant sections, the system employs **Section-Based Chunking**. Each specific field in the JSON schema (e.g., `contraindications` for Ibuprofen) acts as an isolated, self-contained document chunk. This ensures that the context retrieved is highly granular and semantically coherent.

### Metadata-Aware Retrieval
The retriever leverages the structural metadata of the chunks. It retrieves only the Top-3 most relevant context blocks to minimize noise and improve generation speed. Furthermore, keyword boosting is applied dynamically; queries containing terms like "side effects" or "adverse" will optionally boost the retrieval scores of chunks specifically labeled with the `serious_side_effects` or `common_side_effects` metadata.

## Safety & Guardrails
A core tenet of this system is safety. The LLM is strictly prompted to:
1. Provide general, educational information based **only** on the retrieved context.
2. Refuse requests for diagnosis or personalized medical advice.
3. Explicitly state when information is unavailable rather than attempting to guess or hallucinate.
4. Always recommend consulting a licensed healthcare professional.

## Performance Optimization
The architecture is designed for extreme efficiency and low resource footprint:
*   **CPU-Bound Embeddings:** Embedding generation is strictly enforced on the CPU using a lightweight model (`all-MiniLM-L6-v2`).
*   **Batched Inference:** Vector embeddings are processed in small, fixed batch sizes to prevent memory spiking during ingestion.
*   **Inner Product Search:** FAISS utilizes Inner Product (`IndexFlatIP`) coupled with normalized vectors to perform extremely fast cosine similarity searches without the overhead of heavy index structures.
*   **Token Caps:** The system strictly limits max output tokens and enforces low-temperature generation (`temperature=0.2`) to ensure deterministic and concise responses.

## Example Query & Output

**User Query:** "What are the common side effects of amoxicillin?"

**System Output:**
> Based on the provided clinical data, the common side effects of amoxicillin include nausea, vomiting, diarrhea, and mild skin rash. 
> 
> *Disclaimer: This information is for educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or qualified health provider with any questions you may have regarding a medical condition.*

## Installation & Run Instructions

### Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com/) installed and running locally.
3. Install the Mistral model: `ollama pull mistral`

### Setup
1. Clone the repository and navigate to the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. **Ingest the Dataset:**
   Parses the JSON schemas, generates embeddings, and builds the FAISS index.
   ```bash
   python main.py --ingest data/
   ```

2. **Run Interactive Query Mode:**
   Starts the interactive CLI session default to local Ollama.
   ```bash
   python main.py --interactive
   ```

   *(Optional) Run using Gemini for generation:*
   ```bash
   python main.py --interactive --use-gemini
   ```

3. **Run the Full Web Application (Frontend + Backend):**
   To experience the premium Apple-inspired Minimal UI, you must run both the API bridge and the Next.js frontend.

   **Terminal 1 (Start the Backend API):**
   ```bash
   python api.py
   ```

   **Terminal 2 (Start the Web Frontend):**
   ```bash
   cd frontend
   npm run dev
   ```
   Open `http://localhost:3000` in your browser to interact with the system.

## Future Improvements
*   **Frontend Interface:** A full Next.js web application is currently under development to provide a clean, accessible UI for healthcare professionals and users.
*   **Cloud Deployment:** Containerizing the backend pipeline (FastAPI/Docker) and migrating vector storage to a managed database (e.g., Pinecone or Postgres pgvector) for scalable cloud hosting.
*   **Expanded Knowledge Base:** Automating the data ingestion pipeline to continuously fetch and structure updated pharmaceutical data from authoritative sources (e.g., OpenFDA API).
