# RAG-Based Agent System

An end-to-end Retrieval-Augmented Generation (RAG) microservices demo that ingests documents, indexes them into a vector store, and serves LLM-powered question answering via a composable HTTP API.



##  Repository Structure
```text .
├── services/ │ ├── ingest/ # Document ingestion & chunking service │ │ ├── Dockerfile │ │ ├── main.py │ │ └── requirements.txt │ ├── embed_indexing/ # Embeddings & FAISS indexing service │ │ ├── Dockerfile │ │ ├── main.py │ │ └── requirements.txt │ ├── query/ # Query router & prompt builder service │ │ ├── Dockerfile │ │ ├── main.py │ │ └── requirements.txt │ └── llm/ # LLM API wrapper service │ ├── Dockerfile │ ├── main.py │ └── requirements.txt ├── .env # Environment variables (API keys, URLs) ├── docker-compose.yml # Orchestrate all 4 services └── README.md # ← You are here

 ```
 Architecture Overview

1. **Ingestion Service** (`/ingest`)  
   - Accepts raw documents (Markdown or plain text) via HTTP POST.  
   - Chunks text into smaller passages and forwards to embedding service.

2. **Embedding & Indexing Service** (`/embed`)  
   - Receives text chunks, computes vector embeddings.  
   - Stores embeddings in FAISS for similarity search.

3. **Query Service** (`/query`)  
   - Receives user query, retrieves top-K similar chunks from FAISS.  
   - Builds a prompt with system message + context + question.

4. **LLM Service** (`/generate`)  
   - Sends prompt to OpenAI (or HuggingFace) and streams/returns the generated answer.

All services communicate over RESTful endpoints in isolated Docker containers.

---

## Prerequisites

- Python 3.8+  
- Docker & Docker Compose (v2 recommended)  
- OpenAI API key (if using OpenAI backend)



##  Setup & Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/saikrishnapotluri/RAG-Based-Agent-System.git
   cd RAG-Based-Agent-System
Configure environment
Copy .env.example to .env and fill in your keys:
OPENAI_API_KEY=your_openai_key
Docker Compose
Build and start all services:
docker-compose up --build

Verify endpoints

Ingestion: POST http://localhost:8001/ingest

Embedding: POST http://localhost:8002/embed

Query: POST http://localhost:8003/query

LLM: POST http://localhost:8004/generate

API Usage
1. Ingestion Service
Endpoint: POST /ingest

Body:
{
  "doc_id": "article-123",
  "content": "# Title\nYour markdown text..."
}


Response:
{ "status": "ok", "chunks": 12 }

Embedding Service
Endpoint: POST /embed

Body:
{
  "doc_id": "article-123",
  "chunks": [
    "Text chunk 1 …",
    "Text chunk 2 …"
  ]
}
Response:
{ "status": "indexed", "vectors": 12 }

Query Service
Endpoint: POST /query

Body:{
  "query": "What is RAG?",
  "top_k": 5
}

Response:
{
  "context": [
    "Relevant chunk A…",
    "Relevant chunk B…"
  ],
  "prompt": "…constructed prompt…"
}

LLM Service
Endpoint: POST /generate

Body:
{
  "prompt": "…constructed prompt…"
}

Response (streaming):
{ "token": "The answer is …" }

