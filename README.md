# Offline RAG Assistant

A fully local Retrieval-Augmented Generation application for chatting with uploaded PDFs on a LAN. It uses FastAPI, Streamlit, SQLite, LangChain, FAISS, BM25, JWT, bcrypt, and local Ollama models only.

## Runtime

This project is developed for Python 3.11.9 only.

```powershell
python --version
```

Expected output:

```text
Python 3.11.9
```

The project includes `.python-version` with `3.11.9`.

## Offline Model Setup

Install Ollama locally, then pull only these models:

```powershell
ollama pull phi3:mini
ollama pull nomic-embed-text
```

Generation uses:

```python
ChatOllama(model="phi3:mini")
```

Embeddings use:

```python
OllamaEmbeddings(model="nomic-embed-text")
```

No OpenAI, Gemini, Anthropic, Cohere, HuggingFace Inference, Azure OpenAI, AWS Bedrock, Vertex AI, Pinecone, Weaviate Cloud, Qdrant Cloud, LangSmith, SaaS platform, cloud model, or hosted vector database is used.

## Installation

Create and activate a virtual environment with Python 3.11.9.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and change `JWT_SECRET_KEY` to a local secret before production use.

## Run

Start Ollama on the host machine, then start the backend:

```powershell
uvicorn backend.main --host 0.0.0.0 --port 8000
```

Start the frontend:

```powershell
streamlit run frontend/app.py --server.address 0.0.0.0
```

Open the Streamlit app from another device on the same WiFi network:

```text
http://HOST_MACHINE_IP:8501
```

In the app Settings panel, set the backend API URL to:

```text
http://HOST_MACHINE_IP:8000
```

## Features

- Register, login, logout
- JWT protected backend routes
- bcrypt password hashing
- Upload, view, and delete PDFs
- Maximum 5 PDFs per user
- PDF parsing with `PyPDFLoader`
- Chunking with `chunk_size = 1200` and `chunk_overlap = 200`
- Local embeddings with `OllamaEmbeddings(model="nomic-embed-text")`
- Persisted FAISS indexes
- BM25 keyword retrieval
- Hybrid retrieval with `EnsembleRetriever`, FAISS `top_k = 5`, BM25 `top_k = 5`, and weights `[0.7, 0.3]`
- Deduplicated final context with 8 chunks maximum
- Multi-turn chat history in SQLite
- Answers with source citations
- LAN access for desktop, laptop, phone, and tablet browsers

## API

- `POST /register`
- `POST /login`
- `GET /me`
- `POST /upload`
- `GET /documents`
- `DELETE /documents/{id}`
- `POST /chat`
- `GET /chat/history`
- `GET /health`

## Data Storage

SQLite database:

```text
backend/data/app.db
```

Uploaded PDFs:

```text
backend/uploads
```

Persisted indexes:

```text
backend/indexes
```

## Tests

```powershell
pytest
```

The tests validate security helpers, upload validation, citation parsing, and RAG utility behavior. Full PDF ingestion and generation require local Ollama with the two required models available.
