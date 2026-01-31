---
title: Nom-sense
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Nom-sense

## Overview
This project is an AI-powered retrieval-augmented generation (RAG) assistant designed for the Fulbright Hán-Nôm library collection. It allows users to ask questions in natural language and receive accurate answers grounded in the library's documents. The system leverages advanced vector search to find relevant book passages and uses a Large Language Model (LLM) to synthesize answers with precise page-level citations.

## Feature Highlights
- **Conversational Search:** Ask questions naturally in Vietnamese or English.
- **Accurate Citations:** Every answer includes direct links to the source book, chapter, and page number using a strict metadata schema.
- **Instant PDF Preview:** Users can click citations to view the exact page in the document viewer side-by-side with the chat.
- **Global Search:** Searches across the entire collection simultaneously without requiring namespace selection.
- **Full Cloud Architecture:** Uses Pinecone Serverless for vector storage, embedding generation (`multilingual-e5-large`), and reranking (`bge-reranker-v2-m3`), removing heavy local dependencies.
- **Standalone Ingestion:** Portable Python script for easy migrating and updating of the knowledge base.

## Tech Stack
- **Backend:** Python, Flask, Waitress
- **AI & RAG:**
    - **Vector Database:** Pinecone (Serverless)
    - **Embeddings:** Pinecone Inference (`multilingual-e5-large`)
    - **Reranking:** Pinecone Inference (`bge-reranker-v2-m3`)
    - **LLM:** OpenAI (`gpt-4o-mini` or configurable)
    - **Framework:** LangChain
- **Frontend:** React, Vite, TypeScript
- **Infrastructure:** Docker (optional)

## Technical Description

### System Architecture
The system consists of a **Flask** backend API and a **React** frontend SPA.

1.  **Ingestion Pipeline:**
    - Source PDFs in the `Word/` directory are processed by `scripts/ingest_pinecone.py`.
    - Documents are split into chunks.
    - Metadata (Book Title, Author "Nguyen Quang Hong", Chapter, Page Number) is rigorously extracted and normalized.
    - Chunks are embedded and stored in a **Pinecone** index.

2.  **Retrieval & Generation Flow (`/ask` endpoint):**
    - The backend receives a user question.
    - It generates an embedding for the question using Pinecone's API.
    - **Vector Search:** Queries Pinecone for the top related chunks (Global Search).
    - **Reranking:** The top results are re-ordered using Pinecone's Re-ranking model for higher precision.
    - **Synthesis:** The finalized context and question are sent to the OpenAI LLM to generate an answer.
    - The response containing the answer and citation metadata is sent back to the frontend.

3.  **Frontend Interaction:**
    - The React app sends requests to the Flask API.
    - It renders the answer with citation buttons.
    - Clicking a citation makes a request to serve the static PDF file, displaying it in the integrated viewer.

## Architecture Overview

```text
├── backend/            # Flask service, RAG pipeline, and ingestion
│   ├── app/            # Main application code
│   ├── config/         # Environment variables
│   ├── scripts/        # Ingestion scripts
│   ├── Word/           # Source PDF documents
│   ├── API.md          # API Documentation
│   └── requirements.txt
├── frontend/           # React + Vite SPA
│   ├── src/            # Components, hooks, styles
│   └── package.json
└── README.md           # Project documentation
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ (for frontend)
- Pinecone API Key (with Index created)
- OpenAI API Key

### Backend Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Nom-sense/backend
    ```

2.  **Install dependencies:**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\Activate.ps1
    # Mac/Linux:
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Copy the example configuration:
    ```bash
    cp config/.env.example config/.env
    # On Windows: copy config\.env.example config\.env
    ```
    Edit `config/.env` and add your keys:
    ```env
    OPENAI_API_KEY=sk-...
    PINECONE_API_KEY=pc-...
    PINECONE_INDEX_NAME=chat-nombot
    PINECONE_NAMESPACE=nom_sense
    DATA_DIR=Word
    ```

4.  **Ingest Data:**
    Run the standalone ingestion script to populate Pinecone:
    ```bash
    python scripts/ingest_pinecone.py
    ```

5.  **Run the Server:**
    ```bash
    # Development
    python -m app.main

    # Production
    waitress-serve --port=7860 app.main:app
    ```

### Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd ../frontend
    ```

2.  Install packages:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm run dev
    ```
    Access the app at `http://localhost:5173`.

## Example Usage

**Question:** "Quan niệm về chữ Nôm của tác giả là gì?"

**Response:**
> Theo tác giả, chữ Nôm không chỉ là phương tiện ghi âm tiếng Việt mà còn phản ánh tư duy văn hóa độc lập của dân tộc... [1]
>
> **Sources:**
> [1] Khái luận văn tự học Chữ Nôm – Chương 1 - p.15

*Clicking on [1] opens the PDF viewer to page 15 of "Khái luận văn tự học Chữ Nôm".*
