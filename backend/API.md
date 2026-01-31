# Nom-sense API Documentation

Base URL: `http://localhost:7860` (or `http://localhost:8000` depending on configuration)

## Endpoints

### 1. Health Check
Checks if the API is running.

- **URL:** `/health`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Root
Basic connectivity check.

- **URL:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "ok"
  }
  ```

### 3. Ask Question
The main RAG endpoint. Accepts a natural language question and returns an answer with citations.

- **URL:** `/ask`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "question": "Chứng tích sớm nhất về chữ Nôm là gì?",
    "top_k": 10,             // Optional: Number of chunks to rerank (default: 4)
    "pool_size": 25,         // Optional: Number of chunks to retrieve initially (default: 25)
    "temperature": 0.0,      // Optional: LLM creativity (default: 0.0)
    "rerank": true           // Optional: Enable/disable reranking (default: true)
  }
  ```

- **Success Response (200 OK):**
  ```json
  {
    "answer": "Chứng tích sớm nhất về chữ Nôm là các bia đá thời Lý...",
    "citations": [
      "Khái luận văn tự học Chữ Nôm – Chương 1 - p.15"
    ],
    "sources": [
      {
        "label": "Khái luận văn tự học Chữ Nôm – Chương 1 - p.15",
        "page_number": 15,
        "chapter": "Chương 1",
        "book_title": "Khái luận văn tự học Chữ Nôm",
        "file_name": "Book1.pdf",
        "text": "..."
      }
    ]
  }
  ```

- **Error Response (422 Unprocessable Entity):**
  - Validation error if request body is invalid.
