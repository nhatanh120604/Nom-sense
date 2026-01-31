"""
Main entry point for the Nom-sense Flask application.
Exposes endpoints for health checks and RAG-based Q&A.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import psutil

# Import internal modules
# Note: In Flask, dependencies are typically initialized globally or via a factory pattern.
# Global instances are maintained here for simplicity and parity with the previous setup.
from .rag.pipeline import RagService
from .schemas import AskRequest, AskResponse
from .settings import get_settings

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# Initialize Settings
settings = get_settings()

# Initialize Flask App
app = Flask(__name__)

# Setup CORS
CORS(app, resources={r"/*": {"origins": settings.allowed_origins or "*"}})

# Initialize RAG Service (Lazy loading or Startup)
# Initialize immediately to fail fast if configuration is invalid.
try:
    rag_service = RagService(settings)
    # Check connection/index validity if needed
    if settings.auto_ingest_on_startup:
        LOGGER.info("Auto-ingesting on startup...")
        rag_service.ensure_vectorstore(force_rebuild=False)
except Exception as e:
    LOGGER.error(f"Failed to initialize RAG service: {e}")
    # Do not exit; allow /health checks to pass, though /ask will fail.


def _gpu_snapshot() -> Optional[dict[str, float | str]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        device_index = torch.cuda.current_device()
        return {
            "device": torch.cuda.get_device_name(device_index),
            "memory_allocated_mb": torch.cuda.memory_allocated(device_index) / (1024 * 1024),
            "memory_reserved_mb": torch.cuda.memory_reserved(device_index) / (1024 * 1024),
        }
    except Exception:
        return None


# --- Routes ---

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask_endpoint():
    # 1. Parse Request Body
    data = request.get_json()
    if not data:
        return jsonify({"detail": "Invalid JSON body"}), 400

    try:
        # Validate with Pydantic
        payload = AskRequest(**data)
    except Exception as e:
        return jsonify({"detail": str(e)}), 422

    # 2. Performance Monitoring
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    mem_before = proc.memory_info()

    # Check GPU if used (unlikely with Pinecone Inference, but good to keep)
    device = getattr(rag_service, "device", None)
    gpu_before = _gpu_snapshot() if device == "cuda" else None

    start_ts = perf_counter()

    # 3. Call RAG Service
    try:
        result = rag_service.ask(
            question=payload.question,
            additional_context=payload.additional_context,
            top_k=payload.top_k,
            pool_size=payload.pool_size,
            temperature=payload.temperature,
            rerank=payload.rerank,
        )
    except Exception as e:
        LOGGER.error(f"Error during RAG ask: {e}", exc_info=True)
        return jsonify({"detail": "Internal Server Error during processing"}), 500

    duration = perf_counter() - start_ts

    # 4. Post-processing metrics
    cpu_after = proc.cpu_times()
    mem_after = proc.memory_info()
    gpu_after = _gpu_snapshot() if device == "cuda" else None

    cpu_user = cpu_after.user - cpu_before.user
    cpu_system = cpu_after.system - cpu_before.system
    mem_used_delta = mem_after.rss - mem_before.rss

    LOGGER.info(
        "ask request completed in %.3fs | cpu_user=%.3fs cpu_system=%.3fs mem_delta=%.2fMB gpu_before=%s gpu_after=%s",
        duration,
        cpu_user,
        cpu_system,
        mem_used_delta / (1024 * 1024),
        gpu_before,
        gpu_after,
    )

    # 5. Build Response
    response = AskResponse.from_chain_result(
        answer=result["answer"],
        citations=result["citations"],
        sources=result["sources"],
    )

    # Convert Pydantic model to dict for Flask jsonify
    return jsonify(response.model_dump())


# --- Static Files / Docs ---
if settings.serve_docs:
    mount_path = settings.docs_mount_path.rstrip("/") or "/docs"
    doc_dir = settings.resolved_data_dir

    @app.route(f"{mount_path}/<path:filename>")
    def serve_docs(filename):
        return send_from_directory(str(doc_dir), filename)


if __name__ == "__main__":
    # Development server
    # For production, utilize 'waitress-serve --port=7860 app.main:app'
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)
