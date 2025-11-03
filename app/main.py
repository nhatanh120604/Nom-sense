from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .deps import get_app_settings, get_rag_service
from .rag.pipeline import RagService
from .schemas import AskRequest, AskResponse
from .settings import Settings

import logging

import psutil

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _gpu_snapshot() -> Optional[dict[str, float | str]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_index = torch.cuda.current_device()
        return {
            "device": torch.cuda.get_device_name(device_index),
            "memory_allocated_mb": torch.cuda.memory_allocated(device_index)
            / (1024 * 1024),
            "memory_reserved_mb": torch.cuda.memory_reserved(device_index)
            / (1024 * 1024),
        }
    except Exception:
        return None


settings = get_app_settings()

app = FastAPI(
    title="Vietnamese RAG Assistant",
    docs_url="/swagger",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mount_static_docs(settings: Settings) -> None:
    if not settings.serve_docs:
        return
    doc_dir: Path = settings.resolved_data_dir
    if not doc_dir.exists():
        return
    mount_path = settings.docs_mount_path.rstrip("/") or "/docs"
    already_mounted = any(
        getattr(route, "path", None) == mount_path for route in app.routes
    )
    if already_mounted:
        return
    app.mount(
        mount_path, StaticFiles(directory=str(doc_dir), check_dir=False), name="docs"
    )


_mount_static_docs(settings)


@app.on_event("startup")
async def configure_app() -> None:
    if settings.auto_ingest_on_startup:
        rag = get_rag_service()
        rag.ensure_vectorstore(force_rebuild=False)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask_endpoint(
    payload: AskRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> AskResponse:
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    mem_before = proc.memory_info()
    gpu_before = _gpu_snapshot() if getattr(rag, "device", None) == "cuda" else None

    start_ts = perf_counter()
    result = rag.ask(
        question=payload.question,
        additional_context=payload.additional_context,
        top_k=payload.top_k,
        pool_size=payload.pool_size,
        temperature=payload.temperature,
        rerank=payload.rerank,
    )
    duration = perf_counter() - start_ts
    cpu_after = proc.cpu_times()
    mem_after = proc.memory_info()
    gpu_after = _gpu_snapshot() if getattr(rag, "device", None) == "cuda" else None

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
    return AskResponse.from_chain_result(
        answer=result["answer"],
        citations=result["citations"],
        sources=result["sources"],
    )
