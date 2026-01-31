"""
Pydantic schemas for API requests and responses.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    additional_context: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1)
    pool_size: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    rerank: bool = True


class SourceChunk(BaseModel):
    label: str
    page_number: Optional[int] = None
    chapter: Optional[str] = None
    book_title: Optional[str] = None
    file_name: Optional[str] = None
    source_path: Optional[str] = None
    text: str
    viewer_url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    sources: List[SourceChunk]

    @classmethod
    def from_chain_result(
        cls, *, answer: str, citations: list[str], sources: list[SourceChunk]
    ) -> "AskResponse":
        return cls(answer=answer, citations=citations, sources=sources)
