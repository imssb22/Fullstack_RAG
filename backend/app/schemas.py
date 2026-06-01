from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    title: str
    filename: str
    source_url: str | None = None
    file_type: str
    created_at: datetime
    chunk_count: int = 0
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None


class RetrievedChunk(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    document_title: str
    filename: str
    source_url: str | None = None
    page: int | None = None
    chunk_index: int
    text: str
    score: float


class Citation(BaseModel):
    source_id: str
    document_id: str
    document_title: str
    filename: str
    source_url: str | None = None
    page: int | None = None
    chunk_index: int
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=12)


class AskResponse(BaseModel):
    answer: str
    supported: bool
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    raw_model_output: dict[str, Any] | None = None


class UploadResponse(BaseModel):
    documents: list[DocumentRecord]


class IngestSamplesResponse(BaseModel):
    documents: list[DocumentRecord]
    downloaded: list[str]
    skipped: list[str]
