"""Pydantic schemas and configuration models for the chatmem local API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ApiConfig(BaseModel):
    """Configuration for chatmem local HTTP service."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8000
    default_mode: Literal["fast", "precise"] = "fast"
    auth_token_path: str = "~/.chatmem/token"
    allowed_cors_origins: list[str] = Field(default_factory=list)
    max_results: int = 10
    max_injected_chars: int = 4000
    max_message_chars: int = 30000
    max_request_bytes: int = 1_000_000
    allow_all_interfaces: bool = False

    # Mode thresholds
    # NOTE: 0.70 is an UNCALIBRATED placeholder default; calibrate on your private eval set.
    fast_gate_threshold: float = 0.70
    precise_gate_threshold: float = 0.010
    precise_rerank_candidates: int = 20


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    embedding_model: str
    rerank_model: str
    reranker_loaded: bool
    conversations_indexed: int
    chunks_indexed: int


class Citation(BaseModel):
    platform: str
    conversation_id: str
    title: str | None
    timestamp: str


class SearchResultChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation: Citation


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Literal["fast", "precise"] | None = None
    top_k: int | None = Field(default=None, ge=1)
    platform: str | None = None
    since: str | None = None


class SearchResponse(BaseModel):
    injected: bool
    reason: str
    gate_score: float | None
    score_scale: str
    results: list[SearchResultChunk]
    timings_ms: dict[str, float]


class ContextRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Literal["fast", "precise"] | None = None
    max_chars: int | None = Field(default=None, ge=100)


class ContextResponse(BaseModel):
    injected: bool
    reason: str
    context: str
    chunks_used: int
    total_chars: int


class IngestMessageItem(BaseModel):
    role: str
    text: str
    timestamp: str | datetime | None = None
    external_id: str | None = None


class IngestMessagesRequest(BaseModel):
    platform: str
    conversation_id: str
    title: str | None = "Untitled"
    messages: list[IngestMessageItem] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    status: str = "ok"
    conversation_id: str
    platform: str
    chunks_indexed: int
    messages_indexed: int


class DeleteResponse(BaseModel):
    status: str = "deleted"
    conversation_id: str
    platform: str
    chunks_deleted: int
