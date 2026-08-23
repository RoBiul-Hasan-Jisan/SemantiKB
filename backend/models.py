"""
Shared Pydantic models / schema definitions.

Hierarchy maintained across the system:

    document -> version -> section -> chunk -> sentences

These models are the single source of truth for what gets stored in
SQLite (structured metadata) and ChromaDB (vectors + a copy of the
metadata needed for filtering).
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    SUMMARIZING = "summarizing"
    READY = "ready"
    FAILED = "failed"


# --------------------------------------------------------------------------------------
# Document / version
# --------------------------------------------------------------------------------------

class Document(BaseModel):
    """A logical document identity. May have many DocumentVersions."""
    document_id: str = Field(default_factory=lambda: new_id("doc_"))
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    latest_version: int = 1
    status: DocumentStatus = DocumentStatus.PENDING


class DocumentVersion(BaseModel):
    """
    A specific version of a document's content.

    valid_from / valid_to support time-scoped queries such as
    "what was the policy in 2025?" — valid_to is None for the
    currently-active version.
    """
    document_id: str
    version: int
    filename: str
    file_path: str
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_to: Optional[datetime] = None
    page_count: int = 0
    is_latest: bool = True


class Page(BaseModel):
    document_id: str
    version: int
    page_number: int
    text: str
    section: Optional[str] = None


# --------------------------------------------------------------------------------------
# Sections / chunks / sentences
# --------------------------------------------------------------------------------------

class Sentence(BaseModel):
    sentence_id: str = Field(default_factory=lambda: new_id("sent_"))
    document_id: str
    version: int
    page_number: int
    section: Optional[str] = None
    index_in_doc: int
    text: str


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: new_id("chunk_"))
    document_id: str
    version: int
    section: Optional[str] = None
    page_start: int
    page_end: int
    sentence_ids: list[str] = Field(default_factory=list)
    text: str
    token_count: int
    strategy: ChunkingStrategy
    chunk_index: int  # position within document for this strategy
    # similarity of this chunk's first sentence to the previous chunk's last
    # sentence, kept for debugging / evaluation of the semantic chunker
    boundary_similarity: Optional[float] = None
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Section(BaseModel):
    section_id: str = Field(default_factory=lambda: new_id("sec_"))
    document_id: str
    version: int
    title: str
    chunk_ids: list[str] = Field(default_factory=list)
    summary: Optional[str] = None


class DocumentSummary(BaseModel):
    document_id: str
    version: int
    summary: str
    section_summaries: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------------------
# Retrieval / chat
# --------------------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    version: int
    page_start: int
    page_end: int
    section: Optional[str] = None
    text: str
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[list[str]] = None
    version: Optional[int] = None
    time_filter: Optional[str] = None  # e.g. "2025", "latest"
    top_k: Optional[int] = None
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC


class Citation(BaseModel):
    document_name: str
    version: int
    page: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class DiffRequest(BaseModel):
    document_id: str
    version_a: int
    version_b: int
