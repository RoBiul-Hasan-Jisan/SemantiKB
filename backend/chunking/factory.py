"""Factory for selecting a chunking strategy at runtime (used by ingestion
and by the evaluation module to run all three strategies uniformly)."""
from __future__ import annotations

from backend.chunking.base import BaseChunker
from backend.chunking.fixed_chunker import FixedSizeChunker
from backend.chunking.recursive_chunker import RecursiveCharacterChunker
from backend.chunking.semantic_chunker import SemanticChunker, SemanticChunkerConfig
from backend.config import settings
from backend.embeddings.embedder import Embedder
from backend.models import ChunkingStrategy


def get_chunker(strategy: ChunkingStrategy, embedder: Embedder | None = None) -> BaseChunker:
    if strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker(
            chunk_size_tokens=settings.fixed_chunk_size_tokens,
            overlap_tokens=settings.fixed_chunk_overlap_tokens,
        )
    if strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterChunker(
            chunk_size_tokens=settings.recursive_chunk_size_tokens,
            overlap_tokens=settings.recursive_chunk_overlap_tokens,
        )
    if strategy == ChunkingStrategy.SEMANTIC:
        assert embedder is not None, "SemanticChunker requires an embedder"
        cfg = SemanticChunkerConfig(
            similarity_threshold=settings.chunk_similarity_threshold,
            min_size_tokens=settings.chunk_min_size_tokens,
            max_size_tokens=settings.chunk_max_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            prefer_paragraph_boundaries=settings.prefer_paragraph_boundaries,
        )
        return SemanticChunker(embedder, cfg)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def get_all_chunkers(embedder: Embedder) -> dict[ChunkingStrategy, BaseChunker]:
    return {s: get_chunker(s, embedder) for s in ChunkingStrategy}
