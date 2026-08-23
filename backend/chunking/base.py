"""
Shared interfaces and utilities for the three chunking strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from backend.models import Chunk, ChunkingStrategy, Sentence


@lru_cache(maxsize=1)
def _get_tokenizer():
    """
    Use tiktoken if available for realistic token counts (matches what an
    LLM actually sees); fall back to a whitespace-word approximation so the
    system has zero hard dependency on any single tokenizer.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _get_tokenizer()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text.split()))


class BaseChunker(ABC):
    strategy: ChunkingStrategy

    @abstractmethod
    def chunk(self, document_id: str, version: int, sentences: list[Sentence]) -> list[Chunk]:
        """Turn an ordered list of sentences into an ordered list of Chunks."""
        raise NotImplementedError


def _build_chunk(
    document_id: str,
    version: int,
    strategy: ChunkingStrategy,
    chunk_index: int,
    sentence_group: list[Sentence],
    boundary_similarity: float | None = None,
) -> Chunk:
    text = " ".join(s.text for s in sentence_group)
    pages = [s.page_number for s in sentence_group]
    sections = [s.section for s in sentence_group if s.section]
    return Chunk(
        document_id=document_id,
        version=version,
        section=sections[0] if sections else None,
        page_start=min(pages),
        page_end=max(pages),
        sentence_ids=[s.sentence_id for s in sentence_group],
        text=text,
        token_count=count_tokens(text),
        strategy=strategy,
        chunk_index=chunk_index,
        boundary_similarity=boundary_similarity,
    )
