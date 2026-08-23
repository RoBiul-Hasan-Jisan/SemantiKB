"""
Semantic / context-aware chunking — the main innovation of this system.

Algorithm
---------
1. Embed every sentence with a Sentence Transformer.
2. Compute cosine similarity between each pair of *consecutive* sentences.
3. A similarity **drop** (similarity falls below `similarity_threshold`)
   is a candidate boundary: it signals the topic has shifted, since
   embeddings of sentences about the same subject cluster tightly while
   embeddings of unrelated sentences do not.
4. Candidate boundaries are only honored if the resulting chunk would sit
   between `min_size_tokens` and `max_size_tokens`; boundaries inside that
   band are skipped (chunk kept growing) and windows that would exceed the
   max are force-split at the best available point even without a strong
   semantic drop.
5. If `prefer_paragraph_boundaries` is set, a paragraph/heading boundary
   already present in the source (sentence.section changes, or the
   sentence starts a new paragraph) is preferred over a purely
   embedding-derived split point when both are candidates within a few
   sentences of each other.
6. Optional sentence-level overlap is carried into the next chunk so
   retrieval doesn't lose context right at a boundary.

This directly addresses why fixed-size chunking is lossy: a fixed
token/character budget has no notion of *meaning* and will cut a chunk
in the middle of a still-developing idea purely because a counter hit a
number. Semantic chunking instead cuts where the discourse itself
changes topic, which keeps each chunk coherent and self-contained for
retrieval and grounding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.chunking.base import BaseChunker, _build_chunk, count_tokens
from backend.embeddings.embedder import Embedder, pairwise_consecutive_similarity
from backend.models import Chunk, ChunkingStrategy, Sentence


@dataclass
class SemanticChunkerConfig:
    similarity_threshold: float = 0.65
    min_size_tokens: int = 80
    max_size_tokens: int = 400
    overlap_tokens: int = 30
    prefer_paragraph_boundaries: bool = True
    boundary_search_window: int = 2  # sentences to look around a semantic drop for a paragraph edge


class SemanticChunker(BaseChunker):
    strategy = ChunkingStrategy.SEMANTIC

    def __init__(self, embedder: Embedder, config: SemanticChunkerConfig | None = None):
        self.embedder = embedder
        self.config = config or SemanticChunkerConfig()

    def chunk(self, document_id: str, version: int, sentences: list[Sentence]) -> list[Chunk]:
        if not sentences:
            return []
        if len(sentences) == 1:
            return [_build_chunk(document_id, version, self.strategy, 0, sentences)]

        texts = [s.text for s in sentences]
        embeddings = self.embedder.embed(texts, normalize=True)
        similarities = pairwise_consecutive_similarity(embeddings)  # len = n-1

        boundaries = self._find_boundaries(sentences, similarities)

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        overlap_carry: list[Sentence] = []
        for b in boundaries + [len(sentences)]:
            group = overlap_carry + sentences[start:b]
            if not group:
                start = b
                continue
            boundary_sim = similarities[b - 1] if 0 < b < len(sentences) else None
            chunks.append(
                _build_chunk(document_id, version, self.strategy, chunk_index, group, boundary_sim)
            )
            chunk_index += 1

            # build overlap for the next chunk
            overlap_carry = self._tail_overlap(group)
            start = b
        return chunks

    def _find_boundaries(self, sentences: list[Sentence], similarities: list[float]) -> list[int]:
        """Return sorted sentence indices at which a new chunk should start
        (i.e. boundary i means sentences[:i] and sentences[i:] are split)."""
        cfg = self.config
        n = len(sentences)
        boundaries: list[int] = []
        current_start = 0
        running_tokens = 0

        for i in range(n):
            running_tokens += count_tokens(sentences[i].text)
            is_last = i == n - 1
            sim_drop = (not is_last) and (similarities[i] < cfg.similarity_threshold)
            chunk_len = i - current_start + 1

            too_big = running_tokens >= cfg.max_size_tokens
            big_enough = running_tokens >= cfg.min_size_tokens

            should_split = False
            if too_big:
                should_split = True
            elif sim_drop and big_enough:
                should_split = True

            if should_split and not is_last and chunk_len > 0:
                split_at = i + 1
                if cfg.prefer_paragraph_boundaries:
                    split_at = self._snap_to_paragraph(sentences, split_at, current_start)
                boundaries.append(split_at)
                current_start = split_at
                running_tokens = 0

        return boundaries

    def _snap_to_paragraph(self, sentences: list[Sentence], candidate: int, chunk_start: int) -> int:
        """Look within `boundary_search_window` sentences of `candidate` for a
        section/paragraph change and prefer that exact point if found."""
        cfg = self.config
        n = len(sentences)
        lo = max(chunk_start + 1, candidate - cfg.boundary_search_window)
        hi = min(n, candidate + cfg.boundary_search_window)
        for offset in range(0, cfg.boundary_search_window + 1):
            for idx in {candidate + offset, candidate - offset}:
                if lo <= idx < hi and idx > chunk_start:
                    prev_section = sentences[idx - 1].section
                    this_section = sentences[idx].section
                    if prev_section != this_section:
                        return idx
        return candidate

    def _tail_overlap(self, group: list[Sentence]) -> list[Sentence]:
        cfg = self.config
        if cfg.overlap_tokens <= 0:
            return []
        tail: list[Sentence] = []
        tokens = 0
        for s in reversed(group):
            t = count_tokens(s.text)
            if tokens + t > cfg.overlap_tokens and tail:
                break
            tail.insert(0, s)
            tokens += t
        return tail
