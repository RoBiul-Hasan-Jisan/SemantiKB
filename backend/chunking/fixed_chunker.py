"""
Baseline 1: fixed-size chunking. Groups sentences until a target token
budget is hit, with a fixed sentence-based overlap carried into the next
chunk. This is the naive approach the semantic chunker is compared
against — it has no awareness of topic boundaries and will happily split
a sentence's surrounding argument in half.
"""
from __future__ import annotations

from backend.chunking.base import BaseChunker, _build_chunk, count_tokens
from backend.models import Chunk, ChunkingStrategy, Sentence


class FixedSizeChunker(BaseChunker):
    strategy = ChunkingStrategy.FIXED

    def __init__(self, chunk_size_tokens: int = 250, overlap_tokens: int = 30):
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document_id: str, version: int, sentences: list[Sentence]) -> list[Chunk]:
        chunks: list[Chunk] = []
        i = 0
        chunk_index = 0
        n = len(sentences)
        while i < n:
            group: list[Sentence] = []
            tokens = 0
            j = i
            while j < n and tokens < self.chunk_size_tokens:
                t = count_tokens(sentences[j].text)
                group.append(sentences[j])
                tokens += t
                j += 1
            if not group:
                break
            chunks.append(_build_chunk(document_id, version, self.strategy, chunk_index, group))
            chunk_index += 1

            # step forward, but back up to create overlap
            overlap_tokens = 0
            back = 0
            for s in reversed(group):
                overlap_tokens += count_tokens(s.text)
                back += 1
                if overlap_tokens >= self.overlap_tokens:
                    break
            i = j - back if j - back > i else j
        return chunks
