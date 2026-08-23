"""
Keyword retrieval via BM25 (rank_bm25). Built in-memory per (document set,
strategy) since corpora here are personal-knowledge-base sized; for very
large corpora this index would move to a persistent store, but BM25 index
build time is linear and fast enough to rebuild per-request at this scale.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from backend.models import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def query(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self._bm25 or not self.chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
