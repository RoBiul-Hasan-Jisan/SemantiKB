"""
Optional cross-encoder reranking stage. Configurable and not mandatory
(per requirement 8) — disabled by default via settings.use_reranker
since cross-encoders are slower than the bi-encoder retrieval step.
"""
from __future__ import annotations

import logging

from backend.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        pairs = [[query, c.text] for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k] if top_k else ranked
