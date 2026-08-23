"""
Retrieval orchestration: vector search (ChromaDB), BM25 keyword search,
and a hybrid combination of the two via min-max normalized score fusion.
Also implements temporal retrieval: resolving "latest" vs a specific
year/version before chunks are fetched.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from backend.config import settings
from backend.database.repository import Repository
from backend.database.vector_store import VectorStore
from backend.embeddings.embedder import Embedder
from backend.models import ChunkingStrategy, RetrievedChunk
from backend.retrieval.bm25_retriever import BM25Retriever


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class TemporalResolver:
    """Resolves a natural-language time filter into a specific version to
    scope retrieval by, per requirement 10 (temporal retrieval)."""

    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

    def __init__(self, repo: Repository):
        self.repo = repo

    def resolve_version(self, document_id: str, time_filter: Optional[str], query: str) -> Optional[int]:
        text = f"{time_filter or ''} {query}".lower()
        if "latest" in text or "current" in text or "now" in text or not time_filter:
            latest = self.repo.get_latest_version(document_id)
            return latest.version if latest and "latest" in text else None
        year_match = self.YEAR_RE.search(text)
        if year_match:
            year = int(year_match.group(0))
            # find the version whose valid_from..valid_to window covers that year
            versions = self.repo.list_versions(document_id)
            for v in versions:
                start_year = v.valid_from.year
                end_year = v.valid_to.year if v.valid_to else datetime.utcnow().year
                if start_year <= year <= end_year:
                    return v.version
        return None


class HybridRetriever:
    def __init__(self, repo: Repository, vector_store: VectorStore, embedder: Embedder):
        self.repo = repo
        self.vector_store = vector_store
        self.embedder = embedder
        self.temporal = TemporalResolver(repo)

    def retrieve(
        self,
        query: str,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        document_ids: Optional[list[str]] = None,
        version: Optional[int] = None,
        time_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k
        mode = mode or settings.retrieval_mode

        # resolve temporal filter to a concrete version, per-document if needed
        resolved_version = version
        if resolved_version is None and document_ids and len(document_ids) == 1:
            resolved_version = self.temporal.resolve_version(document_ids[0], time_filter, query)

        vector_hits = self._vector_search(query, strategy, document_ids, resolved_version, top_k * 3)

        bm25_hits: list[tuple[str, float]] = []
        if mode in ("bm25", "hybrid"):
            bm25_hits = self._bm25_search(query, strategy, document_ids, resolved_version, top_k * 3)

        merged = self._merge(vector_hits, bm25_hits, mode, top_k)
        return self._to_retrieved_chunks(merged)

    def _vector_search(self, query, strategy, document_ids, version, k) -> list[dict]:
        q_emb = self.embedder.embed_one(query).tolist()
        return self.vector_store.query(q_emb, strategy, top_k=k, document_ids=document_ids, version=version)

    def _bm25_search(self, query, strategy, document_ids, version, k) -> list[tuple[str, float]]:
        # pull candidate chunks from SQLite scoped by document/version, build
        # an ephemeral BM25 index over them
        chunks = []
        doc_ids = document_ids or [d.document_id for d in self.repo.list_documents()]
        for doc_id in doc_ids:
            v = version or (self.repo.get_latest_version(doc_id).version if self.repo.get_latest_version(doc_id) else 1)
            chunks.extend(self.repo.get_chunks(doc_id, v, strategy.value))
        retriever = BM25Retriever(chunks)
        results = retriever.query(query, top_k=k)
        return [(c.chunk_id, score) for c, score in results]

    def _merge(self, vector_hits: list[dict], bm25_hits: list[tuple[str, float]], mode: str, top_k: int) -> list[dict]:
        vec_by_id = {h["chunk_id"]: h for h in vector_hits}
        vec_scores = _minmax_normalize([h["score"] or 0.0 for h in vector_hits])
        vec_norm = {h["chunk_id"]: s for h, s in zip(vector_hits, vec_scores)}

        bm25_ids = [cid for cid, _ in bm25_hits]
        bm25_raw = [score for _, score in bm25_hits]
        bm25_norm_scores = _minmax_normalize(bm25_raw)
        bm25_norm = dict(zip(bm25_ids, bm25_norm_scores))
        bm25_score_map = dict(bm25_hits)

        all_ids = set(vec_by_id) | set(bm25_norm)
        alpha = settings.hybrid_alpha

        scored = []
        for cid in all_ids:
            v_score = vec_norm.get(cid, 0.0)
            b_score = bm25_norm.get(cid, 0.0)
            if mode == "vector":
                final = v_score
            elif mode == "bm25":
                final = b_score
            else:
                final = alpha * v_score + (1 - alpha) * b_score
            scored.append({
                "chunk_id": cid,
                "vector_score": vec_by_id.get(cid, {}).get("score"),
                "bm25_score": bm25_score_map.get(cid),
                "hybrid_score": final,
                "metadata": vec_by_id.get(cid, {}).get("metadata"),
                "text": vec_by_id.get(cid, {}).get("text"),
            })
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:top_k]

    def _to_retrieved_chunks(self, merged: list[dict]) -> list[RetrievedChunk]:
        out = []
        for m in merged:
            chunk = self.repo.get_chunk(m["chunk_id"])
            if chunk is None:
                continue
            doc = self.repo.get_document(chunk.document_id)
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=doc.filename if doc else chunk.document_id,
                    version=chunk.version,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    text=chunk.text,
                    vector_score=m.get("vector_score"),
                    bm25_score=m.get("bm25_score"),
                    hybrid_score=m.get("hybrid_score"),
                )
            )
        return out
