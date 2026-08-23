"""
Thin wrapper around ChromaDB. Chosen over FAISS (per spec) because
metadata filtering (document_id, version, strategy, section, page) is
central to temporal / scoped retrieval and Chroma supports it natively
without hand-rolled index bookkeeping.

One collection per chunking strategy so the three approaches (fixed,
recursive, semantic) can be evaluated side-by-side without collisions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.models import Chunk, ChunkingStrategy


class VectorStore:
    def __init__(self, persist_path: Path):
        self.client = chromadb.PersistentClient(
            path=str(persist_path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collections: dict[str, "chromadb.Collection"] = {}

    def _collection(self, strategy: ChunkingStrategy):
        name = f"chunks_{strategy.value}"
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        strategy = chunks[0].strategy
        coll = self._collection(strategy)
        coll.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "version": c.version,
                    "section": c.section or "",
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        strategy: ChunkingStrategy,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
        version: Optional[int] = None,
    ) -> list[dict]:
        coll = self._collection(strategy)
        where: dict = {}
        clauses = []
        if document_ids:
            clauses.append({"document_id": {"$in": document_ids}})
        if version is not None:
            clauses.append({"version": version})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        result = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or None,
        )
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, cid in enumerate(ids):
            # cosine distance -> similarity
            similarity = 1 - dists[i] if dists else None
            out.append({"chunk_id": cid, "text": docs[i], "metadata": metas[i], "score": similarity})
        return out

    def delete_document(self, document_id: str, strategy: Optional[ChunkingStrategy] = None) -> None:
        strategies = [strategy] if strategy else list(ChunkingStrategy)
        for s in strategies:
            coll = self._collection(s)
            coll.delete(where={"document_id": document_id})
