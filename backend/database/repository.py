"""
Repository layer: all SQLite reads/writes go through here so the rest of
the codebase never touches raw SQL.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.database.schema import get_conn, init_db
from backend.models import (
    Chunk,
    Document,
    DocumentStatus,
    DocumentSummary,
    DocumentVersion,
    Page,
    Section,
)


class Repository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        init_db(db_path)

    # --- documents -----------------------------------------------------------------
    def upsert_document(self, doc: Document) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """INSERT INTO documents (document_id, filename, created_at, latest_version, status)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                     latest_version=excluded.latest_version, status=excluded.status""",
                (doc.document_id, doc.filename, doc.created_at.isoformat(), doc.latest_version, doc.status.value),
            )

    def set_document_status(self, document_id: str, status: DocumentStatus) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("UPDATE documents SET status=? WHERE document_id=?", (status.value, document_id))

    def list_documents(self) -> list[Document]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [
            Document(
                document_id=r["document_id"],
                filename=r["filename"],
                created_at=datetime.fromisoformat(r["created_at"]),
                latest_version=r["latest_version"],
                status=DocumentStatus(r["status"]),
            )
            for r in rows
        ]

    def get_document(self, document_id: str) -> Optional[Document]:
        with get_conn(self.db_path) as conn:
            r = conn.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
        if not r:
            return None
        return Document(
            document_id=r["document_id"],
            filename=r["filename"],
            created_at=datetime.fromisoformat(r["created_at"]),
            latest_version=r["latest_version"],
            status=DocumentStatus(r["status"]),
        )

    # --- versions --------------------------------------------------------------------
    def add_version(self, v: DocumentVersion) -> None:
        with get_conn(self.db_path) as conn:
            # close out the previous "latest" version, if any
            conn.execute(
                """UPDATE document_versions SET is_latest=0, valid_to=?
                   WHERE document_id=? AND is_latest=1""",
                (v.valid_from.isoformat(), v.document_id),
            )
            conn.execute(
                """INSERT INTO document_versions
                   (document_id, version, filename, file_path, content_hash, created_at,
                    updated_at, valid_from, valid_to, page_count, is_latest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    v.document_id, v.version, v.filename, v.file_path, v.content_hash,
                    v.created_at.isoformat(), v.updated_at.isoformat(), v.valid_from.isoformat(),
                    v.valid_to.isoformat() if v.valid_to else None, v.page_count, 1,
                ),
            )

    def list_versions(self, document_id: str) -> list[DocumentVersion]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM document_versions WHERE document_id=? ORDER BY version",
                (document_id,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_version(self, document_id: str, version: int) -> Optional[DocumentVersion]:
        with get_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM document_versions WHERE document_id=? AND version=?",
                (document_id, version),
            ).fetchone()
        return self._row_to_version(r) if r else None

    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]:
        with get_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM document_versions WHERE document_id=? AND is_latest=1",
                (document_id,),
            ).fetchone()
        return self._row_to_version(r) if r else None

    def get_version_at_time(self, document_id: str, ts: datetime) -> Optional[DocumentVersion]:
        """Find the version that was valid at a given point in time."""
        with get_conn(self.db_path) as conn:
            r = conn.execute(
                """SELECT * FROM document_versions WHERE document_id=?
                   AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                   ORDER BY version DESC LIMIT 1""",
                (document_id, ts.isoformat(), ts.isoformat()),
            ).fetchone()
        return self._row_to_version(r) if r else None

    @staticmethod
    def _row_to_version(r) -> DocumentVersion:
        return DocumentVersion(
            document_id=r["document_id"],
            version=r["version"],
            filename=r["filename"],
            file_path=r["file_path"],
            content_hash=r["content_hash"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
            valid_from=datetime.fromisoformat(r["valid_from"]),
            valid_to=datetime.fromisoformat(r["valid_to"]) if r["valid_to"] else None,
            page_count=r["page_count"],
            is_latest=bool(r["is_latest"]),
        )

    # --- pages -------------------------------------------------------------------------
    def add_pages(self, pages: list[Page]) -> None:
        with get_conn(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO pages (document_id, version, page_number, text, section)
                   VALUES (?, ?, ?, ?, ?)""",
                [(p.document_id, p.version, p.page_number, p.text, p.section) for p in pages],
            )

    def get_pages(self, document_id: str, version: int) -> list[Page]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE document_id=? AND version=? ORDER BY page_number",
                (document_id, version),
            ).fetchall()
        return [Page(document_id=r["document_id"], version=r["version"], page_number=r["page_number"],
                      text=r["text"], section=r["section"]) for r in rows]

    # --- chunks ------------------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> None:
        with get_conn(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, document_id, version, section, page_start, page_end, text,
                    token_count, strategy, chunk_index, boundary_similarity, summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.chunk_id, c.document_id, c.version, c.section, c.page_start, c.page_end,
                     c.text, c.token_count, c.strategy.value, c.chunk_index, c.boundary_similarity,
                     c.summary, c.created_at.isoformat())
                    for c in chunks
                ],
            )

    def get_chunks(self, document_id: str, version: int, strategy: str) -> list[Chunk]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM chunks WHERE document_id=? AND version=? AND strategy=?
                   ORDER BY chunk_index""",
                (document_id, version, strategy),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with get_conn(self.db_path) as conn:
            r = conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._row_to_chunk(r) if r else None

    def update_chunk_summary(self, chunk_id: str, summary: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute("UPDATE chunks SET summary=? WHERE chunk_id=?", (summary, chunk_id))

    @staticmethod
    def _row_to_chunk(r) -> Chunk:
        from backend.models import ChunkingStrategy
        return Chunk(
            chunk_id=r["chunk_id"], document_id=r["document_id"], version=r["version"],
            section=r["section"], page_start=r["page_start"], page_end=r["page_end"],
            text=r["text"], token_count=r["token_count"], strategy=ChunkingStrategy(r["strategy"]),
            chunk_index=r["chunk_index"], boundary_similarity=r["boundary_similarity"],
            summary=r["summary"], created_at=datetime.fromisoformat(r["created_at"]),
        )

    # --- sections -------------------------------------------------------------------
    def add_sections(self, sections: list[Section]) -> None:
        with get_conn(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO sections (section_id, document_id, version, title, summary) VALUES (?,?,?,?,?)",
                [(s.section_id, s.document_id, s.version, s.title, s.summary) for s in sections],
            )

    # --- summaries ------------------------------------------------------------------
    def add_document_summary(self, s: DocumentSummary) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO document_summaries
                   (document_id, version, summary, section_summaries, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (s.document_id, s.version, s.summary, json.dumps(s.section_summaries), s.created_at.isoformat()),
            )

    def get_document_summary(self, document_id: str, version: int) -> Optional[DocumentSummary]:
        with get_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM document_summaries WHERE document_id=? AND version=?",
                (document_id, version),
            ).fetchone()
        if not r:
            return None
        return DocumentSummary(
            document_id=r["document_id"], version=r["version"], summary=r["summary"],
            section_summaries=json.loads(r["section_summaries"] or "{}"),
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    # --- eval ---------------------------------------------------------------------------
    def add_eval_run(self, run_id: str, strategy: str, metrics: dict) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_runs (run_id, strategy, created_at, metrics_json) VALUES (?,?,?,?)",
                (run_id, strategy, datetime.utcnow().isoformat(), json.dumps(metrics)),
            )

    def list_eval_runs(self) -> list[dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY created_at DESC").fetchall()
        return [
            {"run_id": r["run_id"], "strategy": r["strategy"], "created_at": r["created_at"],
             "metrics": json.loads(r["metrics_json"])}
            for r in rows
        ]
