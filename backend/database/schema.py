"""
SQLite schema for structured metadata (documents, versions, sections,
chunks, summaries). ChromaDB stores vectors + a denormalized copy of the
filter fields; SQLite is the source of truth for relational data such as
version history and hierarchy relationships.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    latest_version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS document_versions (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    page_count INTEGER DEFAULT 0,
    is_latest INTEGER DEFAULT 1,
    PRIMARY KEY (document_id, version),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS pages (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    section TEXT,
    PRIMARY KEY (document_id, version, page_number)
);

CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    section TEXT,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    boundary_similarity REAL,
    summary TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_ver_strategy
    ON chunks(document_id, version, strategy);

CREATE TABLE IF NOT EXISTS document_summaries (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    summary TEXT NOT NULL,
    section_summaries TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (document_id, version)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
