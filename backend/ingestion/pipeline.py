"""
End-to-end ingestion pipeline:

  upload -> parse -> segment sentences -> chunk (all 3 strategies)
         -> embed -> store in vector DB + SQLite -> hierarchical summarize

All three chunking strategies are run and stored so the evaluation module
can compare them later using identical source documents and queries
(requirement 3).
"""
from __future__ import annotations

import logging

from backend.chunking.factory import get_all_chunkers
from backend.config import settings
from backend.database.repository import Repository
from backend.database.vector_store import VectorStore
from backend.embeddings.embedder import Embedder
from backend.ingestion.document_parser import parse_document
from backend.ingestion.sentence_segmenter import segment_pages
from backend.llm.ollama_client import OllamaClient
from backend.models import ChunkingStrategy, Document, DocumentStatus, Page
from backend.summarization.summarizer import HierarchicalSummarizer
from backend.versioning.version_manager import VersionManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        repo: Repository,
        vector_store: VectorStore,
        embedder: Embedder,
        llm: OllamaClient | None = None,
    ):
        self.repo = repo
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm or OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_temperature)
        self.version_manager = VersionManager(repo)

    def ingest(self, file_path: str, filename: str, existing_document_id: str | None = None) -> Document:
        pages = parse_document(file_path, filename)
        full_text = "\n".join(p.text for p in pages)

        doc, version = self.version_manager.register_upload(
            filename=filename, file_path=file_path, full_text=full_text,
            page_count=len(pages), existing_document_id=existing_document_id,
        )
        self.repo.set_document_status(doc.document_id, DocumentStatus.PARSING)

        self.repo.add_pages([
            Page(document_id=doc.document_id, version=version.version, page_number=p.page_number,
                 text=p.text, section=p.section)
            for p in pages
        ])

        self.repo.set_document_status(doc.document_id, DocumentStatus.CHUNKING)
        sentences = segment_pages(doc.document_id, version.version, pages)

        chunkers = get_all_chunkers(self.embedder)
        self.repo.set_document_status(doc.document_id, DocumentStatus.EMBEDDING)

        semantic_chunks = None
        for strategy, chunker in chunkers.items():
            chunks = chunker.chunk(doc.document_id, version.version, sentences)
            if not chunks:
                continue
            self.repo.add_chunks(chunks)
            embeddings = self.embedder.embed([c.text for c in chunks]).tolist()
            self.vector_store.upsert_chunks(chunks, embeddings)
            if strategy == ChunkingStrategy.SEMANTIC:
                semantic_chunks = chunks
            logger.info("Indexed %d chunks for %s using %s strategy", len(chunks), doc.filename, strategy.value)

        if settings.enable_hierarchical_summarization and semantic_chunks:
            self.repo.set_document_status(doc.document_id, DocumentStatus.SUMMARIZING)
            summarizer = HierarchicalSummarizer(self.repo, self.llm, settings.summary_max_tokens)
            try:
                summarizer.run(doc.document_id, version.version, semantic_chunks)
            except Exception as e:
                logger.warning("Summarization skipped due to error: %s", e)

        self.repo.set_document_status(doc.document_id, DocumentStatus.READY)
        return doc
