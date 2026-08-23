"""
Hierarchical summarization, maintaining the relationship:

    document -> section -> chunk -> sentences

Uses the local Ollama LLM for abstractive summaries. Falls back to a
simple extractive summary (first + highest-tf-idf sentences) if Ollama is
unreachable, so ingestion never hard-fails just because the LLM is down.
"""
from __future__ import annotations

import logging

from backend.database.repository import Repository
from backend.llm.ollama_client import OllamaClient
from backend.models import Chunk, DocumentSummary, Section

logger = logging.getLogger(__name__)

CHUNK_SUMMARY_PROMPT = """Summarize the following passage in 1-2 concise sentences, \
preserving key facts, names, and numbers. Do not add outside information.

Passage:
{text}

Summary:"""

SECTION_SUMMARY_PROMPT = """Combine the following chunk summaries from the same section \
into a single coherent paragraph summarizing the section. Do not add outside information.

Chunk summaries:
{summaries}

Section summary:"""

DOCUMENT_SUMMARY_PROMPT = """Combine the following section summaries into a concise \
overall document summary (3-5 sentences). Do not add outside information.

Section summaries:
{summaries}

Document summary:"""


def _extractive_fallback(text: str, max_sentences: int = 2) -> str:
    from backend.ingestion.sentence_segmenter import split_sentences
    sents = split_sentences(text)
    return " ".join(sents[:max_sentences]) if sents else text[:200]


class HierarchicalSummarizer:
    def __init__(self, repo: Repository, llm: OllamaClient, max_tokens: int = 200):
        self.repo = repo
        self.llm = llm
        self.max_tokens = max_tokens

    def summarize_chunk(self, chunk: Chunk) -> str:
        try:
            summary = self.llm.generate(CHUNK_SUMMARY_PROMPT.format(text=chunk.text), max_tokens=self.max_tokens)
            return summary.strip()
        except Exception as e:
            logger.warning("LLM summarization failed for chunk %s (%s); using extractive fallback", chunk.chunk_id, e)
            return _extractive_fallback(chunk.text)

    def summarize_section(self, chunk_summaries: list[str]) -> str:
        joined = "\n".join(f"- {s}" for s in chunk_summaries)
        try:
            return self.llm.generate(SECTION_SUMMARY_PROMPT.format(summaries=joined), max_tokens=self.max_tokens).strip()
        except Exception as e:
            logger.warning("LLM section summarization failed (%s); concatenating", e)
            return " ".join(chunk_summaries)

    def summarize_document(self, section_summaries: list[str]) -> str:
        joined = "\n".join(f"- {s}" for s in section_summaries)
        try:
            return self.llm.generate(DOCUMENT_SUMMARY_PROMPT.format(summaries=joined), max_tokens=self.max_tokens).strip()
        except Exception as e:
            logger.warning("LLM document summarization failed (%s); concatenating", e)
            return " ".join(section_summaries)

    def run(self, document_id: str, version: int, chunks: list[Chunk]) -> DocumentSummary:
        """Full pipeline: summarize every chunk, group by section, summarize
        each section, then summarize the document as a whole."""
        chunk_summaries: dict[str, str] = {}
        for chunk in chunks:
            summary = self.summarize_chunk(chunk)
            chunk_summaries[chunk.chunk_id] = summary
            self.repo.update_chunk_summary(chunk.chunk_id, summary)

        sections: dict[str, list[Chunk]] = {}
        for c in chunks:
            key = c.section or "Unsectioned"
            sections.setdefault(key, []).append(c)

        section_summary_map: dict[str, str] = {}
        section_objs = []
        for title, section_chunks in sections.items():
            summaries = [chunk_summaries[c.chunk_id] for c in section_chunks]
            sec_summary = self.summarize_section(summaries)
            section_summary_map[title] = sec_summary
            section_objs.append(
                Section(
                    document_id=document_id, version=version, title=title,
                    chunk_ids=[c.chunk_id for c in section_chunks], summary=sec_summary,
                )
            )
        self.repo.add_sections(section_objs)

        doc_summary_text = self.summarize_document(list(section_summary_map.values()))
        doc_summary = DocumentSummary(
            document_id=document_id, version=version,
            summary=doc_summary_text, section_summaries=section_summary_map,
        )
        self.repo.add_document_summary(doc_summary)
        return doc_summary
