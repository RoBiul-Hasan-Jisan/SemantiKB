"""
Baseline 2: recursive character/separator chunking, in the spirit of
LangChain's RecursiveCharacterTextSplitter — recursively split on the
highest-priority separator (paragraph -> line -> sentence -> word) that
gets each piece under the size budget. Still no semantic awareness; it
just respects textual structure a bit more than pure fixed-size.
"""
from __future__ import annotations

from backend.chunking.base import BaseChunker, count_tokens
from backend.models import Chunk, ChunkingStrategy, Sentence

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text(text: str, max_tokens: int, separators: list[str]) -> list[str]:
    if count_tokens(text) <= max_tokens or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_text(text, max_tokens, rest)

    chunks: list[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if count_tokens(candidate) <= max_tokens:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if count_tokens(part) > max_tokens:
                chunks.extend(_split_text(part, max_tokens, rest))
                buf = ""
            else:
                buf = part
    if buf:
        chunks.append(buf)
    return chunks


class RecursiveCharacterChunker(BaseChunker):
    strategy = ChunkingStrategy.RECURSIVE

    def __init__(self, chunk_size_tokens: int = 250, overlap_tokens: int = 30):
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document_id: str, version: int, sentences: list[Sentence]) -> list[Chunk]:
        # reconstruct full text while keeping a sentence index map so we can
        # still attach page/section metadata to the resulting text chunks
        full_text = " ".join(s.text for s in sentences)
        raw_pieces = _split_text(full_text, self.chunk_size_tokens, SEPARATORS)

        # apply character-based overlap between consecutive pieces
        pieces_with_overlap: list[str] = []
        prev_tail = ""
        for piece in raw_pieces:
            piece_full = (prev_tail + " " + piece).strip() if prev_tail else piece
            pieces_with_overlap.append(piece_full)
            words = piece.split()
            overlap_word_count = max(1, self.overlap_tokens // 2)
            prev_tail = " ".join(words[-overlap_word_count:]) if words else ""

        chunks: list[Chunk] = []
        cursor = 0  # index into sentences, approximate mapping by text search
        sent_texts = [s.text for s in sentences]
        for idx, piece in enumerate(pieces_with_overlap):
            matched = self._match_sentences(piece, sentences, cursor)
            if not matched:
                continue
            pages = [s.page_number for s in matched]
            sections = [s.section for s in matched if s.section]
            chunks.append(
                Chunk(
                    document_id=document_id,
                    version=version,
                    section=sections[0] if sections else None,
                    page_start=min(pages),
                    page_end=max(pages),
                    sentence_ids=[s.sentence_id for s in matched],
                    text=piece.strip(),
                    token_count=count_tokens(piece),
                    strategy=self.strategy,
                    chunk_index=idx,
                )
            )
        return chunks

    @staticmethod
    def _match_sentences(piece: str, sentences: list[Sentence], start_hint: int) -> list[Sentence]:
        """Best-effort mapping of a text piece back to the sentences it overlaps,
        used only to recover page/section metadata for the chunk."""
        matched = [s for s in sentences if s.text and s.text[:20] in piece]
        return matched if matched else sentences[start_hint:start_hint + 1]
