"""
Sentence segmentation. Uses NLTK's punkt tokenizer when available (better
handling of abbreviations, decimals, etc.); falls back to a regex
splitter so the system still works with zero extra downloads.
"""
from __future__ import annotations

import re

from backend.ingestion.document_parser import ParsedPage
from backend.models import Sentence

_ABBREVS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs", "etc", "e.g", "i.e", "Inc", "Ltd", "Co", "St", "No")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TRAILING_ABBREV_RE = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVS) + r")\.$")


def _nltk_available() -> bool:
    try:
        import nltk  # noqa: F401
        from nltk.tokenize import sent_tokenize  # noqa: F401
        sent_tokenize("Warm up check.")
        return True
    except Exception:
        return False


_NLTK_OK = None


def split_sentences(text: str) -> list[str]:
    global _NLTK_OK
    text = text.strip()
    if not text:
        return []
    if _NLTK_OK is None:
        _NLTK_OK = _nltk_available()
    if _NLTK_OK:
        from nltk.tokenize import sent_tokenize
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    # regex fallback: split, then re-merge any split that happened right
    # after a known abbreviation (e.g. "Dr." should not end a sentence)
    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    merged: list[str] = []
    for part in raw_parts:
        if merged and _TRAILING_ABBREV_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def segment_pages(document_id: str, version: int, pages: list[ParsedPage]) -> list[Sentence]:
    sentences: list[Sentence] = []
    idx = 0
    for page in pages:
        # split on paragraphs first so section headings don't get merged
        # into surrounding sentences
        for para in re.split(r"\n\s*\n", page.text):
            para = para.strip()
            if not para:
                continue
            for s in split_sentences(para):
                sentences.append(
                    Sentence(
                        document_id=document_id,
                        version=version,
                        page_number=page.page_number,
                        section=page.section,
                        index_in_doc=idx,
                        text=s,
                    )
                )
                idx += 1
    return sentences
