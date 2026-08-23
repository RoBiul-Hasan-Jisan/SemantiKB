"""
Document parsing: extracts text while preserving page numbers and, where
detectable, section/heading information.

PDF parsing uses pypdf (pure-python, no system deps). Section detection is
a lightweight heuristic (short lines, ALL CAPS / Title Case, numbered
headings) rather than full layout analysis — good enough to give the
chunker paragraph/heading boundaries to prefer, per spec requirement 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str
    section: str | None


HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,\-:&/]{2,80})$"
)


def _guess_section(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line.isupper() and len(line.split()) <= 12:
        return True
    if HEADING_RE.match(line) and len(line.split()) <= 12 and not line.endswith("."):
        return True
    return False


def _extract_sections(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Return the running section title for a page's text plus any headings found."""
    headings = []
    for line in text.splitlines():
        if _guess_section(line):
            headings.append(line.strip())
    return (headings[0] if headings else None), headings


def parse_pdf(file_path: str) -> list[ParsedPage]:
    reader = PdfReader(file_path)
    pages: list[ParsedPage] = []
    current_section: str | None = None
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        heading, all_headings = _extract_sections(raw)
        if heading:
            current_section = heading
        pages.append(ParsedPage(page_number=i, text=raw.strip(), section=current_section))
    return pages


def parse_txt(file_path: str, chars_per_page: int = 3000) -> list[ParsedPage]:
    """
    Plain text has no native pagination, so we synthesize pages by
    splitting on paragraph boundaries near a target character budget.
    This keeps the same page_number/section metadata contract as PDFs.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    paragraphs = re.split(r"\n\s*\n", content)
    pages: list[ParsedPage] = []
    buf: list[str] = []
    buf_len = 0
    page_num = 1
    current_section: str | None = None

    def flush():
        nonlocal buf, buf_len, page_num
        if buf:
            text = "\n\n".join(buf)
            heading, _ = _extract_sections(text)
            pages.append(ParsedPage(page_number=page_num, text=text.strip(),
                                     section=heading or current_section))
            page_num += 1
            buf, buf_len = [], 0

    for para in paragraphs:
        heading, _ = _extract_sections(para)
        if heading:
            current_section = heading
        buf.append(para)
        buf_len += len(para)
        if buf_len >= chars_per_page:
            flush()
    flush()
    if not pages:
        pages = [ParsedPage(page_number=1, text=content.strip(), section=None)]
    return pages


def parse_document(file_path: str, filename: str) -> list[ParsedPage]:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return parse_pdf(file_path)
    elif ext == "txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: pdf, txt")
