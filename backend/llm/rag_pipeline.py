"""
The final RAG step: takes retrieved (and optionally reranked) chunks,
builds a strictly-grounded prompt, calls Ollama, and returns an answer
with structured citations.
"""
from __future__ import annotations

from backend.llm.ollama_client import OllamaClient
from backend.models import ChatResponse, Citation, RetrievedChunk

SYSTEM_PROMPT = """You are a careful knowledge-base assistant. Follow these rules strictly:

1. Answer ONLY using the provided context. Do not use outside knowledge.
2. Do NOT hallucinate facts, names, numbers, or dates that are not in the context.
3. If the answer is not present in the context, explicitly say: \
"I don't have enough information in the provided documents to answer that."
4. When you state a fact, mention which source it came from using its citation \
tag, e.g. [Source 1].
5. Respect any document version or time constraints given in the context — if the \
context is scoped to a specific version or year, answer only for that scope and say so.
6. Be concise and factual. Do not pad the answer with filler."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        section = f", section: {c.section}" if c.section else ""
        blocks.append(
            f"[Source {i}] Document: {c.filename} (version {c.version}, {pages}{section})\n{c.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_context(chunks) if chunks else "(no relevant context found)"
    return f"""Context:
{context}

Question: {query}

Answer (cite sources like [Source 1]):"""


def answer_query(llm: OllamaClient, query: str, chunks: list[RetrievedChunk], max_tokens: int = 512) -> ChatResponse:
    prompt = build_prompt(query, chunks)
    if not chunks:
        answer = "I don't have enough information in the provided documents to answer that."
    else:
        answer = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens).strip()

    citations = [
        Citation(
            document_name=c.filename,
            version=c.version,
            page=(f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"),
            chunk_id=c.chunk_id,
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, citations=citations, retrieved_chunks=chunks)
