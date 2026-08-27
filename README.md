# SemantiKB — Smart Semantic Chunking

A local, privacy-friendly RAG (Retrieval-Augmented Generation) system for uploading documents and asking questions about them. Its core innovation is **context-aware semantic chunking**: instead of splitting documents into fixed-size blocks, it uses sentence embeddings to detect where the *topic* actually changes, and cuts there.

Everything runs locally — Hugging Face Sentence Transformers for embeddings, ChromaDB for vector storage, and Ollama for the LLM. No paid APIs are required.

### Pipeline

```mermaid
flowchart LR
    A[📄 Upload] --> B[Parse]
    B --> C[Sentence<br/>Segmentation]
    C --> D[Sentence<br/>Embeddings]
    D --> E[Semantic Boundary<br/>Detection]
    E --> F[Context-Aware<br/>Chunking]
    F --> G[Hierarchical<br/>Summarization]
    G --> H[(Vector Store)]
    H --> I[Hybrid Retrieval<br/><i>+ optional reranking</i>]
    I --> J[Ollama LLM]
    J --> K[✅ Grounded Answer<br/>+ Citations]

    classDef ingest fill:#e0f2fe,stroke:#0369a1,stroke-width:1px,color:#0c4a6e
    classDef chunk fill:#fef3c7,stroke:#b45309,stroke-width:1px,color:#78350f
    classDef store fill:#ede9fe,stroke:#6d28d9,stroke-width:1px,color:#4c1d95
    classDef answer fill:#dcfce7,stroke:#15803d,stroke-width:1px,color:#14532d

    class A,B,C,D ingest
    class E,F,G chunk
    class H store
    class I,J,K answer
```

*Documents move through ingestion (blue) into semantic chunking (amber), land in the vector store (purple), and are retrieved and answered (green) — all on-device.*

---

## Table of Contents

- [Why Semantic Chunking?](#why-semantic-chunking)
- [Architecture](#architecture)
- [Setup](#setup)
- [Using the Evaluation Module](#using-the-evaluation-module)
- [Configuration Reference](#configuration-reference)
- [Privacy & Cost](#privacy--cost)
- [Known Limitations](#known-limitations)

---

## Why Semantic Chunking?

### The Problem with Fixed-Size Chunking

Fixed-size (or fixed-token) chunking splits a document every *N* tokens, regardless of what's being said at that point. If a paragraph explaining a policy's exception clause happens to straddle the 250-token mark, the explanation and its exception get separated into two different chunks. At retrieval time, a query about the exception may only retrieve the first half — the model then either answers incompletely or, worse, confidently states the general rule without the exception, because the exception simply isn't in its context window. The chunk boundary is a property of the *counter*, not of the *content*.

Recursive character/separator chunking (e.g. LangChain's `RecursiveCharacterTextSplitter`) improves on this by preferring to split on paragraph or sentence boundaries when possible — but it still has no notion of *meaning*. Two adjacent paragraphs about completely different subjects will happily be merged into one chunk if they fit the size budget, while a single coherent argument that runs slightly over budget gets cut in half anyway.

### How Semantic Similarity Identifies Topic Transitions

Sentence embeddings place semantically related sentences close together in vector space. If every sentence in a document is embedded with a Sentence Transformer and the cosine similarity is computed between each pair of *consecutive* sentences, the result is a similarity signal that stays high while the discourse continues on the same topic and *drops* exactly at the point where the topic shifts — a new section starts, an example ends and a new point begins, a policy's general rule transitions into an exception, etc.

The system's `SemanticChunker` (`backend/chunking/semantic_chunker.py`) implements this directly:

1. Embed every sentence in the document (`all-MiniLM-L6-v2` by default).
2. Compute consecutive-sentence cosine similarity across the whole document.
3. Wherever similarity falls below `CHUNK_SIMILARITY_THRESHOLD`, mark a *candidate* chunk boundary.
4. Only honor a candidate boundary once the chunk-so-far is at least `CHUNK_MIN_SIZE_TOKENS` — this prevents the chunker from producing lots of tiny, fragmentary chunks on naturally choppy prose (dialogue, lists, etc.).
5. Force a split regardless of similarity if a chunk would otherwise exceed `CHUNK_MAX_SIZE_TOKENS` — semantic chunking still respects a hard ceiling so chunks stay embeddable and fit comfortably in the LLM's context window.
6. If `PREFER_PARAGRAPH_BOUNDARIES` is enabled, a nearby paragraph/heading boundary already present in the source is preferred over a purely embedding-derived cut point, since document authors' own structure is a strong (free) signal.
7. A configurable number of trailing sentences (`CHUNK_OVERLAP_TOKENS`) is carried into the next chunk so that retrieval doesn't lose context right at a boundary.

### How the Threshold Affects Chunk Boundaries

`CHUNK_SIMILARITY_THRESHOLD` is a value between 0 and 1 (default `0.65`):

| Threshold | Effect | Best for |
|---|---|---|
| **Lower** (e.g. `0.4`) | Splits only on very strong topic shifts → fewer, larger chunks | Narrative or loosely-structured text where broad context per chunk is useful |
| **Higher** (e.g. `0.8`) | Splits on subtler shifts in wording and emphasis → more, smaller, tightly-focused chunks | Reference documents (policies, specs) where users ask narrow, specific questions |

Because the threshold, minimum size, and maximum size are all environment variables (see `.env.example`), the chunker can be retuned per corpus without touching code, and the evaluation module lets you empirically find the threshold that maximizes retrieval quality for your documents.

### Why Semantic Chunking May Improve Retrieval

Retrieval quality depends on each stored chunk being a coherent, self-contained unit of meaning:

- If a chunk mixes two unrelated ideas, its embedding is a blurry average of both, and it will rank poorly for queries about *either* idea.
- If a chunk cuts a single idea in half, neither half's embedding fully represents the idea, and a query about it may miss both halves or retrieve an incomplete one.

Semantic chunking directly targets this problem by aligning chunk boundaries with actual topic boundaries, so:

- Each chunk's embedding is a more faithful representation of a single, complete idea → higher **precision** (retrieved chunks are actually about the query) and higher **recall** (the whole relevant passage is captured in one chunk, not split across two).
- The LLM sees complete arguments rather than fragments, reducing the odds of a grounded-but-incomplete or subtly wrong answer.

This system doesn't just assert that semantic chunking is better — it ships an evaluation harness (`backend/evaluation/`) that runs fixed, recursive, and semantic chunking through *the same* retrieval pipeline against *the same* labeled queries, and reports Precision@K, Recall@K, MRR, NDCG, answer relevance, faithfulness, chunk count, storage footprint, and retrieval latency side by side — so the claim can be checked quantitatively on your own documents.

---

## Architecture

```
backend/
  ingestion/       PDF/TXT parsing, sentence segmentation, ingestion pipeline
  chunking/         Fixed / recursive / semantic chunkers + factory
  embeddings/       Sentence Transformer wrapper
  retrieval/        BM25, hybrid vector+keyword retrieval, temporal resolution
  reranking/        Optional cross-encoder reranker
  summarization/    Hierarchical chunk → section → document summaries
  versioning/       Document version registration, temporal lookup, diffing
  llm/              Ollama client + grounded RAG prompt/answer construction
  evaluation/       IR metrics + strategy comparison harness
  database/         SQLite schema/repository, ChromaDB vector store wrapper
  tests/            Unit tests
  config.py         All tunables, loaded from environment variables
  models.py         Pydantic schema shared across the system
  main.py           FastAPI app

frontend/
  components/       DocumentSidebar, ChatPanel, VersionHistory,
                     DocumentInsightsPanel, EvaluationPanel
  pages/App.js       Top-level page wiring the components together
  services/api.js    Fetch wrapper around the backend REST API
  index.html / styles.css
```

Data flows **document → version → section → chunk → sentences** at every layer (SQLite schema, Pydantic models, and the summarizer's hierarchy all mirror this).

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with a model pulled, e.g.:
  ```bash
  ollama pull llama2.1
  ollama serve
  ```
- No GPU required — CPU embedding works fine for `all-MiniLM-L6-v2`. Set `EMBEDDING_DEVICE=cuda` in `.env` if you have one.

### Backend

```bash
cd pka
python -m venv .venv && source .venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt

cp .env.example .env                                 # adjust as needed

# NLTK's sentence tokenizer data (optional — falls back to a regex
# splitter automatically if this is skipped or offline):
python -c "import nltk; nltk.download('punkt_tab')"

python -m uvicorn backend.main:app --reload --port 8000
```

The first request that touches the embedding model will download `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face (~90MB) and cache it locally; after that, everything runs offline.

Run the test suite:

```bash
pytest backend/tests -v
```

### Frontend

The frontend is a dependency-free vanilla JS app (ES modules, no build step). Serve it with any static file server, e.g.:

```bash
cd frontend
python -m http.server 5173
```

Then open `http://localhost:5173`. If your backend isn't on `localhost:8000`, edit the `window.PKA_API_BASE_URL` value in `frontend/index.html`.

---

## Using the Evaluation Module

`POST /api/evaluate` accepts a list of labeled queries:

```json
[
  {
    "query": "What is the refund policy?",
    "document_id": "doc_abc123",
    "relevant_pages": [2, 3],
    "reference_answer": "Refunds are available within 30 days of purchase."
  }
]
```

It runs each of the three chunking strategies through identical retrieval (and, when a `reference_answer` is supplied, LLM-judged answer relevance/faithfulness), and returns a metrics comparison. Results are also stored in SQLite (`eval_runs` table) and viewable via `GET /api/evaluate/history`. The **Evaluation** tab in the UI wraps this endpoint with a simple form.

---

## Configuration Reference

All tunables live in `.env` (see `.env.example` for the full list and defaults), including:

- Semantic chunking threshold, min/max size, and overlap
- Baseline chunker sizes (fixed / recursive)
- Retrieval mode (`vector` / `bm25` / `hybrid`) and hybrid weighting
- Optional reranker toggle
- Ollama model/host
- Hierarchical summarization toggle

---

## Privacy & Cost

| Component | Runs | Notes |
|---|---|---|
| Embeddings | Local (Sentence Transformers) | No API key required |
| Vector DB | Local (ChromaDB) | File-based persistence under `./data/chroma` |
| LLM | Local (Ollama) | No data leaves your machine |

No mandatory paid APIs are used anywhere in the pipeline.

---

## Known Limitations

- **PDF section/heading detection** is heuristic (regex-based), not full layout analysis — works well for typically-formatted reports/policies, less well for heavily multi-column or image-heavy PDFs.
- **Recursive chunker metadata** (page/section attachment) is best-effort text matching rather than an exact index, since it reconstructs from raw text rather than tracking sentence offsets directly (the semantic and fixed chunkers track this precisely).
- **BM25 index** is rebuilt in-memory per query for simplicity; for very large corpora, a persistent inverted index would be preferable.
