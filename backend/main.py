"""
FastAPI application: the HTTP surface for the Personal Knowledge Assistant.

Run with:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.chunking.base import count_tokens
from backend.config import settings
from backend.database.repository import Repository
from backend.database.vector_store import VectorStore
from backend.embeddings.embedder import get_embedder
from backend.evaluation.evaluator import EvalQuery, Evaluator
from backend.ingestion.pipeline import IngestionPipeline
from backend.llm.ollama_client import OllamaClient
from backend.llm.rag_pipeline import answer_query
from backend.models import ChatRequest, ChatResponse, ChunkingStrategy, DiffRequest
from backend.reranking.reranker import Reranker
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.versioning.version_manager import VersionManager

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal Knowledge Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- shared singletons -----------------------------------------------------------------
repo = Repository(settings.sqlite_path)
vector_store = VectorStore(settings.vector_db_path)
embedder = get_embedder(settings.embedding_model, settings.embedding_device)
llm = OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_temperature)
pipeline = IngestionPipeline(repo, vector_store, embedder, llm)
retriever = HybridRetriever(repo, vector_store, embedder)
version_manager = VersionManager(repo)
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(settings.reranker_model)
    return _reranker


@app.get("/api/health")
def health():
    return {"status": "ok", "ollama_reachable": llm.health_check()}


# --- documents ---------------------------------------------------------------------------
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), existing_document_id: str | None = None):
    ext = file.filename.lower().rsplit(".", 1)[-1]
    if ext not in ("pdf", "txt"):
        raise HTTPException(400, "Only .pdf and .txt files are supported")

    dest = Path(settings.upload_dir) / f"{uuid.uuid4().hex}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        doc = pipeline.ingest(str(dest), file.filename, existing_document_id)
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {e}")
    return {"document_id": doc.document_id, "filename": doc.filename, "status": doc.status.value}


@app.get("/api/documents")
def list_documents():
    return [d.model_dump() for d in repo.list_documents()]


@app.get("/api/documents/{document_id}/versions")
def list_versions(document_id: str):
    return [v.model_dump() for v in repo.list_versions(document_id)]


@app.get("/api/documents/{document_id}/summary")
def get_summary(document_id: str, version: int | None = None):
    v = version or (repo.get_latest_version(document_id) and repo.get_latest_version(document_id).version)
    if v is None:
        raise HTTPException(404, "No versions found")
    summary = repo.get_document_summary(document_id, v)
    if summary is None:
        raise HTTPException(404, "Summary not available yet")
    return summary.model_dump()


@app.get("/api/documents/{document_id}/chunks")
def get_chunks(document_id: str, version: int | None = None, strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC):
    v = version or (repo.get_latest_version(document_id) and repo.get_latest_version(document_id).version)
    if v is None:
        raise HTTPException(404, "No versions found")
    chunks = repo.get_chunks(document_id, v, strategy.value)
    return [c.model_dump() for c in chunks]


@app.post("/api/documents/diff")
def diff_versions(req: DiffRequest):
    return version_manager.diff_versions(req.document_id, req.version_a, req.version_b)


# --- chat / retrieval ----------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = retriever.retrieve(
        query=req.query, strategy=req.strategy, document_ids=req.document_ids,
        version=req.version, time_filter=req.time_filter, top_k=req.top_k,
    )
    if settings.use_reranker and chunks:
        chunks = get_reranker().rerank(req.query, chunks, top_k=req.top_k or settings.top_k)
    return answer_query(llm, req.query, chunks)


@app.post("/api/retrieve")
def retrieve_only(req: ChatRequest):
    chunks = retriever.retrieve(
        query=req.query, strategy=req.strategy, document_ids=req.document_ids,
        version=req.version, time_filter=req.time_filter, top_k=req.top_k,
    )
    return [c.model_dump() for c in chunks]


# --- configuration (read-only view of current tunables for the UI) -----------------
@app.get("/api/config/chunking")
def get_chunking_config():
    return {
        "similarity_threshold": settings.chunk_similarity_threshold,
        "min_size_tokens": settings.chunk_min_size_tokens,
        "max_size_tokens": settings.chunk_max_size_tokens,
        "overlap_tokens": settings.chunk_overlap_tokens,
        "prefer_paragraph_boundaries": settings.prefer_paragraph_boundaries,
    }


# --- evaluation -----------------------------------------------------------------------
@app.post("/api/evaluate")
def evaluate(eval_queries: list[dict], k: int = 5):
    """
    Body: list of {query, document_id, relevant_pages, reference_answer?}
    Runs fixed / recursive / semantic strategies through identical
    retrieval + (optionally) generation and returns comparison metrics.
    """
    queries = [EvalQuery(**q) for q in eval_queries]
    evaluator = Evaluator(repo, vector_store, embedder, llm)
    results = evaluator.compare_all(queries, k=k)
    return [r.to_dict() for r in results]


@app.get("/api/evaluate/history")
def evaluation_history():
    return repo.list_eval_runs()
