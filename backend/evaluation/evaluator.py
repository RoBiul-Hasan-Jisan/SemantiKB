"""
Evaluation harness: runs fixed / recursive / semantic chunking through the
same retrieval pipeline against the same labeled queries and reports
Precision@K, Recall@K, MRR, NDCG, answer relevance, faithfulness, chunk
count, storage footprint, and retrieval latency (requirement 12).

Ground truth format (EvalQuery): a query paired with the set of chunk_ids
(from any strategy, matched by page overlap — see `label_relevant_chunks`)
that should be considered relevant. Because chunk boundaries differ across
strategies, "relevant" is defined at the page level and mapped onto each
strategy's own chunk_ids, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

from backend.config import settings
from backend.database.repository import Repository
from backend.database.vector_store import VectorStore
from backend.embeddings.embedder import Embedder
from backend.evaluation.metrics import mean, ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k
from backend.llm.ollama_client import OllamaClient
from backend.models import ChunkingStrategy
from backend.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class EvalQuery:
    query: str
    document_id: str
    relevant_pages: list[int]  # ground-truth relevant page numbers
    reference_answer: str | None = None  # optional, for answer-relevance scoring


@dataclass
class StrategyMetrics:
    strategy: str
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    num_chunks: int = 0
    storage_bytes: int = 0
    avg_retrieval_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__


def label_relevant_chunk_ids(repo: Repository, document_id: str, version: int, strategy: ChunkingStrategy, relevant_pages: list[int]) -> set[str]:
    chunks = repo.get_chunks(document_id, version, strategy.value)
    relevant = set()
    for c in chunks:
        chunk_pages = set(range(c.page_start, c.page_end + 1))
        if chunk_pages & set(relevant_pages):
            relevant.add(c.chunk_id)
    return relevant


class Evaluator:
    def __init__(self, repo: Repository, vector_store: VectorStore, embedder: Embedder, llm: OllamaClient | None = None):
        self.repo = repo
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.retriever = HybridRetriever(repo, vector_store, embedder)

    def evaluate_strategy(self, strategy: ChunkingStrategy, eval_queries: list[EvalQuery], k: int = 5) -> StrategyMetrics:
        precisions, recalls, rrs, ndcgs, latencies = [], [], [], [], []
        answer_rel_scores, faithfulness_scores = [], []
        total_chunks = 0
        total_bytes = 0
        seen_docs = set()

        for eq in eval_queries:
            latest = self.repo.get_latest_version(eq.document_id)
            if latest is None:
                continue
            version = latest.version
            relevant_ids = label_relevant_chunk_ids(self.repo, eq.document_id, version, strategy, eq.relevant_pages)

            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(
                query=eq.query, strategy=strategy, document_ids=[eq.document_id], version=version, top_k=k,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            retrieved_ids = [r.chunk_id for r in retrieved]
            precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
            recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
            rrs.append(reciprocal_rank(retrieved_ids, relevant_ids))
            ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))

            if eq.reference_answer and self.llm is not None:
                rel, faith = self._score_answer(eq, retrieved)
                answer_rel_scores.append(rel)
                faithfulness_scores.append(faith)

            if eq.document_id not in seen_docs:
                seen_docs.add(eq.document_id)
                chunks = self.repo.get_chunks(eq.document_id, version, strategy.value)
                total_chunks += len(chunks)
                total_bytes += sum(len(c.text.encode("utf-8")) for c in chunks)

        return StrategyMetrics(
            strategy=strategy.value,
            precision_at_k=mean(precisions),
            recall_at_k=mean(recalls),
            mrr=mean(rrs),
            ndcg_at_k=mean(ndcgs),
            answer_relevance=mean(answer_rel_scores),
            faithfulness=mean(faithfulness_scores),
            num_chunks=total_chunks,
            storage_bytes=total_bytes,
            avg_retrieval_latency_ms=mean(latencies),
        )

    def _score_answer(self, eq: EvalQuery, retrieved) -> tuple[float, float]:
        """LLM-as-judge: scores 0-1 for relevance to the question and
        faithfulness to the retrieved context (proxy for hallucination rate)."""
        from backend.llm.rag_pipeline import answer_query
        response = answer_query(self.llm, eq.query, retrieved)
        judge_prompt = f"""Rate the ANSWER on two dimensions, each 0.0-1.0, given the QUESTION, \
CONTEXT, and REFERENCE.

QUESTION: {eq.query}
REFERENCE: {eq.reference_answer}
CONTEXT: {' '.join(r.text for r in retrieved)[:2000]}
ANSWER: {response.answer}

Return ONLY a JSON object like {{"relevance": 0.0, "faithfulness": 0.0}}.
- relevance: does the answer address the question and match the reference?
- faithfulness: is every claim in the answer supported by the context (no hallucination)?"""
        try:
            raw = self.llm.generate(judge_prompt, max_tokens=100)
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            return float(data.get("relevance", 0.0)), float(data.get("faithfulness", 0.0))
        except Exception:
            return 0.0, 0.0

    def compare_all(self, eval_queries: list[EvalQuery], k: int = 5) -> list[StrategyMetrics]:
        results = []
        for strategy in ChunkingStrategy:
            metrics = self.evaluate_strategy(strategy, eval_queries, k)
            results.append(metrics)
            run_id = f"eval_{strategy.value}_{int(time.time())}"
            self.repo.add_eval_run(run_id, strategy.value, metrics.to_dict())
        return results


def print_comparison_table(results: list[StrategyMetrics]) -> None:
    headers = ["strategy", "P@K", "R@K", "MRR", "NDCG", "ans_rel", "faithful", "chunks", "bytes", "latency_ms"]
    rows = [
        [r.strategy, f"{r.precision_at_k:.3f}", f"{r.recall_at_k:.3f}", f"{r.mrr:.3f}", f"{r.ndcg_at_k:.3f}",
         f"{r.answer_relevance:.3f}", f"{r.faithfulness:.3f}", str(r.num_chunks), str(r.storage_bytes),
         f"{r.avg_retrieval_latency_ms:.1f}"]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
