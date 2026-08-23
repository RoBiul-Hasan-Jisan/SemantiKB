import numpy as np
import pytest

from backend.chunking.fixed_chunker import FixedSizeChunker
from backend.chunking.recursive_chunker import RecursiveCharacterChunker
from backend.chunking.semantic_chunker import SemanticChunker, SemanticChunkerConfig
from backend.models import ChunkingStrategy, Sentence


def make_sentences(texts, section=None):
    return [
        Sentence(document_id="d1", version=1, page_number=1, section=section, index_in_doc=i, text=t)
        for i, t in enumerate(texts)
    ]


class FakeEmbedder:
    """Deterministic fake embedder: sentences with the same 'topic' word get
    near-identical vectors; different topics get orthogonal vectors, so we
    can test the boundary-detection logic without loading a real model."""

    def __init__(self, topic_map: dict[str, int], dim: int = 8):
        self.topic_map = topic_map
        self.dim = dim

    def embed(self, texts, batch_size=32, normalize=True):
        vecs = []
        for t in texts:
            topic_idx = 0
            for key, idx in self.topic_map.items():
                if key in t:
                    topic_idx = idx
                    break
            v = np.zeros(self.dim)
            v[topic_idx % self.dim] = 1.0
            vecs.append(v)
        return np.array(vecs)


def test_fixed_chunker_respects_size_budget():
    texts = [f"This is sentence number {i} with some words." for i in range(30)]
    sentences = make_sentences(texts)
    chunker = FixedSizeChunker(chunk_size_tokens=30, overlap_tokens=5)
    chunks = chunker.chunk("d1", 1, sentences)
    assert len(chunks) > 1
    for c in chunks:
        assert c.strategy == ChunkingStrategy.FIXED
        assert c.token_count <= 40  # budget + a little slack from the last sentence


def test_recursive_chunker_produces_chunks():
    text_block = "Paragraph one sentence one. Paragraph one sentence two.\n\nParagraph two sentence one. Paragraph two sentence two."
    sentences = make_sentences(text_block.split(". "))
    chunker = RecursiveCharacterChunker(chunk_size_tokens=10, overlap_tokens=2)
    chunks = chunker.chunk("d1", 1, sentences)
    assert len(chunks) >= 1
    assert all(c.strategy == ChunkingStrategy.RECURSIVE for c in chunks)


def test_semantic_chunker_splits_on_topic_change():
    texts = (
        ["Cats are small domesticated mammals. Cats like to sleep a lot. Cats often chase mice."]
        + ["Stock markets fluctuate daily. Stock markets are driven by supply and demand. Stock markets can be volatile."]
    )
    sentences = make_sentences(
        [s.strip() + "." for group in texts for s in group.split(". ") if s.strip()]
    )
    embedder = FakeEmbedder(topic_map={"Cats": 0, "Stock": 1})
    config = SemanticChunkerConfig(similarity_threshold=0.5, min_size_tokens=1, max_size_tokens=1000, overlap_tokens=0)
    chunker = SemanticChunker(embedder, config)
    chunks = chunker.chunk("d1", 1, sentences)

    assert len(chunks) == 2
    assert "Cats" in chunks[0].text
    assert "Stock" in chunks[1].text


def test_semantic_chunker_respects_max_size_even_without_topic_change():
    texts = [f"Cats fact number {i}." for i in range(20)]
    sentences = make_sentences(texts)
    embedder = FakeEmbedder(topic_map={"Cats": 0})
    config = SemanticChunkerConfig(similarity_threshold=0.1, min_size_tokens=1, max_size_tokens=20, overlap_tokens=0)
    chunker = SemanticChunker(embedder, config)
    chunks = chunker.chunk("d1", 1, sentences)
    assert len(chunks) > 1  # forced splits despite no semantic drop
