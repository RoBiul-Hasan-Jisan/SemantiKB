"""
Local embedding generation via Hugging Face Sentence Transformers.
No API key, no network calls at inference time (model is downloaded once
and cached by the `sentence-transformers` / `huggingface_hub` libraries).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension))
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )

    def embed_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.embed([text], normalize=normalize)[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


@lru_cache(maxsize=4)
def get_embedder(model_name: str, device: str = "cpu") -> Embedder:
    """Process-wide cache so the (potentially large) model is loaded once."""
    return Embedder(model_name, device)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, assumes not pre-normalized."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_consecutive_similarity(embeddings: np.ndarray) -> list[float]:
    """
    Cosine similarity between each embedding[i] and embedding[i+1].
    Returns a list of length len(embeddings) - 1.
    Assumes embeddings are normalized (as produced by `embed()` above),
    so this is a plain dot product.
    """
    if len(embeddings) < 2:
        return []
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return sims.tolist()
