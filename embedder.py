"""Sentence embedding utilities using SentenceTransformers."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Return embeddings for input texts."""
    texts = [text.strip() for text in texts if text and text.strip()]
    if not texts:
        return []
    model = load_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return [embedding.tolist() for embedding in embeddings]
