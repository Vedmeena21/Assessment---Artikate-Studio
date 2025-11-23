"""FAISS vector store utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from embedder import embed_texts
from utils import DATA_DIR, ensure_data_dir, load_facts_metadata, save_facts_metadata

_INDEX: faiss.Index | None = None
_FACTS: List[str] = []
_INDEX_PATH = DATA_DIR / "index.faiss"


def _get_dimension(embeddings: Sequence[Sequence[float]]) -> int:
    if not embeddings:
        raise ValueError("Embeddings required to determine dimension")
    return len(embeddings[0])


def build_index(facts: Sequence[str]) -> None:
    """Build an in-memory FAISS index for provided facts."""
    global _INDEX, _FACTS
    if not facts:
        raise ValueError("Cannot build index without facts")

    embeddings = embed_texts(facts)
    if not embeddings:
        raise ValueError("Embedding model produced no embeddings")

    dim = _get_dimension(embeddings)
    index = faiss.IndexFlatIP(dim)
    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)

    _INDEX = index
    _FACTS = list(facts)
    save_facts_metadata(_FACTS)
    save_index(_INDEX_PATH)


def save_index(path: str | Path = _INDEX_PATH) -> None:
    """Persist the FAISS index to disk."""
    ensure_data_dir()
    if _INDEX is None:
        raise RuntimeError("No index to save. Call build_index first.")
    faiss.write_index(_INDEX, str(path))


def load_index(path: str | Path = _INDEX_PATH) -> None:
    """Load the FAISS index and associated metadata."""
    global _INDEX, _FACTS
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found at {path}")
    _INDEX = faiss.read_index(str(path))
    _FACTS = load_facts_metadata()
    if not _FACTS:
        raise RuntimeError("Missing facts metadata. Rebuild the index.")


def _ensure_index_loaded() -> faiss.Index:
    if _INDEX is None:
        raise RuntimeError("Vector index not loaded. Call build_index or load_index first.")
    return _INDEX


def search(query_embedding: Sequence[float], k: int = 3) -> List[str]:
    """Return top-k fact strings for the provided query embedding."""
    index = _ensure_index_loaded()
    if not _FACTS:
        raise RuntimeError("No facts metadata loaded.")
    vector = np.array([query_embedding], dtype=np.float32)
    _, indices = index.search(vector, min(k, len(_FACTS)))
    hits: List[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(_FACTS):
            hits.append(_FACTS[idx])
    return hits
