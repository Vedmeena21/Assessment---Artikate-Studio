"""End-to-end fact-checking pipeline orchestration."""
from __future__ import annotations

from typing import Dict, List

from claim_extractor import extract_claims
from embedder import embed_texts
from utils import load_facts
from vector_store import build_index, search
from verifier import verify_claim


def run_pipeline(text: str, k: int = 3) -> Dict[str, object]:
    """Run extraction, retrieval, and verification over the supplied text."""
    if not text or not text.strip():
        raise ValueError("Input text is required")

    facts = load_facts()
    build_index(facts)

    claims = extract_claims(text)
    if not claims:
        return {"input_text": text, "claims": []}

    embeddings = embed_texts(claims)
    results: List[Dict[str, object]] = []
    for claim, embedding in zip(claims, embeddings):
        retrieved = search(embedding, k=k)
        verdict = verify_claim(claim, retrieved)
        results.append(
            {
                "claim": claim,
                "evidence": retrieved,
                "analysis": verdict,
            }
        )

    return {"input_text": text, "claims": results}
