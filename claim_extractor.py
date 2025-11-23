"""Claim extraction utilities powered by spaCy."""
from __future__ import annotations

from typing import List

import spacy
from spacy.language import Language

_MODEL_NAME = "en_core_web_sm"
_ENT_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "NORP",
    "FAC",
    "EVENT",
    "LAW",
    "WORK_OF_ART",
}
_NLP: Language | None = None


def _load_nlp() -> Language:
    """Load (and cache) the spaCy language model."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(_MODEL_NAME)
        except OSError:
            from spacy.cli import download

            download(_MODEL_NAME)
            _NLP = spacy.load(_MODEL_NAME)
    return _NLP


def _sentence_is_actionable(sent: "spacy.tokens.Span") -> bool:
    has_verb = any(token.pos_ == "VERB" for token in sent)
    long_enough = len(sent.text.strip().split()) >= 5
    return bool(has_verb and long_enough)


def extract_claims(text: str) -> List[str]:
    """Identify actionable claims from free-form text."""
    if not text or not text.strip():
        return []

    doc = _load_nlp()(text)
    claims: List[str] = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue
        if not _sentence_is_actionable(sent):
            continue

        unique_entities = sorted({ent.text.strip() for ent in sent.ents if ent.label_ in _ENT_LABELS and ent.text.strip()})
        if unique_entities:
            sentence_text = f"{sentence_text} [Entities: {', '.join(unique_entities)}]"
        claims.append(sentence_text)

    if not claims:
        claims = [text.strip()]

    return claims
