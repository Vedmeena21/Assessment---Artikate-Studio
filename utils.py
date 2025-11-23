"""Utility helpers for the fact-checker pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
FACTS_PATH = DATA_DIR / "facts.csv"
META_PATH = DATA_DIR / "facts_meta.json"


def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_facts(csv_path: Path | str = FACTS_PATH) -> List[str]:
    """Load fact statements from the CSV file."""
    ensure_data_dir()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Facts file not found at {path}")

    df = pd.read_csv(path)
    if "fact" not in df.columns:
        raise ValueError("facts.csv must contain a 'fact' column")

    facts = df["fact"].dropna().astype(str).tolist()
    LOGGER.debug("Loaded %d facts", len(facts))
    return facts


def save_facts_metadata(facts: List[str]) -> None:
    """Persist facts metadata so the vector store can map results to statements."""
    ensure_data_dir()
    META_PATH.write_text(json.dumps({"facts": facts}, indent=2), encoding="utf-8")


def load_facts_metadata() -> List[str]:
    """Load facts metadata saved alongside the FAISS index."""
    if META_PATH.exists():
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        return data.get("facts", [])
    return []
