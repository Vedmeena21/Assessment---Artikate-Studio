"""LLM-powered claim verification (local Ollama)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = "mistral"
ALLOWED_VERDICTS = {
    "true": "True",
    "false": "False",
    "unverifiable": "Unverifiable",
}
_PROMPT_HEADER = (
    "You are a fact-checking engine. Analyze each CLAIM using the retrieved EVIDENCE.\n"
    "Classify strictly as one of: True (supported), False (contradicted), or Unverifiable (insufficient).\n"
    "Return JSON only with the keys 'verdict', 'reasoning', and 'evidence'."
)


def _get_model_name() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def _format_prompt(claim: str, retrieved: List[str]) -> str:
    evidence_text = "\n".join(f"{idx + 1}. {fact}" for idx, fact in enumerate(retrieved)) or "No evidence available."
    return (
        f"{_PROMPT_HEADER}\n\n"
        "CLAIM:\n"
        f"{claim}\n\n"
        "EVIDENCE:\n"
        f"{evidence_text}\n\n"
        "Respond ONLY with JSON in the form:\n"
        "{\n  \"verdict\": \"True|False|Unverifiable\",\n  \"reasoning\": \"...\",\n  \"evidence\": [\"fact snippet\"]\n}"
    )


def _run_ollama(prompt: str) -> str:
    model = _get_model_name()
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama CLI not found. Install via 'brew install ollama'.") from exc

    try:
        stdout, stderr = process.communicate(prompt, timeout=120)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise RuntimeError("Ollama response timed out.") from exc

    if process.returncode != 0:
        raise RuntimeError(f"Ollama error: {stderr.strip() or 'Unknown error'}")

    return stdout.strip()


def _normalize_result(payload: str, fallback: Dict[str, object]) -> Dict[str, object]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return fallback

    verdict_raw = str(parsed.get("verdict", "")).strip().lower()
    verdict = ALLOWED_VERDICTS.get(verdict_raw, "Unverifiable")
    reasoning = str(parsed.get("reasoning", "")) or fallback["reasoning"]
    evidence = parsed.get("evidence")
    if not isinstance(evidence, list):
        evidence = fallback["evidence"]

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "evidence": evidence,
    }


def verify_claim(claim: str, retrieved: List[str]) -> Dict[str, object]:
    """Verify a single claim against retrieved evidence via local LLM."""
    if not claim:
        raise ValueError("Claim is required")

    if not retrieved:
        return {
            "verdict": "Unverifiable",
            "reasoning": "No supporting evidence could be retrieved.",
            "evidence": [],
        }

    prompt = _format_prompt(claim, retrieved)
    fallback = {
        "verdict": "Unverifiable",
        "reasoning": "Model response missing or malformed.",
        "evidence": retrieved,
    }

    try:
        response_text = _run_ollama(prompt)
    except RuntimeError as exc:
        return {
            "verdict": "Unverifiable",
            "reasoning": str(exc),
            "evidence": retrieved,
        }

    return _normalize_result(response_text or json.dumps(fallback), fallback)
