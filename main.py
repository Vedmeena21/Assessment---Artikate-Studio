"""CLI entrypoint for the fact-checker pipeline."""
from __future__ import annotations

import json
import sys

from pipeline import run_pipeline


def main() -> int:
    text = input("Enter the text you want to fact-check:\n> ").strip()
    if not text:
        print("No text provided. Exiting.")
        return 1

    k_raw = input("Enter number of evidence facts to retrieve (default 3): ").strip()
    k = int(k_raw) if k_raw else 3

    result = run_pipeline(text, k=k)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
