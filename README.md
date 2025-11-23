# FactChecker Pipeline

A modular, production-ready fact-checking workflow that extracts actionable claims, retrieves semantically similar government policy statements, and verifies each claim using GPT analysis.

## Features
- **Claim Extraction** via spaCy sentence parsing and entity heuristics
- **Embedding Layer** powered by SentenceTransformers `all-MiniLM-L6-v2`
- **Vector Retrieval** backed by FAISS with persistent metadata
- **LLM Verification** using local Ollama models with structured JSON verdicts
- **CLI Interface** for quick experimentation and automation

## Setup
1. Python 3.10+
2. [Install Ollama](https://ollama.com) (`brew install ollama` on macOS)
3. Pull the default model: `ollama pull mistral`
4. Create and activate a virtual environment
5. Install system dependencies for FAISS (macOS/Linux homebrew/apt packages if needed)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Environment Variables
Create a `.env` file next to `main.py` (optional overrides):

```
LLM_MODEL=mistral
```

## Running
```bash
python main.py --text "The Ministry of Finance launched a green sovereign bond in 2024" --k 3
```

### Sample Input/Output
**Input:**
```
The Ministry of Finance launched a green sovereign bond in 2024 to fund renewable infrastructure.
```

**Output:**
```json
{
  "input_text": "The Ministry of Finance launched a green sovereign bond in 2024 to fund renewable infrastructure.",
  "claims": [
    {
      "claim": "The Ministry of Finance launched a green sovereign bond in 2024 to fund renewable infrastructure.",
      "evidence": [
        "The Ministry of Finance launched a green sovereign bond worth $5 billion in April 2024.",
        "The Ministry of Power committed to 60% renewable energy generation by 2030 under the National Energy Mission.",
        "Startup India 3.0 introduced zero capital gains tax for DPIIT-recognized startups until 2028."
      ],
      "analysis": {
        "verdict": "TRUE",
        "reasoning": "The primary evidence confirms a green sovereign bond launched by the Ministry of Finance in 2024, matching the claim context.",
        "evidence": [
          "The Ministry of Finance launched a green sovereign bond worth $5 billion in April 2024."
        ]
      }
    }
  ]
}
```

## Pipeline Diagram
```
┌──────────────┐    ┌────────────────┐    ┌────────────────┐    ┌─────────────────┐
│ Input Text   │ -> │ Claim Extractor │ -> │ Embed + FAISS  │ -> │ GPT Verifier     │
└──────────────┘    └────────────────┘    └────────────────┘    └─────────────────┘
                        sentences & ents       semantic search        verdict JSON
```

## Data & Indexing
- `data/facts.csv` ships with 30 seed government policy statements.
- `utils.save_facts_metadata` stores fact text alongside FAISS indexes to keep IDs stable.
- Use `vector_store.save_index()` and `vector_store.load_index()` when persisting between runs (optional for CLI flow).

## Development Tips
- Extend `claim_extractor.py` for domain-specific heuristics.
- Swap in a larger embedding model by editing `_MODEL_NAME` in `embedder.py`.
- Add automated tests around `pipeline.run_pipeline` for regression safety.
