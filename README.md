# WorldQuantBrain-Agent

A local multi-agent research pipeline for generating WorldQuant Brain alpha ideas from consultant forum PDFs.

This project uses CrewAI agents, a local/remote OpenAI-compatible LLM endpoint, and a Chroma vector database built from PDF documents.

## What this project does

- Ingests WorldQuant-related forum PDFs into a vector database.
- Retrieves relevant discussion content with semantic search.
- Runs a sequential multi-agent workflow to:
  - research forum content,
  - propose low-correlation alpha ideas,
  - convert the best idea into a Brain expression,
  - validate and format final output.
- Writes run logs to the `logs` folder.

## Project structure

- `wqbagent_v2.py`: main pipeline script with logging + retrieval + multi-agent workflow.
- `wqbagent_v1.py`: earlier pipeline version.
- `wqbagent_test.py`: lightweight test runner for terminal/color/log pipeline checks.
- `scripts/wqbagent.bat`: Windows batch launcher for main run.
- `scripts/wqbagent_test.bat`: Windows batch launcher for test run.
- `scripts/wqbagent_pws.ps1`: PowerShell launcher variant.
- `TODO.md`: short project todo notes.

## Tech stack

- Python
- CrewAI
- LangChain + Chroma
- HuggingFace embeddings (`BAAI/bge-m3`)
- PDF loaders from `langchain_community`

## Prerequisites

- Windows (project scripts are currently Windows-focused).
- Python 3.10+ recommended.
- A virtual environment (example folder in this repo: `wqbagentvenv`).
- Access to your LLM endpoint (OpenAI-compatible API).
- Forum PDF data on local disk.

## Quick start

### 1) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv wqbagentvenv
.\wqbagentvenv\Scripts\Activate.ps1
```

CMD:

```bat
python -m venv wqbagentvenv
wqbagentvenv\Scripts\activate.bat
```

### 2) Install dependencies

No `requirements.txt` is tracked yet, so install from imports used by the scripts:

```powershell
pip install crewai langchain-chroma langchain-text-splitters langchain-community langchain-huggingface chromadb pypdf sentence-transformers
```

Optional (used by script pipelines that convert ANSI logs to HTML):

```powershell
pip install ansi2html rich
```

### 3) Configure paths and credentials

Update values in `wqbagent_v2.py` for your machine:

- `BASE_DIR`
- `WQB_CONSULTANT_FORUM_PDF_PATH`
- cache and log folders derived from `BASE_DIR`
- LLM settings (`model`, `base_url`, `api_key`)

### 4) Build vector DB (first run only)

In `wqbagent_v2.py`, uncomment this line once:

```python
# vectorstore = ingest_forum()
```

Then run once to create embeddings and persist Chroma DB.  
After DB is created, comment it back again to avoid repeated ingestion.

### 5) Run the agent

Direct Python run:

```powershell
python .\wqbagent_v2.py
```

Or use launcher scripts:

```bat
.\scripts\wqbagent.bat
```

Test run:

```bat
.\scripts\wqbagent_test.bat
```

## Output

- Terminal output with CrewAI verbose logs.
- Timestamped logs under `logs/`.
- Some launch scripts also produce HTML logs.

## Before pushing to GitHub (important)

1. Remove hardcoded secrets from code (especially API keys).
2. Rotate/revoke any previously exposed API keys.
3. Prefer environment variables for sensitive values:
   - `OPENAI_API_KEY` (or your provider key)
   - endpoint URL variables if needed
4. Keep local data/cache directories out of git (`logs`, `cache`, `wqbagentvenv`, vector DB folders).

## Suggested next improvements

- Add a proper `requirements.txt` or `pyproject.toml`.
- Move all paths/secrets into `.env` + config module.
- Add one-click setup script and basic tests for ingestion/retrieval flow.
- Document expected PDF data format and folder layout.

## License

No license file is currently included. Add a `LICENSE` file before open-sourcing.

