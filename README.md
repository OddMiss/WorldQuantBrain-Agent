# WorldQuantBrain-Agent

Local CrewAI-based toolkit for building embeddings from WorldQuant Brain consultant materials and experimenting with multi-agent alpha idea generation. The repo includes a standalone embedding builder, test utilities, and notebooks for interactive development.

## Repository contents

- `wqbagent_embedding.py` — builds/updates a Chroma vector store from PDF/text sources and exposes the `retrieve_text_data` tool.
- `wqbagent_output_test.py` — minimal CrewAI pipeline for validating terminal colors/logging and LLM connectivity.
- `wqbquant_searchtool_test.py` — health check helper for search/retrieval tools.
- `wqbagent-v2.ipynb`, `wqbagent-terminal.ipynb` — notebooks for interactive development and experiments.
- `releases/` — archived v1 artifacts.
- `scripts/` — Windows batch/PowerShell launchers (paths and entrypoints need to be customized).
- `requirements.txt` — Python dependencies.

## Prerequisites

- Python 3.10+
- Windows recommended for the provided launch scripts (they can be adapted for other OSes)
- Access to an OpenAI-compatible LLM endpoint

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create `config/config.py` (gitignored) and add your API key:

   ```python
   API_KEY = "YOUR_KEY_HERE"
   ```

4. Place your documents under the expected folders or update the paths in `wqbagent_embedding.py`:

   - `Docs/Forums`
   - `Docs/OfficialDocs`
   - `Docs/PaymentPolicy`

## Build embeddings and retrieval

1. In `wqbagent_embedding.py`, update `BASE_DIR` and the doc paths if needed.
2. Uncomment the one-time ingestion lines and run:

   ```powershell
   python .\wqbagent_embedding.py
   ```

3. Re-comment the ingestion lines after the DB is built to avoid reprocessing.

Embeddings are stored in `embedding_db/wqb_embedding_db` (gitignored). Ingest tracking is stored as `ingested_files.json` inside each docs folder.

## Run utilities

- Output/log formatting test:

  ```powershell
  python .\wqbagent_output_test.py
  ```

- Search tool health check (requires you to wire in the tools/LLM from your pipeline):

  ```powershell
  python .\wqbquant_searchtool_test.py
  ```

## Windows launchers

`scripts/wqbagent.bat`, `scripts/wqbagent_test.bat`, and `scripts/wqbagent_pws.ps1` are templates that:

- activate a venv
- force UTF-8 output
- pipe ANSI output to HTML using `ansi2html`

Update the venv path and the Python entrypoint (the default points to `wqbagent_v2.py`, which is not tracked in this repo) to match your local script or notebook export.

## Generated files

The following are created at runtime and are excluded from git:

- `logs/`
- `cache/`
- `embedding_db/` and `wqb_embedding_db/`
- `quant_forum_chroma/`, `quant_forum_bgem3/`

## License

No license file is currently included.
