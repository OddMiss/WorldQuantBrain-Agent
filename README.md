# WorldQuantBrain-Agent (v2.2)

Local CrewAI-based toolkit (v2.2) for building embeddings from WorldQuant Brain consultant materials and experimenting with multi-agent alpha idea generation. The repo includes the v2.2 agent pipeline, embedding notebook, test utilities, and notebooks for interactive development.

## Repository contents

- `wqbagent_v2_2.py`: v2.2 CrewAI pipeline for embedding retrieval, search tools, and alpha simulation.
- `wqbagent_embedding.ipynb`: v2.2 embedding build notebook for PDF/text sources.
- `wqbagent-v2.2.ipynb`: interactive notebook for the full v2.2 agent workflow.
- `wqbagent_output_test.py`, `wqbagent_output_test.ipynb`: output/log formatting and LLM connectivity checks.
- `wqbquant_searchtool_test.py`: health check helper for search/retrieval tools.
- `releases/`: archived v1/v2 artifacts (e.g., `wqbagent_v1.py`, `wqbagent-v1.ipynb`, `wqbagent-v2.ipynb`).
- `scripts/`: Windows batch/PowerShell launchers (update venv paths and Python entry points; see Windows launchers below).
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+.
- Windows recommended for the provided launch scripts; they can be adapted for other operating systems.
- Access to an OpenAI-compatible LLM endpoint.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create `config/api_key.py` (gitignored) and add your API keys:

   ```python
   API_KEY_MOONSHOT = "YOUR_KEY_HERE"
   API_KEY_GEMINI_C26 = "YOUR_KEY_HERE"
   API_KEY_GEMINI_CU = "YOUR_KEY_HERE"
   API_KEY_DEEPSEEK = "YOUR_KEY_HERE"
   ```

   Only set keys for the providers you plan to use; other entries can be left blank or removed. The variable names mirror the provider choices in `wqbagent_v2_2.py` (Moonshot, Gemini variants, and DeepSeek).

4. Place your documents under the expected folders or update the paths in `wqbagent_v2_2.py` / `wqbagent_embedding.ipynb`:

   - `Docs/Forums/wqb_china_consultant_pdf`
   - `Docs/Forums/wqb_global_consultant_pdf`
   - `Docs/Forums/wqb_research_pdf`
   - `Docs/Forums/wqb_brain_tips_pdf`
   - `Docs/OfficialDocs`

   Note: Move any existing PaymentPolicy PDFs from `Docs/PaymentPolicy` into `Docs/Forums/wqb_brain_tips_pdf` manually for v2.2.

## Build embeddings and retrieval

1. In `wqbagent_v2_2.py` / `wqbagent_embedding.ipynb`, update `BASE_DIR` and the doc paths if needed.
2. Run the embedding build workflow (recommended: `wqbagent_embedding.ipynb`):

   ```powershell
   jupyter lab
   ```

3. Execute the ingestion cells once to build the embedding DBs.
4. Embeddings are stored under `embedding_db/` (gitignored) with v2.2 subfolders:

   - `wqb_forum_china_embedding_db`
   - `wqb_forum_global_embedding_db`
   - `wqb_forum_research_embedding_db`
   - `wqb_forum_tips_embedding_db`
   - `wqb_official_docs_embedding_db`

   Ingest tracking is stored as `ingested_files.json` inside each docs folder.

## Run the v2.2 agent

```powershell
python .\wqbagent_v2_2.py
```

## Run utilities

- Output/log formatting test:

  ```powershell
  python .\wqbagent_output_test.py
  ```

- Search tool health check (import `test_agents` and pass your tool functions plus the LLM instance from your pipeline):

  ```powershell
  python .\wqbquant_searchtool_test.py
  ```

## Windows launchers

`scripts/wqbagent.bat`, `scripts/wqbagent_test.bat`, and `scripts/wqbagent_pws.ps1` are templates that:

- activate a venv
- force UTF-8 output
- pipe ANSI output to HTML using `ansi2html`

Update the venv path and the Python entry point to match an available script like `wqbagent_v2_2.py` or `wqbagent_output_test.py`, or your notebook export:

- `.bat`: update the venv activation line and the `python -u` command.
- `.ps1`: update `$venvActivate` and `$pythonScript`.

## Generated files

The following are created at runtime and are excluded from git:

- `logs/` (run logs)
- `cache/` (HF/transformers cache)
- `embedding_db/` (v2.2 vector stores, e.g. `wqb_forum_*_embedding_db` and `wqb_official_docs_embedding_db`)
- `wqb_embedding_db/` (legacy v1 vector store if configured separately)
- `quant_forum_chroma/`, `quant_forum_bgem3/` (legacy vector stores from earlier versions)

## License

No license file is currently included.
