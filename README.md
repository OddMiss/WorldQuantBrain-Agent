# WorldQuantBrain-Agent (v2.3)

Local CrewAI-based toolkit (v2.3) for building embeddings from WorldQuant Brain materials and running a multi-agent alpha research workflow with retrieval-augmented generation (RAG). The repo includes the v2.3 agent pipeline, embedding notebook, API simulator client, and test utilities.

## Repository contents

- `wqbagent_v2_3.py`: v2.3 CrewAI pipeline (retrieval tools, LLM routing, and simulation integration).
- `wqbagent_embedding.ipynb`: embedding build notebook for PDF/text sources.
- `wqbagent_pdf2text.ipynb`: batch PDF-to-text extraction helper (optional preprocessing).
- `wqbagent-v2.3.ipynb`: interactive notebook for the full v2.3 agent workflow.
- `wqbagent_output_test.py`, `wqbagent_output_test.ipynb`: output/log formatting and LLM connectivity checks (update `BASE_DIR` if needed).
- `wqbquant_searchtool_test.py`, `wqbquant_searchtool_test.ipynb`: health check helper for search/retrieval tools.
- `wqbagentcore/`: core modules (LLM setup, embeddings, tools, crews).
- `wqb_api/`: WorldQuant Brain API client and simulation helpers.
- `config/`: configuration constants and provider notes (see `config/api-doc.ipynb`, plus gitignored API keys).
- `utils/`: logging and data-cleanup helpers.
- `materials/`: reference materials and notes.
- `scripts/`: Windows batch helpers and launchers.
- `releases/`: archived v1/v2 artifacts.
- `requirements.txt`: Python dependencies.

## Prerequisites

- 🚨 Python version must be <= 3.12.
- Windows is recommended for the provided launch scripts (they can be adapted for other OSes).
- Access to an OpenAI-compatible LLM endpoint (Moonshot, DeepSeek, Gemini, or a local proxy).
- WorldQuant Brain credentials if you plan to run the simulator API.
- JupyterLab if you plan to run the notebooks.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Ensure `config/` exists, then create `config/api_key.py` (gitignored):

   ```python
   API_KEY_MOONSHOT = "YOUR_KEY_HERE"
   API_KEY_GEMINI_C26 = "YOUR_KEY_HERE"
   API_KEY_GEMINI_CU = "YOUR_KEY_HERE"
   API_KEY_DEEPSEEK = "YOUR_KEY_HERE"
   API_KEY_GOOGLE_CLOUD = "YOUR_KEY_HERE"
   ```

   Define all variables; for providers you are not using, set empty strings.

4. Add WorldQuant Brain credentials (only required if you use the API simulator). Create
   `Credentials/brain_credentials_0.txt` with JSON content:

   ```json
   ["username", "password"]
   ```

5. Place your documents and metadata under the expected folders (or update paths in `wqbagent_v2_3.py` / `wqbagent_embedding.ipynb`):

   - `Docs/Forums/wqb_china_consultant_pdf`
   - `Docs/Forums/wqb_global_consultant_pdf`
   - `Docs/Forums/wqb_research_pdf`
   - `Docs/Forums/wqb_brain_tips_pdf`
   - `Docs/OfficialDocs`
   - `Operators/Operators-Agent.json`
   - `DataFields/Datafield-Dataset-Category-Description.json`

   Note: If migrating from older versions with PaymentPolicy PDFs in `Docs/PaymentPolicy`, move them into
   `Docs/Forums/wqb_brain_tips_pdf` (v2.3 treats PaymentPolicy content as part of the brain tips corpus).

## Build embeddings and retrieval

1. (Optional) Use `wqbagent_pdf2text.ipynb` to combine PDFs into text files if you prefer text-only ingestion.
2. Update `BASE_DIR` and doc paths in `wqbagent_embedding.ipynb` if you keep data outside the repo.
3. Run the embedding build workflow (recommended: `wqbagent_embedding.ipynb`):

   ```powershell
   jupyter lab
   ```

4. Execute the ingestion cells once to build the embedding DBs.
5. Embeddings are stored under `embedding_db/` with v2.3 subfolders:

   - `wqb_forum_china_embedding_db`
   - `wqb_forum_global_embedding_db`
   - `wqb_forum_research_embedding_db`
   - `wqb_forum_tips_embedding_db`
   - `wqb_official_docs_embedding_db`

   Ingest tracking is stored as `ingested_files.json` inside each docs folder.

## Run the v2.3 agent

```powershell
python .\wqbagent_v2_3.py
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

`scripts/wqbagent.bat`, `scripts/wqbagent_test.bat`, and `scripts/wqbtool_test.bat` are templates that:

- activate a venv
- force UTF-8 output
- pipe ANSI output to HTML using `ansi2html`

Update the venv path and the Python entry point to match an available script like `wqbagent_v2_3.py`,
`wqbagent_output_test.py`, or `wqbquant_searchtool_test.py`.

Additional helpers include `scripts/add_user_path.bat`, `scripts/add_user_path_here.py`, and `scripts/hf_wqb_sync.bat`.

## Generated files

The following are created at runtime and are excluded from git:

- `logs/` (run logs)
- `cache/` (HF/transformers cache)
- `embedding_db/` (v2.3 vector stores)
- `Docs/`, `DataFields/`, `Operators/`, `Credentials/` (local datasets and credentials)
- `wqb_embedding_db/`, `quant_forum_chroma/`, `quant_forum_bgem3/` (legacy vector stores from earlier versions)

## License

No license file is currently included.
