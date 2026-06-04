Hugging Face repositories (specifically private Datasets or Model repos) make fantastic, secure object storage "buckets."

Using the official `huggingface_hub` library, both your local client and your Space can safely pull the API key file directly from your HF storage repository (`embedding-api-storage`). Furthermore, configuring your Vector DBs to read straight from `/data/embedding_db` aligns perfectly with Hugging Face's Persistent Storage tier.

Here is the corrected setup for your Hugging Face Space server and your local client.

---

## Step 1: Hugging Face Space Server Setup

### 1. Update `requirements.txt`

Add `huggingface_hub` to ensure the Space can communicate with your HF repository to download the API key file.

```text
fastapi
uvicorn
sentence-transformers
langchain
langchain-core
langchain-chroma
langchain-huggingface
langchain-text-splitters
langchain-community
langchain-openai
flashrank
# Include whichever package contains your langchain_classic imports
```

### 2. Update `app.py`

This script initializes the vector databases from `/data/embedding_db` and reaches out to your Hugging Face bucket (`embedding-api-storage`) to pull down your authentication token.

> 💡 **Tip:** Make sure to go to your Space's **Settings -> Variables and secrets** and add your `HF_TOKEN` as a Secret so the app has permission to read your private bucket repository.

```python
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MergerRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from flashrank import Ranker

app = FastAPI()

# --- FETCH API KEY FROM HUGGING FACE STORAGE BUCKET ---
def load_verification_key():
    try:
        hf_token = os.getenv("HF_TOKEN")
        # Downloads 'api_key.txt' from your HF repository/bucket
        # Note: Change repo_type to "model" or "space" if your bucket isn't a dataset
        file_path = hf_hub_download(
            repo_id="your-hf-username/embedding-api-storage",
            filename="api_key.txt",
            repo_type="dataset",
            token=hf_token
        )
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Failed to pull verification token from HF Hub: {e}")
        return os.getenv("SPACE_API_KEY", "emergency_fallback_token")

SPACE_API_KEY = load_verification_key()
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != f"Bearer {SPACE_API_KEY}":
        raise HTTPException(status_code=403, detail="Unauthorized API Key")

# --- INITIALIZE PIPELINE ON HUGGING FACE STORAGE ---
DB_BASE_DIR = Path("/data/embedding_db")
HF_CACHE_DIR = Path("/data/hf_cache") 
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_DIRS = [
    DB_BASE_DIR / "wqb_forum_china_embedding_db",
    DB_BASE_DIR / "wqb_forum_global_embedding_db",
    DB_BASE_DIR / "wqb_forum_research_embedding_db",
    DB_BASE_DIR / "wqb_forum_tips_embedding_db",
    DB_BASE_DIR / "wqb_official_docs_embedding_db"
]

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    cache_folder=str(HF_CACHE_DIR),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=False
)

retrievers = []
for db_dir in DB_DIRS:
    if db_dir.exists():
        vectorstore = Chroma(persist_directory=str(db_dir), embedding_function=embeddings)
        retrievers.append(vectorstore.as_retriever(search_kwargs={"k": 6}))
    else:
        print(f"⚠️ Database directory not found at: {db_dir}")

base_combined_retriever = MergerRetriever(retrievers=retrievers)
flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(HF_CACHE_DIR))
compressor = FlashrankRerank(client=flashrank_client, top_n=5)
final_ranker_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_combined_retriever)

class QueryPayload(BaseModel):
    query: str

@app.post("/retrieve")
def retrieve_documents(payload: QueryPayload, _ = Depends(verify_api_key)):
    try:
        docs = final_ranker_retriever.invoke(payload.query)
        return {"documents": [doc.page_content for doc in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

---

## Step 2: Local Client Integration

Back on your local machine, ensure you have `huggingface_hub` installed (`pip install huggingface_hub`). You can completely purge the local embeddings setup code and replace it with this remote-calling architecture:

```python
# ... [Keep your initial imports, directory setups, and variable loads] ...

# Delete the # ====================== RAG PIPELINE SETUP ====================== block
# You no longer need to load embeddings or Chroma locally.

# ====================== EXPLICIT PYTHON TOOLS ======================
import requests
from config.api_key import API_KEY_HF_SPACE # Make sure to add this to your config

SPACE_URL = "https://yourusername-your-space-name.hf.space/retrieve"
HEADERS = {"Authorization": f"Bearer {API_KEY_HF_SPACE}"}

def tool_retrieve_text_data(query: str) -> str:
    """Fetches relevant text snippets from the WorldQuant knowledge base via HuggingFace Space.
    Args:
        query (str): Highly specific search phrase. Be precise.
                     Good examples: "high Sharpe analyst revisions", 
                     "group_neutralize turnover control", "fq1_cons_eps_revisions template"
    Returns:
        str: Concatenated relevant document snippets separated by ---.
             Returns "No relevant documents found." if nothing matches.
    """
    try:
        response = requests.post(
            SPACE_URL,
            json={"query": query},
            headers=HEADERS,
            timeout=30 # Add a timeout so it doesn't hang forever
        )
        response.raise_for_status()
      
        docs = response.json().get("documents", [])
      
        if not docs:
            return "No relevant documents found."
          
        output = "\n---\n".join(docs)
        return clean_community_data(output)
      
    except requests.exceptions.RequestException as e:
        return f"Network Error contacting retrieval server: {e}"
    except Exception as e:
        return f"Error processing retrieved text: {e}"
```

## Step 3: Docker file

Named `Dockerfile`

```dockerfile
FROM python:3.10-slim

# Set up a working directory
WORKDIR /code

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application files
COPY . .

# Hugging Face Spaces require your app to serve on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Step 4: Readme file

Named `README.md`

```markdown
---
title: Embedding Api
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker       # 👈 CHANGE THIS TO DOCKER
pinned: false
---
```

Make sure to swap out `your-hf-username` and `your-space-name` with your exact Hugging Face handles, and you will be completely free of the local RAM burden!
