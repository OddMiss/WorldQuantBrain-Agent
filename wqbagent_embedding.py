import os
import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import datetime

def setup_logger(log_dir, log_name, logger_obj_name="logger_obj_name"):
    """
    Configure the logger system: output to both console and file simultaneously.
    """
    # 1. If the log directory doesn't exist, create it
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created: {log_dir}")

    # 2. Generate a timestamped log filename, e.g., scraper_log_20231027_103000.log
    log_filename = f"{log_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 3. Create Logger object
    logger = logging.getLogger(logger_obj_name)
    logger.setLevel(logging.INFO) # Set the minimum logger level
    logger.handlers = [] # Clear previous handlers to prevent duplicate printing

    # --- Define a unified format (time accurate to the second) ---
    # %(asctime)s : Time
    # %(levelname)s : Log level (INFO/ERROR)
    # %(message)s : Your message content
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )

    # --- Handler 1: File output (detailed, with timestamp) ---
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Handler 2: Console output (concise, for human reading) ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"✅ logger System Started")
    logger.info(f"Log file path: {log_filepath}")
    return logger

# ====================== CONFIG: EVERYTHING ON YOUR OTHER DRIVE ======================
BASE_DIR = Path.cwd() # current directory
WQB_FORUM_PATH = BASE_DIR / "Docs" / "Forums"
WQB_OFFICIAL_DOCS_PATH = BASE_DIR / "Docs" / "OfficialDocs"
WQB_PAYMENT_POLICY_PATH = BASE_DIR / "Docs" / "PaymentPolicy"
EMBEDDING_DB_DIR = BASE_DIR / "embedding_db" / "wqb_embedding_db"
HF_CACHE_DIR = BASE_DIR / "cache" / "hf"
LOG_DIR = BASE_DIR / "logs" / datetime.datetime.now().strftime("%Y%m")

# Create the folders (pathlib makes this easy too)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ====================== INITIALIZE LOGGER ======================
logger = setup_logger(LOG_DIR, "wqb_agent", "wqb_main_logger")

# Loader Dict
LOADER_DICT = {
    "pdf": PyPDFLoader,
    "txt": TextLoader
}

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", # excellent for Chinese
    cache_folder=str(HF_CACHE_DIR),  # Use the custom cache directory
    model_kwargs={"device": "cpu"},           # force CPU (your low GPU setup)
    encode_kwargs={"normalize_embeddings": True},  # best for Chroma similarity search
    show_progress=True
)

# ==================================== Ingest PDFs (ONE-TIME) ====================================
def ingest_embeddings(docs_path, embeddings, vectorstore_dir, docs_type, source_type=""):
    logger.info(f"Starting {source_type} ingestion from: {docs_path}")

    if docs_type not in LOADER_DICT:
        logger.error(f"Unsupported document type: {docs_type}. Supported types are: {list(LOADER_DICT.keys())}")
        return False

    docs_path_abs = os.path.abspath(docs_path)
    vectorstore_dir_abs = os.path.abspath(vectorstore_dir)
    ingest_state_path = os.path.join(docs_path_abs, "ingested_files.json")

    # State is tied to vectorstore_dir. If DB path changes, treat as a fresh DB.
    ingest_state = {
        "vectorstore_dir": vectorstore_dir_abs,
        "files": []
    }

    if os.path.exists(ingest_state_path):
        try:
            with open(ingest_state_path, "r", encoding="utf-8") as f:
                existing_state = json.load(f)
            if isinstance(existing_state, dict) and existing_state.get("vectorstore_dir") == vectorstore_dir_abs:
                ingest_state = {
                    "vectorstore_dir": vectorstore_dir_abs,
                    "files": existing_state.get("files", []) if isinstance(existing_state.get("files", []), list) else []
                }
            else:
                logger.info("Detected a new embedding DB path. Resetting ingest tracking state.")
        except Exception as e:
            logger.warning(f"Failed to read ingest state file, starting fresh. Reason: {e}")

    already_ingested = set(ingest_state["files"])

    loader = DirectoryLoader(
        docs_path,
        glob=f"**/*.{docs_type}",
        loader_cls=LOADER_DICT.get(docs_type),
        silent_errors=False,
        show_progress=True
    )

    logger.info("Loading documents (this may take a while)...")
    try:
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} {docs_type.upper()} files.")
    except Exception as e:
        logger.error(f"❌ Failed to load {docs_type.upper()}s: {e}", exc_info=True)
        return False

    docs_to_ingest = []
    newly_added_files = set()
    # for doc in docs:
    #     source_file = os.path.abspath(doc.metadata.get("source", ""))
    #     if source_file and source_file in already_ingested:
    #         continue
    #     docs_to_ingest.append(doc)
    #     if source_file:
    #         newly_added_files.add(source_file)

    # ============ Fix: Use relative path record ===============
    for doc in docs:
        raw_source = doc.metadata.get("source", "")
        
        if raw_source:
            # 1. Get the absolute path of the file
            abs_source = os.path.abspath(raw_source)
            
            # 2. Find the relative path from your project root (BASE_DIR)
            # This turns '/data/Docs/OfficialDocs/file.pdf' into 'Docs/OfficialDocs/file.pdf'
            # Docs/OfficialDocs/file.pdf is better than /Docs/OfficialDocs/file.pdf
            rel_source = os.path.relpath(abs_source, start=BASE_DIR)
            
            # 3. Force forward slashes for Windows/Linux cross-compatibility
            source_file = rel_source.replace("\\", "/")
        else:
            source_file = ""
            
        if source_file and source_file in already_ingested:
            continue
            
        docs_to_ingest.append(doc)
        
        if source_file:
            newly_added_files.add(source_file)
    
    if not docs_to_ingest:
        logger.info("No new files to ingest. All matching files already exist in ingest state.")
        return True

    logger.info(f"New files to ingest: {len(docs_to_ingest)}")

    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    splits = text_splitter.split_documents(docs_to_ingest)
    logger.info(f"Created {len(splits)} text chunks.")

    # Add source metadata to each chunk for better traceability if source_type is provided
    if source_type:
        for chunk in splits:
            chunk.metadata["source_type"] = source_type
        logger.info(f"Added source_type metadata to each chunk: {source_type}")

    logger.info("Initializing Chroma vectorstore and generating embeddings...")
    try:
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vectorstore_dir
        )

        updated_files = sorted(already_ingested.union(newly_added_files))
        with open(ingest_state_path, "w", encoding="utf-8") as f:
            json.dump({"vectorstore_dir": vectorstore_dir_abs, "files": updated_files}, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Successfully ingested {len(splits)} chunks into Chroma DB at {vectorstore_dir} for source: {source_type}")
        logger.info(f"Updated ingest tracking file: {ingest_state_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error during Chroma DB creation: {e}", exc_info=True)
        raise e

# First run only (First run forum data then official docs)
# Success_Forum = ingest_embeddings(WQB_FORUM_PATH, embeddings, EMBEDDING_DB_DIR, "pdf", "forum_pdf")
# Success_OfficialDocs = ingest_embeddings(WQB_OFFICIAL_DOCS_PATH, embeddings, EMBEDDING_DB_DIR, "pdf", "official_docs_pdf")
# Success_PaymentPolicy = ingest_embeddings(WQB_PAYMENT_POLICY_PATH, embeddings, EMBEDDING_DB_DIR, "pdf", "payment_policy_pdf")

# ==================================== Initialize Retriever (for querying) ====================================
vectorstore = Chroma(persist_directory=EMBEDDING_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ====================== DOCS SEARCH TOOL ======================
@tool("retrieve_text_data")
def retrieve_text_data(query: str) -> str:
    """Fetches relevant text snippets based on a query.
    Input a highly specific financial concept or math operator (e.g., 'supply chain momentum', 'analyst revision').
    Returns text context to be used for answering user queries."""
    
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# ==================================== DEBUG: Test if your docs PDFs are loaded ====================================
print("Testing retrieve_text_data tool...")

test_result = retrieve_text_data.run("AllRightsReserved")

print("\n" + "="*80)
print("TOOL TEST RESULT:")
print(test_result)  # first 1500 chars
print("\n" + "="*80)
print(f"Length of returned text: {len(test_result)} characters")