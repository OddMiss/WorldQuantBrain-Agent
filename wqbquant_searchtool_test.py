import os
import sys
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import json
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from langchain_chroma import Chroma
from langchain_classic.retrievers import MergerRetriever
from crewai.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import Abbr_To_Full
from config.api_key import (
    API_KEY_MOONSHOT
)
import datetime
from utils.logger import setup_logger
from utils.htmlcolorlog import capture_and_log

# ====================== CONFIG: EVERYTHING ON YOUR OTHER DRIVE ======================
BASE_DIR = Path.cwd() # current directory
# WQB_FORUM_PATH = BASE_DIR / "Docs" / "Forums"
WQB_FORUM_CHINA_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_china_consultant_pdf"
WQB_FORUM_GLOBAL_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_global_consultant_pdf"
WQB_FORUM_RESEARCH_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_research_pdf"
WQB_FORUM_TIPS_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_brain_tips_pdf"
WQB_OFFICIAL_DOCS_PATH = BASE_DIR / "Docs" / "OfficialDocs"
OPERATOR_FILE_PATH = BASE_DIR / "Operators" / "Operators-Agent.json"
DATAFIELDS_FILE_PATH = BASE_DIR / "DataFields" / "Datafield-Dataset-Category-Description.json"
# Note: PaymentPolicy store in WQB_FORUM_TIPS_PATH since it has only few pdfs
# WQB_PAYMENT_POLICY_PATH = BASE_DIR / "Docs" / "PaymentPolicy"

EMBEDDING_DB_FORUM_CHINA_DIR = BASE_DIR / "embedding_db" / "wqb_forum_china_embedding_db"
EMBEDDING_DB_FORUM_GLOBAL_DIR = BASE_DIR / "embedding_db" / "wqb_forum_global_embedding_db"
EMBEDDING_DB_FORUM_RESEARCH_DIR = BASE_DIR / "embedding_db" / "wqb_forum_research_embedding_db"
EMBEDDING_DB_FORUM_TIPS_DIR = BASE_DIR / "embedding_db" / "wqb_forum_tips_embedding_db"
EMBEDDING_DB_OFFICIALDOCS_DIR = BASE_DIR / "embedding_db" / "wqb_official_docs_embedding_db"
EMBEDDING_DB_DIRECTORIES = {
    EMBEDDING_DB_FORUM_CHINA_DIR,
    EMBEDDING_DB_FORUM_GLOBAL_DIR,
    EMBEDDING_DB_FORUM_RESEARCH_DIR,
    EMBEDDING_DB_FORUM_TIPS_DIR,
    EMBEDDING_DB_OFFICIALDOCS_DIR
}
HF_CACHE_DIR = BASE_DIR / "cache" / "hf"
LOG_DIR = BASE_DIR / "logs" / datetime.datetime.now().strftime("%Y%m")

# Create a timestamp for the color html log and transcript files
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
HTML_FILE = LOG_DIR / f"wqb_agent-{timestamp}.html"

# Create the folders (pathlib makes this easy too)
for directory in [
    HF_CACHE_DIR, EMBEDDING_DB_FORUM_CHINA_DIR, EMBEDDING_DB_FORUM_GLOBAL_DIR, 
    EMBEDDING_DB_FORUM_RESEARCH_DIR, EMBEDDING_DB_FORUM_TIPS_DIR, 
    EMBEDDING_DB_OFFICIALDOCS_DIR, LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ====================== LOAD JSON DATA ======================
with open(OPERATOR_FILE_PATH, 'r', encoding='utf-8') as f:
    operators_data = json.load(f)

with open(DATAFIELDS_FILE_PATH, 'r', encoding='utf-8') as f:
    datafields_data = json.load(f)

# ====================== INITIALIZE LOGGER ======================
logger = setup_logger(LOG_DIR, "wqb_agent", "shared_logger")

base_moonshot_url = "https://api.moonshot.cn/v1"
model_moonshot_url = "https://api.moonshot.cn/v1/models"
flash_moonshot_model = "moonshot/moonshot-v1-128k"

base_url = base_moonshot_url
flash_model = flash_moonshot_model
API_KEY = API_KEY_MOONSHOT

llm_flash = LLM(
    model=flash_model,   # ← change if your proxy uses a different model name
    base_url=base_url,
    api_key=API_KEY,
    temperature=0.6,          # slightly lower = more stable
    max_tokens=8192,
    timeout=600,              # give it more time
    max_retries=3,            # extra retries
)

# ==================================== Initialize Retriever (for querying) ====================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", # excellent for Chinese
    cache_folder=str(HF_CACHE_DIR),  # Use the custom cache directory
    model_kwargs={"device": "cpu"},           # force CPU (your low GPU setup)
    encode_kwargs={"normalize_embeddings": True},  # best for Chroma similarity search
    show_progress=True
)

def Combine_Multiple_Embedding_Databases(embedding_db_directories, embeddings):
    retrievers = []
    for db_dir in embedding_db_directories:
        # Load the vectorstore for each directory
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        # Create a retriever for it
        # Note: If you set k=8 here, each DB will return 8 docs. 
        # With 5 databases, you'll get up to 40 documents back initially.
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        retrievers.append(retriever)
    # 3. Combine all individual retrievers into one
    combined_retriever = MergerRetriever(retrievers=retrievers)
    
    logger.info("Retriever Initialization", "✅ Combined retriever initialized with multiple embedding databases.")
    
    # Now you can use this exactly like a standard single retriever (test with a simple query)
    # combined_retriever.invoke("alpha")
    return combined_retriever

combined_retriever = Combine_Multiple_Embedding_Databases(EMBEDDING_DB_DIRECTORIES, embeddings)

# ====================== DOCS SEARCH TOOL ======================
@tool("retrieve_text_data_test")
def retrieve_text_data_test(query: str) -> str:
    """Fetches relevant text snippets based on a string query.
    Input a highly specific financial concept or math operator (e.g., 'supply chain momentum', 'analyst revision').
    Returns text context to be used for answering user queries."""
    
    # If is_test is True, return only the first 500 characters to avoid overwhelming the test output
    docs = combined_retriever.invoke(query)
    result = "\n\n---\n\n".join([doc.page_content for doc in docs])
    logger.info("RETRIEVE TEXT DATA TEST", f"Original retrieved text length: {len(result)} characters")
    return result[:500] + "..."  # Return only the first 500 characters for testing

# ====================== JSON SEARCH TOOLS ======================
@tool("search_operators")
def search_operators(query: str) -> str:
    """Search the operator definitions and syntax. 
    Input a concept or specific operator name."""
    results = []
    query_lower = query.lower()
    for op_name, op_details in operators_data.items():
        if (query_lower in op_name.lower() or 
            query_lower in op_details.get('description', '').lower() or 
            query_lower in op_details.get('category', '').lower()):
            
            res = f"Operator: {op_name}\nSyntax: {op_details.get('definition')}\nDesc: {op_details.get('description')}"
            results.append(res)
            
        if len(results) >= 15:  # Limit results to save context window
            break
            
    return "\n---\n".join(results) if results else "No matching operators found."

@tool("search_datafields")
def search_datafields(query: str) -> str:
    """Search the data dictionary for dataset fields. 
    Input a concept or specific field name."""
    results = []
    query_lower = query.lower()
    for field_name, field_details in datafields_data.items():
        if (query_lower in field_name.lower() or 
            query_lower in field_details.get('description', '').lower() or 
            query_lower in field_details.get('category_name', '').lower()): # Note: matching your JSON typo 'category_name'
            
            abbr_region_list = field_details.get('region', [])
            full_region_list = [Abbr_To_Full.get(abbr, abbr) for abbr in abbr_region_list]
            res = f"Field: {field_name}\nType: {field_details.get('type')}\nDesc: {field_details.get('description')}\nRegion(s): {', '.join(full_region_list)}"
            results.append(res)
            
        if len(results) >= 15:  # Limit results to save context window
            break
            
    return "\n---\n".join(results) if results else "No matching data fields found."

def test_agents(
    retrieve_text_data_test, search_operators, 
    search_datafields, llm,
    html_file, logger
):
    """Run a strict health check for the three search tools."""
    diagnostic_tester = Agent(
        role="API Formatting Assistant",
        goal="Format queries perfectly to test the text-based search interfaces.",
        backstory="""You are an AI assistant interacting with external text-search APIs provided in your environment.
    You do not need direct access to local databases; you simply output the strict `[ToolCalls]` formatting requested to trigger the external search.""",
        tools=[retrieve_text_data_test, search_operators, search_datafields],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    diagnostic_task = Task(
        description="""
    Perform a strict health check on the system's tools.

    STEP 1: Execute a test for each available search function:
    - search 'AllRightsReserved' using `retrieve_text_data_test`
    - search 'neutralize' using `search_operators`
    - search 'health' using `search_datafields`

    STEP 2: Only AFTER you have observed the results from all three tools, formulate your final response.
    If any tool fails, returns an error, or is denied, clearly state the failure in the final report.""",
        expected_output="""A strict health check report formatted as a checklist:
    - [Tool Name]: ✅PASS/❌FAIL - [Short snippet of what was returned (first 20 words)]
    """,
        agent=diagnostic_tester,
    )

    crew = Crew(
        agents=[diagnostic_tester],
        tasks=[diagnostic_task],
        process=Process.sequential,
        verbose=True,
    )

    with capture_and_log(html_file):
        logger.info("Test Agents", f"🚀 Kickstarting Crew process.")
        
        try:
            result = crew.kickoff()
            logger.info("Test Agents", "✅ Crew kickoff completed successfully.")
            logger.info("Test Agents", f"\n{'='*50}\nFINAL RESULT\n{'='*50}\n{result}")
            
        except Exception as e:
            logger.error("Test Agents", f"❌ Fatal error during Crew execution: {e}", exc_info=True)

    return result

def test_search_tools(search_tool_fun, search_tool_name, search_query, logger):
    logger.info("Test Search Tools", f"Testing {search_tool_name} tool...")

    test_result = search_tool_fun.run(search_query)

    logger.info("Test Search Tools", f"\n{'='*80}")
    logger.info("Test Search Tools", "TOOL TEST RESULT:")
    logger.info("Test Search Tools", test_result)  # first 1500 chars
    logger.info("Test Search Tools", "\n" + "="*80)
    logger.info("Test Search Tools", f"Length of returned text: {len(test_result)} characters")

if __name__ == "__main__":
    test_search_tools(retrieve_text_data_test, "retrieve_text_data_test", "AllRightsReserved", logger)
    test_search_tools(search_operators, "search_operators", "neutralize", logger)
    test_search_tools(search_datafields, "search_datafields", "health", logger)
    test_agents(
        retrieve_text_data_test=retrieve_text_data_test, 
        search_operators=search_operators, 
        search_datafields=search_datafields, 
        llm=llm_flash, 
        html_file=HTML_FILE,
        logger=logger
    )