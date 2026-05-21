import os
import json
import requests
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from langchain_chroma import Chroma
from langchain_classic.retrievers import MergerRetriever
from crewai.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from config.api_key import API_KEY_MOONSHOT, API_KEY_GEMINI_C26, API_KEY_GEMINI_CU, API_KEY_DEEPSEEK
import datetime
from utils.logger import setup_logger
from utils.htmlcolorlog import capture_and_log
from wqbquant_searchtool_test import test_search_tools

# ====================== IMPORT FROM YOUR SIMULATOR ======================
from wqb_api.wqb_api_v1 import (
    initialize_global_variables,
    _validate_regular_formula,
    _build_simulation_payload,
    simulate_and_evaluate_alpha
)

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
TRANSCRIPT_FILE = LOG_DIR / f"wqb_agent-{timestamp}.transcript.txt"
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

# Get Model List

base_moonshot_url = "https://api.moonshot.cn/v1"
model_moonshot_url = "https://api.moonshot.cn/v1/models"
base_gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/models"
base_deepseek_url = "https://api.deepseek.com/v1"
model_deepseek_url = "https://api.deepseek.com/v1/models"
base_url = base_deepseek_url
model_url = model_deepseek_url
# API_KEY = API_KEY_MOONSHOT
API_KEY = API_KEY_DEEPSEEK

def Get_Model_List(url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        models = response.json()
        logger.info("Main", f"Response: {models}")
        model_list = []
        for model in models['data']:
            model_list.append(model['id'])
            logger.info("Main", f"Model ID: {model['id']}")
        return model_list
    else:
        logger.error("Main", f"Error: {response.status_code}, {response.text}")
        return []

Get_Model_List(model_url, API_KEY)

# Use your exact proxy settings
# pro_model = "moonshot/kimi-k2.5" # gemini-3.1-pro (if use reserve gemini)
# flash_model = "moonshot/moonshot-v1-128k" # gemini-3.0-flash-thinking (if use reserve gemini)
pro_model = "deepseek/deepseek-v4-pro"
flash_model = "deepseek/deepseek-v4-flash"
# 🚨🚨 CRITICAL WARNING: LLM Provider must be provided. Pass in the LLM provider you are trying to call. Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers

llm_pro = LLM(
    model=pro_model,   # ← change if your proxy uses a different model name
    base_url=base_url,
    api_key=API_KEY,
    temperature=0.6,          # slightly lower = more stable
    max_tokens=8192,
    timeout=180,              # give it more time
    max_retries=3,            # extra retries
)

llm_flash = LLM(
    model=flash_model,   # ← change if your proxy uses a different model name
    base_url=base_url,
    api_key=API_KEY,
    temperature=0.6,          # slightly lower = more stable
    max_tokens=8192,
    timeout=180,              # give it more time
    max_retries=3,            # extra retries
)

# LLM query test

def test_llm_query(llm, query):
    try:
        # CHANGE THIS: Use .call() to execute the query
        response = llm.call(query) 
        
        logger.info("LLM Test", f"Query: {query}")
        logger.info("LLM Test", f"Response: {response}")
    except Exception as e:
        logger.error("LLM Test", f"Error: {str(e)}")

# test_llm_query(llm_flash, "Who are you and what can you do?")

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
@tool("retrieve_text_data")
def retrieve_text_data(query: str) -> str:
    """Fetches relevant text snippets based on a string query.
    
    IMPORTANT FORMATTING RULE: 
    The 'query' argument MUST be a plain string. 
    DO NOT pass a dictionary.
    For example, 
    Correct: "momentum" 
    Incorrect: {"type": "str", "value": "momentum"}
    
    Input a highly specific financial concept or math operator (e.g., 'supply chain momentum', 'analyst revision').
    Returns text context to be used for answering user queries."""
    
    docs = combined_retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

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
            
        if len(results) >= 10:  # Limit results to save context window
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
            
            res = f"Field: {field_name}\nType: {field_details.get('type')}\nDesc: {field_details.get('description')}"
            results.append(res)
            
        if len(results) >= 15:  # Limit results to save context window
            break
            
    return "\n---\n".join(results) if results else "No matching data fields found."

# Initialize simulator global variables (operators, datafields, multipliers)
# This must be called before the agents attempt to use the tools
account_no = "0"  # You can change this if needed
success, init_msg = initialize_global_variables(account_no=account_no)
if not success: raise RuntimeError(f"Failed to initialize simulator: {init_msg}")

@tool("check_regular_formula")
def check_regular_formula(regular_formula: str, region: str, delay: int, universe: str) -> str:
    """
    Validates the syntax and data fields of the regular formula locally BEFORE running a full simulation.
    Call this to detect invalid datafields, bad syntax, or missing operators.
    
    Inputs:
    - regular_formula: The alpha mathematical expression.
    - region: The region {"USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"}
    - delay: Integer delay, 0 or 1. The rule: {
        "USA": {0, 1},
        "GLB": {1},
        "EUR": {0, 1},
        "ASI": {1},
        "CHN": {0, 1},
        "IND": {1},
        "KOR": {1},
        "TWN": {1},
        "MEA": {1}
    }
    - universe: The target universe. The rule: {
        "USA": {"TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "TOPSP500", "ILLIQUID_MINVOL1M"},
        "GLB": {"TOP3000", "MINVOL1M", "MINVOL10M", "TOPDIV3000"},
        "EUR": {"TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M", "TOPCS1600"},
        "ASI": {"TOP500", "MINVOL1M", "MINVOL10M", "ILLIQUID_MINVOL1M"},
        "CHN": {"TOP2000U"},
        "IND": {"TOP500"},
        "KOR": {"TOP600"},
        "TWN": {"TOP500", "TOP100"},
        "MEA": {"TOP400", "TOP300"}
    }
    """
    try:
        success, result = _validate_regular_formula(
            regular_formula=regular_formula,
            region=region,
            delay=int(delay),
            universe=universe,
            account_no=account_no # global variable set during initialization
        )
        if not success:
            return f"❌ Validation Failed: {result}\nReview the datafields and operators used."
        return f"✅ Formula Check Passed! Multiplier details: {result}"
    except Exception as e:
        return f"❌ Checker Exception: {e}"

@tool("wqb_simulate_api")
def wqb_simulate_api(settings: str, regular_formula: str) -> str:
    """
    Run full alpha simulation and evaluation by calling the WQB API.
    
    Inputs:
    - settings: JSON string of simulator settings. Must included keys: [
        "instrumentType", # default "EQUITY"
        "region", # "USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"
        "universe", # depends on region, e.g. "TOP3000", "MINVOL1M", etc.
        "delay", # 0 or 1, depends on region
        "decay", # between 0 and 512, suggest to be integer for better performance, but float is also acceptable
        "neutralization", # depends on region, e.g. "SECTOR", "INDUSTRY", "COUNTRY", "GICS_SECTOR"
        "truncation", # between 0 and 1, inclusive, float only, suggest keep 2 decimal places for better performance, but more decimal places are also acceptable
        "pasteurization", # either ON or OFF
        "nanHandling", # either ON or OFF
        "testPeriod", # Format 1: with year and without month, like "P1Y"; Format 2: with year and month, like "P1Y2M"; Format 3: with month only, like "P2M"
        "maxTrade", # either ON or OFF
        "maxPosition" # either ON or OFF
        # Max Position and Max Trade cannot both be set to On simultaneously
    ]
    - regular_formula: The WorldQuant expression string.
    
    Output:
    Returns the JSON payload containing Simulation Status, IS_Checks, and Correlation.
    If the status is not 'COMPLETE' or IS_Checks fail, read the feedback and try again!
    """
    try:
        settings_payload = json.loads(settings) if isinstance(settings, str) else settings
    except json.JSONDecodeError as e:
        return f"❌ Invalid settings JSON: {e}"

    # 1. Build Payload and validate settings exactly as the simulator expects
    success, sim_payload = _build_simulation_payload(settings_payload, regular_formula, account_no="0")
    if not success:
        return f"❌ Payload Build Error: {sim_payload}"

    # 2. Run simulation and evaluation
    success, result = simulate_and_evaluate_alpha(
        alpha_settings=sim_payload,
        regular=regular_formula,
        account_no=account_no, # global variable set during initialization
        include_self_corr=True,
        include_prod_corr=True
    )
    
    if not success:
        return f"❌ API Request Failed: {result}"
        
    return json.dumps(result, ensure_ascii=False, indent=2)

# ====================== AGENTS (Your Quant Research Team) ======================
# 💡 Note: # allow_delegation=False means the agent cannot delegate to other agents and must complete the task itself. 
# This is important for the researcher to ensure it fully utilizes the retrieval tool and doesn't skip steps.
researcher = Agent(
    role="WorldQuant Docs Researcher & Master Analyst",
    goal="Use the `retrieve_text_data` tool to fetch context for the user's request. Base your output on the tool's results. Never answer from general knowledge.",
    backstory="""You are a veteran WorldQuant Brain consultant. You are an advanced AI agent equipped with a local vector database interface.
    You do not have the PDFs in your internal memory; you rely ENTIRELY on the `retrieve_text_data` tool. You use lateral thinking. 
    If a user asks about 'volume', you search for 'liquidity shock', 'turnover spike', or 'institutional block trades'.
    You always call the `retrieve_text_data` tool multiple times with completely different vocabulary each time to get a full picture.
    """,
    tools=[retrieve_text_data],
    llm=llm_pro,
    verbose=True,
    allow_delegation=False
)

ideator = Agent(
    role="Low-Correlation BRAIN Innovator",
    goal="Create truly innovative, submittable alphas using specific alternative data fields.",
    backstory="""You are a contrarian quant. You MUST use the `search_datafields` tool to find real, specific dataset names (like 'anti_pollution_policy_industry_rank') to build your hypotheses. Do not hallucinate data field names.""",
    tools=[search_datafields], # <--- ADDED TOOL
    llm=llm_pro,
    verbose=True
)

coder = Agent(
    role="WorldQuant BRAIN Expression Expert",
    goal="Convert the idea into a valid expression using exact Operator syntax and Exact Data fields.",
    backstory="""You are an ex-WorldQuant Brain coder. 
    1. You MUST use `search_operators` to check the exact syntax of functions (e.g., checking if add() takes a filter argument). 
    2. You MUST use `search_datafields` to ensure you are using the exact string name for data sets.
    You never guess operator syntax.""",
    tools=[search_operators, search_datafields], # <--- ADDED TOOLS
    llm=llm_flash,
    verbose=True
)

# New: Validator can now call the simulator tool to get real feedback and iteratively improve the formula until it passes IS checks with a COMPLETE status.
validator = Agent(
    role="WorldQuant Submission Validator & Iterative Optimizer",
    goal="Ensure the alpha simulates correctly, passes all IS checks, and is ready to submit. Output ONLY in the exact user-specified format.",
    backstory="""You are the final gatekeeper and debugging expert. You never pass a broken alpha. 
    You are highly skilled at reading WQB API error messages (like 'Unknown Operator' or 'Invalid Datafield') 
    and iteratively tweaking the formula until the API returns a 'COMPLETE' status with passing IS criteria.
    You never give up on the first error; you adjust and resimulate.
    You are a strict code validator. You NEVER guess or fabricate 
    the simulation results. You MUST call your tools to get the real metrics.""",
    tools=[check_regular_formula, wqb_simulate_api],  # Equipped with the real tools
    llm=llm_pro,
    verbose=True,
    allow_delegation=False
)

# ====================== TASKS & CREW ======================
task1 = Task(
    description="""
    Based on the user's request, you are authorized and required to use the `retrieve_text_data` tool.
    
    STEP 1: Brainstorm 3 different, highly specific keyword phrases related to the request.
    STEP 2: Call the `retrieve_text_data` tool using one of your brainstormed phrases. If the result is poor, call it again with a different phrase.
    STEP 3: Synthesize the returned database snippets.
    
    Focus on extracting real opinions and specific discussions.
    """,
    expected_output="Structured summary of the retrieved database content with direct quotes.",
    agent=researcher
)

task2 = Task(
    description="""
    Generate 3-5 genuinely innovative alpha ideas.
    CRITICAL: You MUST use the `search_datafields` tool to search for keywords related to your ideas (e.g., "ESG", "Analyst", "Supply Chain") and include the EXACT field names in your output.
    Focus on low correlation and economic rationale.
    """,
    expected_output="Numbered list of 3-5 alpha ideas containing exact Data Field names, hypothesis, and low-correlation justification.",
    agent=ideator
)

task3 = Task(
    description="""
    Take the BEST idea from Task 2 and write a clean, valid WorldQuant BRAIN expression.
    CRITICAL: You MUST use the `search_operators` tool to verify the syntax of every math/logic function you plan to use before writing the final expression.
    Choose realistic Target Settings (Region, Universe, Neutralization, Delay, Decay, Truncation).
    """,
    expected_output="One complete alpha in the exact user format (Alpha Name + Economic Hypothesis + Target Settings + Full BRAIN Expression).",
    agent=coder
)

# New: Validator now has an iterative workflow to debug and improve the formula until it passes the simulator checks with a COMPLETE status.
task4 = Task(
    description="""
    Act as a strict WorldQuant reviewer and iterate until the alpha is perfect.
    You are FORBIDDEN from writing down or fabricating the tool results yourself.
    
    CRITICAL WORKFLOW:
    1. Validate the formula locally using the `check_regular_formula` tool. Pass the formula, region, delay, and universe.
       - If it fails, fix the datafields or operators and check again.
    2. Once local validation passes, call `wqb_simulate_api` with the full settings JSON and the formula.
       - Wait for the API response. 
    3. Analyze the API Output:
       - Look at `"simulation": {"status": ... }`. If it is "ERROR", read the message, modify the formula, and re-run step 2.
       - Look at `"evaluation": {"is_checks": {"Status": ... }}`. If it's "FAIL" (e.g., low Sharpe, high Turnover), tweak your formula parameters, operators, or settings, and re-run step 2.
    4. Repeat this iterative debugging process up to 4 times until you achieve a 'COMPLETE' status and a 'PASS' in IS_Checks.

    THEN output ONLY the final working alpha in the EXACT format the user wants:
    
    **Alpha Name:** ...
    **Economic Hypothesis:** ...
    **Target Settings:** Region: ___ | Universe: ___ | Neutralization: ___ | Delay: ___ | Decay: ___ | Truncation: ___
    **Full BRAIN Expression:** ...
    
    Do not add any extra explanation, reasoning, or debugging text outside this format in your final output.
    """,
    expected_output="Final working alpha in the exact markdown format requested by the user.",
    agent=validator
)

crew = Crew(
    agents=[researcher, ideator, coder, validator],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True,
    max_rpm=12
    # tracing=True
)

user_request = ("""
Explore forum discussions specifically around Analyst Estimates (EPS, Revisions) or Supply Chain inventory data. 
Find out what basic combinations people are using, and generate alphas that apply non-linear operators 
(like sign, abs, or conditional logic) to these specific datasets to achieve low correlation.
""").strip()

# ====================== RUN ======================
if __name__ == "__main__":
    with capture_and_log(TRANSCRIPT_FILE, HTML_FILE):
        logger.info("Main", f"🚀 Kickstarting Crew process with input: '{user_request}'")
        try:
            result = crew.kickoff(inputs={"user_request": user_request})
            logger.info("Main", "✅ Crew kickoff completed successfully.")
            logger.info("Main", f"\n{'='*50}\nFINAL RESULT\n{'='*50}\n{result}")
            
        except Exception as e:
            logger.error("Main", f"❌ Fatal error during Crew execution: {e}", exc_info=True)