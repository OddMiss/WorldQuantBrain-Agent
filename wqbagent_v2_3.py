import os
import json
import requests
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from langchain_chroma import Chroma
from langchain_classic.retrievers import MergerRetriever
from crewai.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import Abbr_To_Full, DELAY, UNIVERSE, NEUTRALIZATION_DICT
from config.api_key import (
    API_KEY_MOONSHOT, API_KEY_GEMINI_C26, API_KEY_GEMINI_CU, 
    API_KEY_DEEPSEEK, API_KEY_GOOGLE_CLOUD)
import datetime
from utils.logger import setup_logger
from utils.htmlcolorlog import capture_and_log
from utils.clean_redundant_data import clean_community_data

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

# 🚨🚨 CRITICAL WARNING: LLM Provider must be provided. Pass in the LLM provider you are trying to call. Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers
base_moonshot_url = "https://api.moonshot.cn/v1"
model_moonshot_url = "https://api.moonshot.cn/v1/models"
pro_moonshot_model = "moonshot/kimi-k2.5"  # temperature must be 1
flash_moonshot_model = "moonshot/moonshot-v1-128k"

base_deepseek_url = "https://api.deepseek.com/v1"
model_deepseek_url = "https://api.deepseek.com/v1/models"
pro_deepseek_model = "deepseek/deepseek-v4-pro"
flash_deepseek_model = "deepseek/deepseek-v4-flash"

base_gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/models"
pro_gemini_model = "openai/gemini-3.1-pro"
flash_gemini_model = "openai/gemini-3.0-flash-thinking"

base_local_googlecloud = "http://127.0.0.1:8000/v1"
model_local_googlecloud = "http://127.0.0.1:8000/v1/models"
pro_googlecloud_model = "openai/gemini-2.5-pro"
flash_googlecloud_model = "openai/gemini-2.5-flash"

base_url = base_moonshot_url
model_url = model_moonshot_url
pro_model = pro_moonshot_model
flash_model = flash_moonshot_model
API_KEY = API_KEY_MOONSHOT
# base_url = base_deepseek_url
# model_url = model_deepseek_url
# pro_model = pro_deepseek_model
# flash_model = flash_deepseek_model
# API_KEY = API_KEY_DEEPSEEK

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

# Model_List = Get_Model_List(model_url, API_KEY)

# for tool calling, use deepseek/moonshot
llm_pro = LLM(
    model=pro_model,   # ← change if your proxy uses a different model name
    base_url=base_url,
    api_key=API_KEY,
    temperature=1,          # slightly lower = more stable
    max_tokens=8192,
    timeout=600,              # give it more time
    max_retries=3,            # extra retries
)
logger.info("Main", f"Initialized Pro LLM with model: {pro_model} and base URL: {base_url}")

llm_flash = LLM(
    model=flash_model,   # ← change if your proxy uses a different model name
    base_url=base_url,
    api_key=API_KEY,
    temperature=0.1,          # slightly lower = more stable
    max_tokens=8192,
    timeout=600,              # give it more time
    max_retries=3,            # extra retries
)
logger.info("Main", f"Initialized Flash LLM with model: {flash_model} and base URL: {base_url}")

# for creative work, use gemini
llm_creative = LLM(
    model=pro_googlecloud_model,      # via your Vertex proxy
    base_url=base_local_googlecloud,
    api_key=API_KEY_GOOGLE_CLOUD,
    temperature=0.3,
)
logger.info("Main", f"Initialized Creative LLM with model: {pro_googlecloud_model} and base URL: {base_local_googlecloud}")

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
    """Fetches relevant text snippets from the WorldQuant knowledge base.
    Args:
        query (str): Highly specific search phrase. Be precise.
                     Good examples: "high Sharpe analyst revisions", 
                     "group_neutralize turnover control", "fq1_cons_eps_revisions template"
    Returns:
        str: Concatenated relevant document snippets separated by ---.
             Returns "No relevant documents found." if nothing matches.
    """
    docs = combined_retriever.invoke(query)
    output = "\n---\n".join([doc.page_content for doc in docs])
    output_cleaned = clean_community_data(output)  # Clean redundant data
    return output_cleaned

# ====================== JSON SEARCH TOOLS ======================
@tool("search_operators")
def search_operators(query: str) -> str:
    """Search for WorldQuant BRAIN operators and their exact syntax.
    Args:
        query (str): Operator name or concept (e.g., "decay_linear", "group_neutralize", "signed_power")
    Returns:
        Formatted list of matching operators with syntax and description.
    """
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
    """Search for WorldQuant data fields/datasets.
    Args:
        query (str): Keyword, field name, or concept (e.g., "eps revision", "esg", "insider buying")
    Returns:
        Formatted list of matching data fields with description and regions.
    """
    results = []
    query_lower = query.lower()
    for field_name, field_details in datafields_data.items():
        if (query_lower in field_name.lower() or 
            query_lower in field_details.get('description', '').lower() or 
            query_lower in field_details.get('category_name', '').lower()): # Note: matching your JSON typo 'category_name'
            
            abbr_region_list = field_details.get('region', [])
            full_region_list = [Abbr_To_Full.get(abbr, abbr) for abbr in abbr_region_list]
            res = f"Field: {field_name} Type: {field_details.get('type')} Desc: {field_details.get('description')} Region(s): {', '.join(full_region_list)}"
            results.append(res)

    if not results:
        # Fallback: suggest similar fields
        # the query may be "eps revision" but the field is "fq1_cons_eps_revisions", so we can do a loose match based on words in the query
        # or the query may be with "_" like "eps_revision" but the field is "fq1_cons_eps_revisions", so we can also split the query by "_" and match with field words
        similar = []
        query_words = set(query_lower.replace("_", " ").split())
        for field_name, field_details in datafields_data.items():
            field_words = set(field_name.lower().replace("_", " ").split()) | set(field_details.get('description', '').lower().replace("_", " ").split())
            if query_words & field_words:  # if there is any overlap in words
                abbr_region_list = field_details.get('region', [])
                full_region_list = [Abbr_To_Full.get(abbr, abbr) for abbr in abbr_region_list]
                similar.append(f"Field: {field_name} Type: {field_details.get('type')} Desc: {field_details.get('description')} Region(s): {', '.join(full_region_list)}")
        return f"No exact match. Similar fields: {similar[:20]}"
    
    return "\n---\n".join(results[:20])

# Initialize simulator global variables (operators, datafields, multipliers)
# This must be called before the agents attempt to use the tools
account_no = "0"  # You can change this if needed
fail, init_msg = initialize_global_variables(account_no=account_no)
if fail: 
    logger.error("WQB Agent", f"Failed to initialize simulator: {init_msg}")
    exit(1)

@tool("get_region_allowed_settings")
def get_region_allowed_settings(region: str) -> str:
    """Returns allowed configuration (Universe, Delay, Neutralization) for a given region.
    Input:
        region: A 3-letter region string, including: "USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"
    """
    regions = ["USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"]
    rules = {}
    for reg in regions:
        rules[reg] = {
            "delay": DELAY.get(reg, []),
            "universe": UNIVERSE.get(reg, []),
            "neutralization": NEUTRALIZATION_DICT.get(reg, [])
        }
    region_upper = region.upper().strip()
    if region_upper in rules:
        return f"Allowed configuration for region {region_upper}:\n{rules[region_upper]}"
    else:
        return f"❌ Region '{region}' not recognized. Choose from: {list(rules.keys())}"

@tool("check_regular_formula")
def check_regular_formula(regular_formula: str, region: str, delay: int, universe: str) -> str:
    """Validates alpha syntax and data fields locally. Ensure you check 'get_region_allowed_settings' first to avoid invalid combinations."""
    try:
        fail, result = _validate_regular_formula(
            regular_formula=regular_formula,
            region=region,
            delay=int(delay),
            universe=universe,
            account_no=account_no # global variable set during initialization
        )
        if fail: return f"❌ Validation Failed: {result}\nReview the datafields and operators used."
        return f"✅ Formula Check Passed! Multiplier details: {result}"
    except Exception as e:
        return f"❌ Checker Exception: {e}"

@tool("wqb_simulate_api")
def wqb_simulate_api(settings: str, regular_formula: str) -> str:
    """
    Run full alpha simulation and evaluation by calling the WQB API.
    
    Inputs:
    - settings (str or dict): JSON string of simulator settings. Must included keys: [
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
    
    Output: Full simulation result as JSON string.
    """
    try:
        settings_payload = json.loads(settings) if isinstance(settings, str) else settings
    except json.JSONDecodeError as e:
        return f"❌ Invalid settings JSON: {e}"

    # 1. Build Payload and validate settings exactly as the simulator expects
    fail, sim_payload = _build_simulation_payload(settings_payload, regular_formula, account_no="0")
    if fail: return f"❌ Payload Build Error: {sim_payload}"

    # 2. Run simulation and evaluation
    fail, result = simulate_and_evaluate_alpha(
        alpha_settings=sim_payload,
        regular=regular_formula,
        account_no=account_no, # global variable set during initialization
        include_self_corr=True,
        include_prod_corr=True
    )
    
    if fail: return f"❌ API Request Failed: {result}"
    return result

# ====================== AGENTS (Your Quant Research Team) ======================
# 💡 Note: # allow_delegation=False means the agent cannot delegate to other agents and must complete the task itself. 
# This is important for the researcher to ensure it fully utilizes the retrieval tool and doesn't skip steps.
researcher = Agent(
    role="WorldQuant Docs Researcher & Master Analyst",
    goal="You are a tool-using machine. You NEVER answer from memory. You ALWAYS call tools.",
    backstory="""You are a veteran WorldQuant Brain consultant.
    You MUST follow this exact workflow:
    1. Think: Brainstorm 3 specific search phrases.
    2. Call retrieve_text_data with one phrase.
    3. If results are weak, call it again with another phrase.
    4. Only after getting real snippets, synthesize.
    
    Never output a final answer before using the tool at least twice.""",
    tools=[retrieve_text_data],
    llm=llm_pro,
    verbose=True,
    allow_delegation=False
)

ideator = Agent(
    role="Low-Correlation BRAIN Innovator",
    goal="Generate creative, innovative, and economically sound alpha ideas based on the research provided.",
    backstory="""You are a contrarian and creative quant researcher.
    Your job is to take the findings from the Researcher (datasets, templates, successful techniques) 
    and generate 3-5 fresh, innovative alpha ideas.
    
    Rules:
    - Be creative: propose novel combinations and twists.
    - Stay grounded: only use datasets and techniques mentioned by the Researcher.
    - Focus on strong economic stories and low-correlation rationales.
    - Do NOT hallucinate new data field names that were not mentioned in the research.""",
    tools=[],
    llm=llm_creative,
    verbose=True
)

coder = Agent(
    role="WorldQuant BRAIN Expression Expert",
    goal="Convert the best idea from the Ideator into a clean, valid, and submittable BRAIN expression.",
    backstory="""You are an ex-WorldQuant Brain coder with strong technical discipline.
    
    You MUST:
    1. Use `search_operators` to check the exact syntax of functions. It will return the precise definition and usage.
    2. Use `search_datafields` to ensure you are using the exact string name for data sets.
    3. Use `get_region_allowed_settings` to check the exact allowed universe, delay, and neutralization options for your chosen region before finalizing your settings and formula.
    
    Never guess operator or field names.""",
    tools=[search_operators, search_datafields, get_region_allowed_settings], # <--- ADDED TOOLS
    llm=llm_pro,
    verbose=True
)

# New: Validator can now call the simulator tool to get real feedback and iteratively improve the formula until it passes IS checks with a COMPLETE status.
validator = Agent(
    role="WorldQuant Submission Validator & Iterative Optimizer",
    goal="Ensure the alpha simulates correctly, passes all IS checks, and is ready to submit. Output ONLY in the exact user-specified format.",
    backstory="""You are the final gatekeeper and debugging expert. You never pass a broken alpha.
    
    Strictly follow the workflow:
    1. Use get_region_allowed_settings
    2. Use check_regular_formula
    3. Use wqb_simulate_api and analyze the JSON output
    4. Iterate up to 4 times if needed (fix errors or failing IS checks).
    
    Never pass a broken alpha.""",
    tools=[check_regular_formula, get_region_allowed_settings, wqb_simulate_api],  # Equipped with the real tools
    llm=llm_pro,
    verbose=True,
    allow_delegation=False
)

# ====================== TASKS & CREW ======================
# 🚨 Fix the critical issue in Task 1: the "user_request" must be passed in as a variable that the agent can access.
task1 = Task(
    description="""
    {user_request}
    
    You are REQUIRED to strictly follow this exact workflow:
    
    1. Brainstorm 3 different, highly specific keyword phrases (related to "high Sharpe", "uncorrelated", "robust alpha template", "passed IS checks", "consultant tips", etc.).
    2. Call the `retrieve_text_data` tool using the best phrase.
    3. If the returned snippets are weak, irrelevant, or too short, immediately call the tool again with a different phrase.
    4. Only after receiving good results, synthesize the information.
    
    ABSOLUTE RULES:
    - Base everything on the tool results only.
    - Never hallucinate forum discussions, consultant quotes, or document content.
    - Extract and include direct quotes from the returned snippets when possible.
    
    Produce a structured analysis.
    """,
    expected_output="""A well-structured report containing:
    - Identified core concept/dataset
    - Extracted structural templates with direct quotes
    - Recommended innovations (neutralization, non-linear operators, turnover control)
    - One complete BRAIN expression""",
    agent=researcher
)

task2 = Task(
    description="""
    Based on the research output from Task 1, generate 3-5 genuinely innovative and low-correlation alpha ideas.
    
    Important:
    - Build your ideas on the real datasets, templates, and techniques mentioned by the Researcher.
    - Be creative: combine concepts, add new twists, or apply different neutralizations/non-linear operators.
    - Provide strong economic rationale for each idea.
    - Clearly mention which specific data fields or concepts from the research you are using.
    
    Do NOT invent new data field names that were not mentioned in the research.
    """,
    expected_output="Numbered list of 3-5 alpha ideas containing exact Data Field names, hypothesis, and low-correlation justification.",
    agent=ideator
)

task3 = Task(
    description="""
    Take the BEST idea from Task 2 and convert it into a clean, valid WorldQuant BRAIN expression.
    
    You MUST do the following:
    1. Use the `search_operators` tool to verify syntax of every operator you plan to use.
    2. Use the `search_datafields` tool to confirm exact field names.
    3. Use the `get_region_allowed_settings` tool to choose valid settings for your region.
    4. Build a realistic and complete expression.
    
    Do not guess any operator syntax or field names.
    """,
    expected_output="One complete alpha in the exact user format (Alpha Name + Economic Hypothesis + Target Settings + Full BRAIN Expression).",
    agent=coder
)

# New: Validator now has an iterative workflow to debug and improve the formula until it passes the simulator checks with a COMPLETE status.
task4 = Task(
    description="""
    Act as a strict WorldQuant reviewer. Take the alpha from Task 3 and iterate until it is production-ready.
    
    You MUST follow this exact workflow:
    
    1. First, call `get_region_allowed_settings` to confirm valid parameters.
    2. Call `check_regular_formula` to validate syntax and data fields.
    3. If validation passes, call `wqb_simulate_api` with proper settings JSON.
    4. Carefully analyze the simulation result:
       - If "ERROR" → fix and retry.
       - If IS_Checks = "FAIL" → tweak formula/settings and retry.
    5. Repeat up to 4 times if necessary.
    
    STRICT OUTPUT RULE:
    Output ONLY the final working alpha. No extra explanations, no debugging notes, no reasoning outside the required format.
    """,
    expected_output="""Final output in this EXACT format only:
    **Alpha Name:** ...
    **Economic Hypothesis:** ...
    **Target Settings:** Region: ___ | Universe: ___ | Neutralization: ___ | Delay: ___ | Decay: ___ | Truncation: ___
    **Full BRAIN Expression:** ...""",
    agent=validator
)

crew = Crew(
    agents=[researcher, ideator, coder, validator],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True,
    max_rpm=8
    # tracing=True
)

user_request = ("""
GOAL: Discover a highly successful alpha strategy from the consultant tips and forum discussions, and build a submittable alpha based on it.

Use your retrieval tools to search for broad success keywords like "high Sharpe", "uncorrelated", "passed IS checks", "robust alpha", "alpha template", "consultant recommendation", or "low turnover".

Requirements:
1. Identify ONE specific dataset or mathematical concept that the WorldQuant community or consultants strongly recommend.
2. Extract the exact structural template or logic discussed in the documents.
3. Innovate on it by applying recommended techniques: non-linear operators (e.g. signed_power), cross-sectional ranking (group_rank), or neutralization (group_neutralize / sector / subsector).
4. Address common pitfalls, especially high turnover, using decay_linear or ts_mean smoothing.
5. Build a complete, valid BRAIN expression that is practical and has a good chance of passing IS checks.

Focus on real, high-quality signals. Prioritize robustness over complexity.
""").strip()

# ====================== RUN ======================
if __name__ == "__main__":
    with capture_and_log(HTML_FILE):
        logger.info("Main", f"🚀 Kickstarting Crew process with input: '{user_request}'")
        try:
            result = crew.kickoff(inputs={"user_request": user_request})
            logger.info("Main", "✅ Crew kickoff completed successfully.")
            logger.info("Main", f"\n{'='*50}\nFINAL RESULT\n{'='*50}\n{result}")
            
        except Exception as e:
            logger.error("Main", f"❌ Fatal error during Crew execution: {e}", exc_info=True)