"""
Version 1

Key Differences & Inversions in this Setup

- Programmatic Tool Execution (No ReAct/Tool Guesswork)
Instead of handing tools to the agent and waiting for it to decide to call them via a reasoning chain, 
the Python loop forces tool execution. Before the expression writer is even called, data fields and operator 
requirements are fetched and directly injected into the prompt.

- Context Window Pruning
When the simulator returns a massive payload or a failure log, the script catches it inside the history_logs 
array, extracts only the specific syntax string or target metrics (Sharpe, Fitness, Turnover), and discards 
the bulky API wrapper boilerplate. This completely avoids context bloat and ensures the LLM stays focused.

- True Backward and Forward Routing
If an alpha completely fails after 4 iterations, the loop handles it natively. It breaks out of the active 
code routine, iterates the list index forward, and hands the next concept to the generation pipeline without 
crashing the runtime process.

- Zero Abstracted Lifecycle Overhead
Every action relies on clear try-except wrappers and explicit Python assignments. If an HTTP call fails or 
the simulator returns an unhandled status code, the exact line number will break in your own workspace, 
allowing you to debug with complete visibility.

This tree demonstrates how data and control flows through the Python runtime. Unlike a hidden multi-agent 
pipeline, control explicitly branches based on programmatic conditionals (if/else loops) rather than LLM 
reasoning chains.

AlphaMiningPipeline.run_pipeline()
│
├── 1. [STATE] Research Phase (Sequential Initialization)
│   ├── System Input: High-level quantitative goal or sector target
│   ├── LLM Interaction (Flash Model): Reads goal -> Emits 3 precise RAG search queries
│   ├── Execution: Loops queries -> Invokes MergerRetriever -> Runs Flashrank Reranker
│   └── State Modification: Flushes raw tokens -> Mutates `self.research_context`
│
├── 2. [STATE] Ideation Phase (Creative Concept Formulation)
│   ├── System Input: `self.research_context`
│   ├── LLM Interaction (Creative Model): Reads context -> Proposes 3 mathematical strategies
│   └── State Modification: Enforces strict JSON -> Populates structural array `self.ideas_list`
│
└── 3. [STATE] Generation & Compilation Phase (Cyclic Processing Node)
    │
    └── FOR EACH: idea IN `self.ideas_list`
        ├── Step 3A: Programmatic Context Hydration (Zero Agent Guesswork)
        │   ├── Runs: tool_search_datafields(idea.keywords) -> Collects valid target fields
        │   ├── Runs: tool_search_operators() -> Collects matching valid operational tools
        │   └── Runs: tool_get_region_settings() -> Extracts target universe profiles (USA, IND)
        │
        └── Step 3B: Inner Compilation Loop (Max 4 Retries Allowed)
            │
            ├── LLM Interaction (Pro Model): Evaluates fields + operators + prior failure logs
            │   └── Output: Enforced submittable JSON payload block (Expression string + parameters)
            │
            ├── GATEWAY CHECK 1: Local Syntax Compiler (`_validate_regular_formula`)
            │   ├── [FAIL] ──> Log error text ──> Mutate `history_logs` ──> Continue to next retry
            │   └── [PASS] ──> Proceed to Gateway Check 2
            │
            ├── GATEWAY CHECK 2: Remote Sandbox Simulator (`simulate_and_evaluate_alpha`)
            │   ├── Connection: POST payload packet directly via API wrapper layer
            │   ├── Processing: Wait for server execution completion response
            │   │
            │   ├── BRANCH A: [IS_Checks == "PASS"]
            │   │   └── ACTION: Print validated portfolio alpha ──> Break loop ──> Terminate Success
            │   │
            │   └── BRANCH B: [IS_Checks == "FAIL"]
            │       ├── ACTION: Parse raw payload string
            │       ├── ACTION: Strip bulky text metadata -> Retain (Sharpe, Turnover, Error line)
            │       └── ACTION: Append pruned footprint to `history_logs` ──> Continue to next retry
            │
            └── LOOP TERMINATION GUARDS
                ├── Case: Max Retries Exhausted (Attempt == 4)
                │   └── Action: Log concept failure -> Advance index to `idea + 1` -> Reset context
                │
                └── Case: All Concepts Processed without an active Alpha Pass
                    └── Action: Log runtime warning -> Save debug metrics dump -> Safe shutdown

New Pure Python Framework vs. Old CrewAI Framework
├── [SECTION] Core Advantages of the New Pure Python Framework
│   ├── 1. Absolute Orchestration Determinism
│   │   ├── Old: Relies on the LLM's internal ReAct loop acting as a black-box scheduler (highly erratic; prone to hallucinating success to escape tasks)
│   │   └── New: Driven by strict Python looping logic (`for attempt in range()`), guaranteeing a hard-coded bound on exact execution cycles
│   ├── 2. Isolation of Context and Minimal Footprint
│   │   ├── Old: Framework automatically flushes entire unparsed JSON payloads from tools into agent history (causes "lost in the middle" and context bloat)
│   │   └── New: Script programmatically strips away API metadata, passing only core metric points (e.g., Sharpe, specific error line) back to the LLM
│   ├── 3. Native Backward Routing (True Cyclic State Loops)
│   │   ├── Old: Sequential chains are structurally forward-only; a validation step cannot route the flow back to the ideator for a clean pivot
│   │   └── New: Uses native Python conditionals (`if/else`) to route execution fluidly between syntax fixing, sim testing, and complete concept re-rolls
│   └── 4. Absolute Visibility & Clean Stack Traces
│       ├── Old: Tool runtime failures, API exceptions, and token timeouts disappear deep into abstract lifecycle layers, obscuring root bugs
│       └── New: System crashes throw standard python tracebacks pointing to the precise line number in your local codebase for immediate debugging
│
└── [SECTION] Core Disadvantages of the New Pure Python Framework
    ├── 1. Significant Software Engineering Overhead
    │   ├── Old: Built-in declarative abstractions handle model switches, memory allocation, and tool routing implicitly with minimal code
    │   └── New: Every part of the state schema, Pydantic type definitions, exception blocks, and looping logic must be written and maintained manually
    ├── 2. Complete Elimination of Emergent Agent Collaboration
    │   ├── Old: Agents can leverage autonomous delegation and cross-critique behaviors to tackle flexible, open-ended research tasks natively
    │   └── New: System is restricted to deterministic, single-turn prompting; agent multi-perspective debates must be explicitly coded
    ├── 3. Total Loss of Framework Ecosystem Tools
    │   ├── Old: Grants native hooks into observability dashboards, short/long-term agent memory features, and performance tracing telemetry
    │   └── New: Every metric log, file storage handler, terminal color decorator, and diagnostic audit trail must be engineered from scratch
    └── 4. Increased Maintenance Rigidity
        ├── Old: Altering or scaling the system layout requires simply adding or rearranging a dictionary object in a Task or Agent list array
        └── New: Adjusting how data flows or adding new analysis steps requires refactoring the structural logic inside the main class state machine
"""

import os
"""
By default, Hugging Face checks online for model updates. If your models are already downloaded, you can 
force LangChain and Hugging Face to run completely offline. This eliminates network latency entirely.
"""
# Force transformers and Hugging Face to skip online checks
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import sys
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import requests
from pathlib import Path
import datetime
import re
from langchain_chroma import Chroma
from langchain_classic.retrievers import MergerRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from flashrank import Ranker
from config.config import Abbr_To_Full, DELAY, UNIVERSE, NEUTRALIZATION_DICT
from config.api_key import (
    API_KEY_MOONSHOT, API_KEY_GEMINI_C26, API_KEY_GEMINI_CU, 
    API_KEY_DEEPSEEK, API_KEY_GOOGLE_CLOUD)
from utils.logger import setup_logger
from utils.clean_redundant_data import clean_community_data

# ====================== IMPORT FROM YOUR SIMULATOR ======================
from wqb_api.wqb_api_v1 import (
    initialize_global_variables,
    _validate_regular_formula,
    _build_simulation_payload,
    simulate_and_evaluate_alpha
)

# ====================== CONFIG & DIRECTORIES ======================
BASE_DIR = Path.cwd()
WQB_FORUM_CHINA_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_china_consultant_pdf"
WQB_FORUM_GLOBAL_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_global_consultant_pdf"
WQB_FORUM_RESEARCH_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_research_pdf"
WQB_FORUM_TIPS_PATH = BASE_DIR / "Docs" / "Forums" / "wqb_brain_tips_pdf"
WQB_OFFICIAL_DOCS_PATH = BASE_DIR / "Docs" / "OfficialDocs"
OPERATOR_FILE_PATH = BASE_DIR / "Operators" / "Operators-Agent.json"
DATAFIELDS_FILE_PATH = BASE_DIR / "DataFields" / "Datafield-Dataset-Category-Description.json"

EMBEDDING_DB_FORUM_CHINA_DIR = BASE_DIR / "embedding_db" / "wqb_forum_china_embedding_db"
EMBEDDING_DB_FORUM_GLOBAL_DIR = BASE_DIR / "embedding_db" / "wqb_forum_global_embedding_db"
EMBEDDING_DB_FORUM_RESEARCH_DIR = BASE_DIR / "embedding_db" / "wqb_forum_research_embedding_db"
EMBEDDING_DB_FORUM_TIPS_DIR = BASE_DIR / "embedding_db" / "wqb_forum_tips_embedding_db"
EMBEDDING_DB_OFFICIALDOCS_DIR = BASE_DIR / "embedding_db" / "wqb_official_docs_embedding_db"
EMBEDDING_DB_DIRECTORIES = {
    EMBEDDING_DB_FORUM_CHINA_DIR, EMBEDDING_DB_FORUM_GLOBAL_DIR,
    EMBEDDING_DB_FORUM_RESEARCH_DIR, EMBEDDING_DB_FORUM_TIPS_DIR, EMBEDDING_DB_OFFICIALDOCS_DIR
}
HF_CACHE_DIR = BASE_DIR / "cache" / "hf"
LOG_DIR = BASE_DIR / "logs" / datetime.datetime.now().strftime("%Y%m")

timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
HTML_FILE = LOG_DIR / f"wqb_agent_pure_llm-{timestamp}.html"

for directory in [HF_CACHE_DIR, LOG_DIR]: directory.mkdir(parents=True, exist_ok=True)

# ====================== LOAD JSON DATA ======================
with open(OPERATOR_FILE_PATH, 'r', encoding='utf-8') as f:
    operators_data = json.load(f)

with open(DATAFIELDS_FILE_PATH, 'r', encoding='utf-8') as f:
    datafields_data = json.load(f)

logger = setup_logger(LOG_DIR, "wqb_agent_pure_llm", "shared_logger")

# ====================== LLM ENDPOINT ROUTING (PURE HTTP) ======================
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

# Active Configurations (Switch providers here safely)
BASE_URL = base_local_googlecloud
PRO_MODEL = pro_googlecloud_model
FLASH_MODEL = flash_googlecloud_model
API_KEY = API_KEY_GOOGLE_CLOUD

# Configuration for Creative Tasks
CREATIVE_BASE_URL = base_local_googlecloud
CREATIVE_MODEL = pro_googlecloud_model
CREATIVE_API_KEY = API_KEY_GOOGLE_CLOUD

# TODO: simple and ultra-low cost llm for tool response parsing tasks to avoid using expensive 
# pro-tier models for basic JSON parsing and tool output formatting.
TOOL_BASE_URL = base_deepseek_url
TOOL_MODEL = flash_deepseek_model
TOOL_API_KEY = API_KEY_DEEPSEEK

# remove the code block sign markdown if present (not the code blocks inside the content)
def remove_code_block_markdown(text):
    """
    Removes ONLY the outer markdown code block delimiters (```json ... ```) 
    and leaves the raw code/content inside completely intact.
    """
    # Group 1 captures an optional language identifier (like 'json' or 'python')
    # Group 2 captures the actual content inside the code block
    pattern = r'```([a-zA-Z0-9_]*)\s*([\s\S]*?)\s*```'
    
    # Replace the whole match with just the internal content (Group 2)
    return re.sub(pattern, r'\2', text)

def call_llm(base_url, model, api_key, system_prompt, user_prompt, temperature=0.2, require_json=False, remove_markdown=False):
    """Deterministic, framework-free wrapper for OpenAI-compatible chat endpoints."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 4096
    }
    if require_json: payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            if remove_markdown:
                content_clean = remove_code_block_markdown(content)
                return content_clean.strip()
            return content.strip()
        else:
            logger.error("LLM API", f"Status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error("LLM API", f"Exception during call: {str(e)}")
        return None

# ====================== RAG PIPELINE SETUP ======================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    cache_folder=str(HF_CACHE_DIR),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=False
)

def combine_embedding_databases(directories, embed_model):
    retrievers = []
    for db_dir in directories:
        if db_dir.exists():
            vectorstore = Chroma(persist_directory=str(db_dir), embedding_function=embed_model)
            retrievers.append(vectorstore.as_retriever(search_kwargs={"k": 6}))
    return MergerRetriever(retrievers=retrievers)

base_combined_retriever = combine_embedding_databases(EMBEDDING_DB_DIRECTORIES, embeddings)
flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(HF_CACHE_DIR))
compressor = FlashrankRerank(client=flashrank_client, top_n=5)
final_ranker_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_combined_retriever)
logger.info("Retriever Initialization", "🚀 Advanced Retriever-Ranker pipeline successfully wired up.")

# ====================== EXPLICIT PYTHON TOOLS ======================
def tool_retrieve_text_data(query: str) -> str:
    """Fetches relevant text snippets from the WorldQuant knowledge base.
    Args:
        query (str): Highly specific search phrase. Be precise.
                     Good examples: "high Sharpe analyst revisions", 
                     "group_neutralize turnover control", "fq1_cons_eps_revisions template"
    Returns:
        str: Concatenated relevant document snippets separated by ---.
             Returns "No relevant documents found." if nothing matches.
    """
    try:
        docs = final_ranker_retriever.invoke(query)
        output = "\n---\n".join([doc.page_content for doc in docs])
        return clean_community_data(output)
    except Exception as e:
        return f"Error retrieving text: {e}"

def tool_search_operators(query: str) -> str:
    """Search for WorldQuant BRAIN operators and their exact syntax.
    Args:
        query (str): Operator name or concept (e.g., "decay_linear", "group_neutralize", "signed_power")
    Returns:
        Formatted list of matching operators with syntax and description.
    """
    results = []
    query_lower = query.lower()
    for op_name, op_details in operators_data.items():
        if query_lower in op_name.lower() or query_lower in op_details.get('description', '').lower():
            results.append(f"Operator: {op_name}\nSyntax: {op_details.get('definition')}\nDesc: {op_details.get('description')}")
        if len(results) >= 10: break
    return "\n---\n".join(results) if results else "No matching operators found."

def tool_search_datafields(query: str) -> str:
    """Search for WorldQuant data fields/datasets.
    Args:
        query (str): Keyword, field name, or concept (e.g., "eps revision", "esg", "insider buying")
    Returns:
        Formatted list of matching data fields with description and regions.
    """
    results = []
    query_lower = query.lower()
    for field_name, field_details in datafields_data.items():
        if query_lower in field_name.lower() or query_lower in field_details.get('description', '').lower():
            abbr_regions = field_details.get('region', [])
            regions = [Abbr_To_Full.get(r, r) for r in abbr_regions]
            results.append(f"Field: {field_name} | Desc: {field_details.get('description')} | Regions: {', '.join(regions)}")
        if len(results) >= 15: break
    return "\n---\n".join(results) if results else "No direct datafield matches found."

def tool_get_all_region_settings() -> str:
    """Returns allowed configuration (Universe, Delay, Neutralization) for a given region.
    Input:
        region: A 3-letter region string, including: "USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"
    """
    regions = ["USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"]
    output = ""
    rules = {}
    for reg in regions:
        rules[reg] = {
            "delay": DELAY.get(reg, []),
            "universe": UNIVERSE.get(reg, []),
            "neutralization": NEUTRALIZATION_DICT.get(reg, [])
        }
    for region, config in rules.items():
        output += f"{region}:\n{config}\n\n"
    return output

# Initialize simulator components
account_no = "0"
fail, init_msg = initialize_global_variables(account_no=account_no)
if fail:
    logger.error("Simulator Init", f"Failed: {init_msg}")
    exit(1)

# ====================== DETERMINISTIC STATE MACHINE PIPELINE ======================
class AlphaMiningPipeline:
    def __init__(self, user_request):
        self.user_request = user_request
        self.research_context = ""
        self.ideas_list = []
        
    def run_pipeline(self):
        logger.info("Pipeline", "🏁 Starting Deterministic Alpha Mining Run...")
        
        # STATE 1: Research Phase (Sequential Execution)
        self.execute_research_state()
        if not self.research_context:
            logger.error("Pipeline", "❌ Research state failed to accumulate context. Exiting.")
            return

        # STATE 2: Ideation Phase
        self.execute_ideation_state()
        if not self.ideas_list:
            logger.error("Pipeline", "❌ Ideation failed to yield valid structured ideas. Exiting.")
            return

        # STATE 3 & 4: Cyclic Generation and Programmatic Compilation Loop
        for index, idea in enumerate(self.ideas_list):
            logger.info("Pipeline", f"➡️ Processing Alpha Idea {index + 1}/{len(self.ideas_list)}")
            success = self.execute_generation_and_validation_loop(idea)
            if success:
                logger.info("Pipeline", f"🎉 Successfully found and verified a passed Alpha for Idea {index + 1}!")
                return
            else:
                logger.warning("Pipeline", f"⚠️ Idea {index + 1} failed all retry validation limits. Proceeding to next concept.")
                
        logger.error("Pipeline", "❌ All ideas exhausted without extracting a passing alpha expression.")

    def execute_research_state(self):
        logger.info("Research State", "🔎 Generating targeted RAG queries...")
        sys_prompt = "You are a WorldQuant Alpha Researcher. Your task is to look at a user goal and generate 3 highly distinct, precise search keywords or codes to find useful alpha templates in the vector store."
        user_prompt = f"Goal:\n{self.user_request}\n\nOutput your response strictly as a JSON object with this key: 'queries': [ 'query1', 'query2', 'query3' ]"
        
        response = call_llm(BASE_URL, FLASH_MODEL, API_KEY, sys_prompt, user_prompt, temperature=0.1, require_json=True, remove_markdown=True)
        try:
            queries = json.loads(response).get('queries', [])
            accumulated_text = []
            for q in queries:
                logger.info("Research State", f"Executing RAG Query: '{q}'")
                result = tool_retrieve_text_data(q)
                if result and "No relevant documents" not in result:
                    accumulated_text.append(result)
                    logger.info("Research State", f"Query '{q}' returned relevant context. Added to research pool.")
            self.research_context = "\n\n=== NEW DB SOURCE ===\n\n".join(accumulated_text)
            logger.info("Research State", f"Successfully gathered {len(accumulated_text)} historical document pieces.")
        except Exception as e:
            logger.error("Research State", f"Failed to parse research queries JSON: {e}")

    def execute_ideation_state(self):
        logger.info("Ideation State", "💡 Generating mathematical alpha hypotheses via Creative LLM...")
        sys_prompt = "You are a senior quantitative portfolio manager. Propose 3 distinct, creative alpha ideas utilizing data fields and strategies found in the research documentation provided. Do not invent non-existent fields."
        user_prompt = f"Research Documentation Context:\n{self.research_context}\n\nOutput a strict JSON object with this key: 'ideas': [ {{'name': '...', 'hypothesis': '...', 'keywords_for_fields': '...'}} ]"
        
        response = call_llm(CREATIVE_BASE_URL, CREATIVE_MODEL, CREATIVE_API_KEY, sys_prompt, user_prompt, temperature=0.7, require_json=True, remove_markdown=True)
        try:
            self.ideas_list = json.loads(response).get('ideas', [])
            logger.info("Ideation State", f"Generated {len(self.ideas_list)} potential concepts.")
        except Exception as e:
            logger.error("Ideation State", f"Failed parsing Ideation response JSON: {e}")

    def execute_generation_and_validation_loop(self, idea):
        # Programmatic Tool Execution prior to prompting the generation LLM
        # This prevents the model from needing to decide when/how to run tools.
        logger.info("Generation Loop", f"Programmatically assembling tool references for concept: {idea['name']}")
        fields_found = tool_search_datafields(idea['keywords_for_fields'])
        operators_found = tool_search_operators("rank decay delta mean corr group")
        
        # We defaults to standard configurations but allow the agent to evaluate USA / IND setups
        allowed_region_settings = tool_get_all_region_settings()
        
        history_logs = []
        max_retries = 4
        
        for attempt in range(1, max_retries + 1):
            logger.info("Generation Loop", f"🔄 Attempt {attempt}/{max_retries} for execution expression creation...")
            
            sys_prompt = """You are an expert WorldQuant Formula Writer. You map a quantitative hypothesis into a single-line mathematical string syntax.
            CRITICAL RULES:
            - Output your entire calculation as ONE single string expression layout. Do not use local multiline variable assignments unless mandatory.
            - Ensure any parenthesis opened are perfectly balanced and enclosed.
            - Use valid operators and matched datafields from the provided lists.
            - Output a strict JSON response format with these exact keys:
              "region": "...", "universe": "...", "delay": 1, "decay": 12, "neutralization": "...", "truncation": 0.01, "pasteurization": "ON", "nanHandling": "OFF", "testPeriod": "P1Y", "maxTrade": "OFF", "maxPosition": "OFF",
              "expression": "..."
            """
            
            user_prompt = f"""
            Hypothesis Idea: {idea['hypothesis']}
            
            Validated Data Fields Available:
            {fields_found}
            
            Validated Operators Reference:
            {operators_found}
            
            Allowed Simulation Parameter Profiles:
            {allowed_region_settings}
            
            Prior Attempt Errors History (Clean Execution State Loop):
            {json.dumps(history_logs, indent=2)}
            
            Generate a submittable configuration payload block following the strict schema requested.
            """
            
            response = call_llm(BASE_URL, PRO_MODEL, API_KEY, sys_prompt, user_prompt, temperature=0.2, require_json=True, remove_markdown=True)
            if not response: continue
                
            try:
                payload = json.loads(response)
                expr = payload.get("expression")
                
                # Setup settings payload dictionary cleanly for simulation execution
                settings = {k: v for k, v in payload.items() if k != "expression"}
                if "instrumentType" not in settings: settings["instrumentType"] = "EQUITY"
                
                logger.info("Compilation Step", f"Testing syntax expression: {expr}")
                
                # Check 1: Pure Syntax Formula Local Validator Call
                fail_local, result_local = _validate_regular_formula(
                    regular_formula=expr,
                    region=settings["region"],
                    delay=int(settings["delay"]),
                    universe=settings["universe"],
                    account_no=account_no
                )
                
                if fail_local:
                    logger.warning("Compilation Step", f"❌ Local Syntax Failure: {result_local}")
                    history_logs.append({"attempt": attempt, "stage": "syntax_validation", "error": result_local, "failed_expression": expr})
                    continue
                
                # Check 2: API Full Simulator Sandbox Call
                logger.info("Simulation Step", "Sending expression payload to WorldQuant Brain Simulator API...")
                fail_sim, sim_payload = _build_simulation_payload(settings, expr, account_no=account_no)
                if fail_sim:
                    logger.warning("Simulation Step", f"❌ Payload mapping failure: {sim_payload}")
                    history_logs.append({"attempt": attempt, "stage": "payload_build", "error": sim_payload, "failed_expression": expr})
                    continue
                    
                fail_api, result_api = simulate_and_evaluate_alpha(
                    alpha_settings=sim_payload, regular=expr, account_no=account_no, include_self_corr=True, include_prod_corr=True
                )
                
                if fail_api:
                    logger.warning("Simulation Step", f"❌ API Transaction Fault: {result_api}")
                    history_logs.append({"attempt": attempt, "stage": "api_endpoint_error", "error": result_api, "failed_expression": expr})
                    continue
                
                # Parse metrics cleanly out of simulation engine result
                # result_api can either be a dictionary or a JSON string depending on the wrapper layer
                sim_data = json.loads(result_api) if isinstance(result_api, str) else result_api
                logger.info("Simulation Step", f"Raw Simulator Response Metadata: {json.dumps(sim_data)}")
                
                # Standard check structure alignment evaluation
                if sim_data.get("IS_Checks") == "PASS":
                    logger.info("Simulation Step", "🔥 SUCCESS! Alpha passed all simulator criteria constraints.")
                    print_final_alpha(idea, settings, expr, sim_data)
                    return True
                else:
                    error_summary = sim_data.get("Log", "Sharpe or Fitness checks failed minimum parameter threshold limits.")
                    logger.warning("Simulation Step", f"❌ Alpha failed simulation constraints: {error_summary}")
                    # Prune the JSON logs to pass only the essential data to avoid filling the context window
                    pruned_metrics = {
                        "Sharpe": sim_data.get("Sharpe"),
                        "Fitness": sim_data.get("Fitness"),
                        "Turnover": sim_data.get("Turnover"),
                        "Message": error_summary
                    }
                    history_logs.append({"attempt": attempt, "stage": "simulation_metrics_fail", "metrics_feedback": pruned_metrics, "failed_expression": expr})
                    
            except Exception as parse_err:
                logger.error("Generation Loop", f"Error evaluating loop index output trace: {parse_err}")
                history_logs.append({"attempt": attempt, "stage": "code_exception_parse", "error": str(parse_err)})
                
        return False

def print_final_alpha(idea, settings, expr, metrics):
    logger.info("Final Alpha", f"\n{'='*60}\nVALIDATED MINED PORTFOLIO ALPHA\n{'='*60}")
    logger.info("Final Alpha", f"**Alpha Name:** {idea['name']}")
    logger.info("Final Alpha", f"**Economic Hypothesis:** {idea['hypothesis']}")
    logger.info("Final Alpha", f"**Target Settings:** Region: {settings['region']} | Universe: {settings['universe']} | Neutralization: {settings['neutralization']} | Delay: {settings['delay']} | Decay: {settings['decay']}")
    logger.info("Final Alpha", f"**Full BRAIN Expression:** {expr}")
    logger.info("Final Alpha", f"**Performance Summary:** Sharpe: {metrics.get('Sharpe')} | Fitness: {metrics.get('Fitness')} | Turnover: {metrics.get('Turnover')}")
    logger.info("Final Alpha", f"{'='*60}\n")

# ====================== PIPELINE ORCHESTRATION EXECUTION ======================
if __name__ == "__main__":
    user_request_prompt = """
    GOAL: Discover a highly successful alpha strategy from the consultant tips and forum discussions, and build a submittable alpha based on it. 
    Focus on replicating the scaling framework rank(A)/rank(B) approach shown in past work for fundamentals or use technical non-linear structures.
    """
    miner = AlphaMiningPipeline(user_request=user_request_prompt.strip())
    miner.run_pipeline()

"""
Example Log

[26-6-2 13:37:39][INFO][Pipeline] 🏁 Starting Deterministic Alpha Mining Run...
[26-6-2 13:37:39][INFO][Research State] 🔎 Generating targeted RAG queries...
[26-6-2 13:37:47][INFO][Research State] Executing RAG Query: 'successful rank(A)/rank(B) fundamentals consultant tips'
[26-6-2 13:37:53][INFO][Research State] Query 'successful rank(A)/rank(B) fundamentals consultant tips' returned relevant context. Added to research pool.
[26-6-2 13:37:53][INFO][Research State] Executing RAG Query: 'high-performance non-linear technical alpha strategy'
[26-6-2 13:37:58][INFO][Research State] Query 'high-performance non-linear technical alpha strategy' returned relevant context. Added to research pool.
[26-6-2 13:37:58][INFO][Research State] Executing RAG Query: 'forum discussions rank ratio scaling framework'
[26-6-2 13:38:03][INFO][Research State] Query 'forum discussions rank ratio scaling framework' returned relevant context. Added to research pool.
[26-6-2 13:38:03][INFO][Research State] Successfully gathered 3 historical document pieces.
[26-6-2 13:38:03][INFO][Ideation State] 💡 Generating mathematical alpha hypotheses via Creative LLM...
[26-6-2 13:38:40][INFO][Ideation State] Generated 3 potential concepts.
[26-6-2 13:38:40][INFO][Pipeline] ➡️ Processing Alpha Idea 1/3
[26-6-2 13:38:40][INFO][Generation Loop] Programmatically assembling tool references for concept: Dynamic Value Re-Ranking Momentum
[26-6-2 13:38:40][INFO][Generation Loop] 🔄 Attempt 1/4 for execution expression creation...
[26-6-2 13:39:15][INFO][Compilation Step] Testing syntax expression: rank(book_to_price) * rank(ts_delta(rank(book_to_price), 20))
[26-6-2 13:39:15][INFO][Check Regular Format (0)] ✅ the regular formula is parsed successfully.
[26-6-2 13:39:15][INFO][Check Regular Format (0)] Datafields: ['book_to_price']
[26-6-2 13:39:15][INFO][Check Regular Format (0)] Operators: ['ts_delta', 'rank']
[26-6-2 13:39:15][ERROR][Check Regular Format (0)] ❌ book_to_price doesn't exist.
[26-6-2 13:39:15][WARNING][Compilation Step] ❌ Local Syntax Failure: ❌ book_to_price doesn't exist.
[26-6-2 13:39:15][INFO][Generation Loop] 🔄 Attempt 2/4 for execution expression creation...
[26-6-2 13:39:36][INFO][Compilation Step] Testing syntax expression: rank(earnings_yield) * rank(ts_delta(rank(earnings_yield), 20))
[26-6-2 13:39:36][INFO][Check Regular Format (0)] ✅ the regular formula is parsed successfully.
[26-6-2 13:39:36][INFO][Check Regular Format (0)] Datafields: ['earnings_yield']
[26-6-2 13:39:36][INFO][Check Regular Format (0)] Operators: ['ts_delta', 'rank']
[26-6-2 13:39:36][ERROR][Check Regular Format (0)] ❌ earnings_yield doesn't exist.
[26-6-2 13:39:36][WARNING][Compilation Step] ❌ Local Syntax Failure: ❌ earnings_yield doesn't exist.
[26-6-2 13:39:36][INFO][Generation Loop] 🔄 Attempt 3/4 for execution expression creation...
[26-6-2 13:39:53][INFO][Compilation Step] Testing syntax expression: rank(ebitda_to_enterprise_value) * rank(ts_delta(rank(ebitda_to_enterprise_value), 20))
[26-6-2 13:39:53][INFO][Check Regular Format (0)] ✅ the regular formula is parsed successfully.
[26-6-2 13:39:53][INFO][Check Regular Format (0)] Datafields: ['ebitda_to_enterprise_value']
[26-6-2 13:39:53][INFO][Check Regular Format (0)] Operators: ['ts_delta', 'rank']
[26-6-2 13:39:53][ERROR][Check Regular Format (0)] ❌ ebitda_to_enterprise_value doesn't exist.
[26-6-2 13:39:53][WARNING][Compilation Step] ❌ Local Syntax Failure: ❌ ebitda_to_enterprise_value doesn't exist.
[26-6-2 13:39:53][INFO][Generation Loop] 🔄 Attempt 4/4 for execution expression creation...
[26-6-2 13:40:17][INFO][Compilation Step] Testing syntax expression: rank(1 / pe_ratio) * rank(ts_delta(rank(1 / pe_ratio), 20))
[26-6-2 13:40:17][INFO][Check Regular Format (0)] ✅ the regular formula is parsed successfully.
[26-6-2 13:40:17][INFO][Check Regular Format (0)] Datafields: ['pe_ratio']
[26-6-2 13:40:17][INFO][Check Regular Format (0)] Operators: ['ts_delta', 'rank']
[26-6-2 13:40:17][ERROR][Check Regular Format (0)] ❌ pe_ratio doesn't exist.
[26-6-2 13:40:17][WARNING][Compilation Step] ❌ Local Syntax Failure: ❌ pe_ratio doesn't exist.
[26-6-2 13:40:17][WARNING][Pipeline] ⚠️ Idea 1 failed all retry validation limits. Proceeding to next concept.
[26-6-2 13:40:17][INFO][Pipeline] ➡️ Processing Alpha Idea 2/3
[26-6-2 13:40:17][INFO][Generation Loop] Programmatically assembling tool references for concept: Sector-Neutralized International Trade Dominance
[26-6-2 13:40:17][INFO][Generation Loop] 🔄 Attempt 1/4 for execution expression creation...
[26-6-2 13:40:43][INFO][Compilation Step] Testing syntax expression: group_neutralize(rank(ts_delta(close, 20)), industry)
[26-6-2 13:40:43][INFO][Check Regular Format (0)] ✅ the regular formula is parsed successfully.
[26-6-2 13:40:43][INFO][Check Regular Format (0)] Datafields: ['close', 'industry']
[26-6-2 13:40:43][INFO][Check Regular Format (0)] Operators: ['group_neutralize', 'ts_delta', 'rank']
[26-6-2 13:40:43][INFO][Check Regular Format (0)] Datasetname Series: {'Price Volume'}
[26-6-2 13:40:43][INFO][Check Regular Format (0)] Region-Delay-Universe combinations: ['EUR-0-ILLIQUID_MINVOL1M', 'EUR-0-TOP1200', 'USA-0-TOP1000', 'MEA-1-TOP400', 'USA-1-TOPSP500', 'CHN-0-TOP2000U', 'EUR-0-TOP2500', 'USA-0-ILLIQUID_MINVOL1M', 'USA-1-TOP200', 'USA-0-TOP3000', 'EUR-0-TOP400', 'EUR-0-TOP800', 'USA-1-TOP3000', 'USA-0-TOP500', 'USA-1-TOP500', 'EUR-1-TOPCS1600', 'EUR-1-TOP2500', 'USA-1-TOP1000', 'TWN-1-TOP100', 'ASI-1-MINVOL1M', 'KOR-1-TOP600', 'ASI-1-ILLIQUID_MINVOL1M', 'GLB-1-TOPDIV3000', 'USA-1-ILLIQUID_MINVOL1M', 'GLB-1-TOP3000', 'EUR-1-ILLIQUID_MINVOL1M', 'GLB-1-MINVOL1M', 'EUR-0-TOPCS1600', 'USA-0-TOP200', 'EUR-1-TOP1200', 'MEA-1-TOP300', 'EUR-1-TOP400', 'IND-1-TOP500', 'CHN-1-TOP2000U', 'USA-0-TOPSP500', 'EUR-1-TOP800', 'TWN-1-TOP500']
[26-6-2 13:40:43][INFO][Check Regular Format (0)] ✅ Input region, delay, universe 'USA-1-TOP3000' matches the expected combinations.
[26-6-2 13:40:43][INFO][Check Regular Format (0)] Combined Multiplier: 1.1
[26-6-2 13:40:43][INFO][Simulation Step] Sending expression payload to WorldQuant Brain Simulator API...
"""

"""
Problem

The log output proves that your Pure Python State Machine is orchestrating exactly as designed. 
It successfully managed sequential RAG queries, transitioned to ideation, caught syntax validation 
errors via try-except wrappers, safely isolated the history footprint, and cleanly routed execution 
forward to Idea 2 without crashing.

However, Idea 1 fell into a blind hallucination loop during its 4 retries.

The Core Bottleneck: "Blind" Retries
Your framework relies on Programmatic Tool Execution prior to entering the loop:

```python
fields_found = tool_search_datafields(idea['keywords_for_fields'])
...
for attempt in range(1, max_retries + 1):
    # The prompt receives the exact same 'fields_found' block every time
```

If the initial ideation keywords (e.g., "Dynamic Value Re-Ranking Momentum") caused tool_search_datafields 
to return weak or empty results, the Pro LLM was forced to guess variables (book_to_price, earnings_yield) 
based on its pre-trained finance knowledge. When the compiler responded with ❌ book_to_price doesn't exist, 
the LLM had no visibility into what fields do exist, causing it to guess another non-existent generic variable 
on every single retry.
"""