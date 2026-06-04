"""
Version 2

1. Dynamic Programmatic Context Hydration (Adaptive Loop)
Instead of keeping fields_found static, use standard Python regex inside the retry loop to parse the 
compiler's error string. If a field fails, strip its tokens and programmatically run a fallback database 
search to inject real choices into the next prompt turn.

2. Add submitted alphas info

3. Fix is_checks error
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
VERSION = "2.0"
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
logger.info("Initialization", f"✅ Configuration, directories, and data successfully loaded. 🎈 Version: {VERSION}")

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

PASSED_ALPHAS_REPOSITORY = """
...
"""

# remove the code block sign markdown if present (not the code blocks inside the content)
# def remove_code_block_markdown(text):
#     """
#     Removes ONLY the outer markdown code block delimiters (```json ... ```) 
#     and leaves the raw code/content inside completely intact.
#     """
#     # Group 1 captures an optional language identifier (like 'json' or 'python')
#     # Group 2 captures the actual content inside the code block
#     pattern = r'```([a-zA-Z0-9_]*)\s*([\s\S]*?)\s*```'
#     
#     # Replace the whole match with just the internal content (Group 2)
#     return re.sub(pattern, r'\2', text)

def remove_code_block_markdown(text):
    """
    Removes ONLY the outer markdown code block delimiters and handles 
    cases where formatting might be slightly off.
    """
    text = text.strip()
    if text.startswith("```"):
        # Split by newlines, drop the first line (```json)
        lines = text.split('\n')
        if len(lines) > 1:
            lines = lines[1:]
        # Drop the last line if it's just the closing backticks
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text

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
        "max_tokens": 8192 # <-- Bump this up to at least 8192, or remove it to use the provider default
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
        if (query_lower in op_name.lower() or 
            query_lower in op_details.get('description', '').lower() or 
            query_lower in op_details.get('category', '').lower()):
            
            res = f"Operator: {op_name}\nSyntax: {op_details.get('definition')}\nDesc: {op_details.get('description')}"
            results.append(res)
        if len(results) >= 15:  # Limit results to save context window
            break
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
                    result_print = result.replace("\n", " ")[:2000]
                    logger.info("Research State", f"Query '{q}' returned relevant context. Added to research pool. (Total length: {len(accumulated_text)}) {result_print}...")
            self.research_context = "\n\n=== NEW DB SOURCE ===\n\n".join(accumulated_text)
            logger.info("Research State", f"Successfully gathered {len(accumulated_text)} historical document pieces.")
        except Exception as e:
            logger.error("Research State", f"Failed to parse research queries JSON: {e}")

    def execute_ideation_state(self):
        logger.info("Ideation State", "💡 Generating mathematical alpha hypotheses via Creative LLM...")
        sys_prompt = "You are a senior quantitative portfolio manager. Propose 3 distinct, creative alpha ideas utilizing data fields and strategies found in the research documentation provided. Do not invent non-existent fields."
        user_prompt = f"""
        Historical Success Repository (Emulate this depth of engineering):
        {PASSED_ALPHAS_REPOSITORY}
        
        New Research Documentation Context:
        {self.research_context}
        
        Output a strict JSON object with this key: 'ideas': [ {{'name': '...', 'hypothesis': '...', 'keywords_for_fields': '...'}} ]
        """
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
        fields_found_print = fields_found.replace("\n", " ")[:2000]
        logger.info("Generation Loop", f"Data Fields Search Result: (Total length: {len(fields_found)}) {fields_found_print}...")  # Log a preview of results
        operators_found = tool_search_operators("rank decay delta mean corr group")
        operators_found_print = operators_found.replace("\n", " ")[:2000]
        logger.info("Generation Loop", f"Operators Search Result: (Total length: {len(operators_found)}) {operators_found_print}...")  # Log a preview of results
        
        # We defaults to standard configurations but allow the agent to evaluate USA / IND setups
        allowed_region_settings = tool_get_all_region_settings()
        
        history_logs = []
        max_retries = 4
        
        for attempt in range(1, max_retries + 1):
            logger.info("Generation Loop", f"🔄 Attempt {attempt}/{max_retries} for execution expression creation...")

            # --- ADAPTIVE PROGRAMMATIC FIX: Parse errors and fetch real replacements ---
            if history_logs and history_logs[-1].get("stage") == "syntax_validation":
                last_error = history_logs[-1].get("error", "")
                # Regex to find whatever field caused the compiler to reject the formula
                match = re.search(r"❌\s*(\w+)\s*doesn't\s*exist", last_error)
                if match:
                    missing_field = match.group(1)
                    logger.info("Generation Loop", f"🛠️ Adaptive Fallback: Querying replacements for failed field '{missing_field}'")
                    
                    # Split 'book_to_price' into ['book', 'price'] to run semantic component searches
                    search_tokens = [t for t in missing_field.split('_') if len(t) > 2]
                    fallback_pool = []
                    for token in search_tokens:
                        res = tool_search_datafields(token)
                        if "No direct datafield matches found" not in res:
                            fallback_pool.append(res)
                    
                    if fallback_pool:
                        fields_found += "\n\n=== REPLACEMENT OPTIONS FOR PREVIOUS FAILURE ===\n" + "\n".join(fallback_pool)

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
            Golden Syntax Benchmarks:
            {PASSED_ALPHAS_REPOSITORY}

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
                if sim_data.get("is_checks", {}).get("Status", "") == "PASS":
                    logger.info("Simulation Step", "🔥 SUCCESS! Alpha passed all simulator criteria constraints.")
                    print_final_alpha(idea, settings, expr, sim_data)
                    return True
                else:
                    logger.warning("Simulation Step", f"❌ Alpha test failed")
                    # Prune the JSON logs to pass only the essential data to avoid filling the context window
                    history_logs.append({"attempt": attempt, "stage": "simulation_metrics_fail", "metrics_feedback": sim_data, "failed_expression": expr})
                    
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