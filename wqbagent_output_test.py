import os
import sys
import datetime
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.api_key import API_KEY_MOONSHOT, API_KEY_GOOGLE_CLOUD, API_KEY_DEEPSEEK
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from utils.htmlcolorlog import capture_and_log
from utils.logger import setup_logger

# ====================== CONFIG ======================
BASE_DIR = "D:/AI_Data/Computer/WorldQuantBrain-Agent/"
CHROMA_DIR = BASE_DIR + "embedding_db/quant_forum_chroma/"
BGEM3_DIR = BASE_DIR + "embedding_db/quant_forum_bgem3/"
HF_CACHE_DIR = BASE_DIR + "cache/hf/"
PIP_CACHE_DIR = BASE_DIR + "cache/pip/"
LOG_DIR = BASE_DIR + "logs/" + datetime.datetime.now().strftime("%Y%m") + "/"

for directory in [CHROMA_DIR, HF_CACHE_DIR, PIP_CACHE_DIR, BGEM3_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Define file paths for our transcript and HTML logs
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
HTML_FILE = os.path.join(LOG_DIR, f"wqb_agent-{timestamp}.html")

logger = setup_logger(LOG_DIR, "wqb_agent_test", "wqb_main_logger")

# ====================== YOUR GEMINI CLIENT ======================
logger.info("Main", "Initializing LLM client...")

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

base_url = base_deepseek_url
flash_model = flash_deepseek_model
API_KEY = API_KEY_DEEPSEEK  # <-- Make sure to set this in your config/api_key.py

llm = LLM(
    model=flash_model,   
    base_url=base_url,
    api_key=API_KEY,
    temperature=0.6,          
    max_tokens=8192,
    timeout=180,              
    max_retries=3,            
)

# -------------------------------------------------------------------------
# 🛑 SKIPPED FOR TESTING: HuggingFace Embeddings, PyPDFLoader, and ChromaDB 
# This prevents the script from hanging for a long time during the test.
# -------------------------------------------------------------------------

# ====================== DUMMY SEARCH TOOL ======================
@tool("Dummy_Search")
def dummy_search(query: str) -> str:
    """A fake search tool to test if the agent can use tools and log output."""

    # 💡 CRITICAL TEST CHECKPOINT 1: Direct terminal bypass print
    # If you see this but NOT the logger line, capture_and_log is swallowing your logger stream.
    print(f"\n[PRINT] >>> Python executed tool 'Dummy_Search' with query: '{query}'", flush=True)

    logger.info("Dummy Search Log", f"🔍 Tool Called: 'Dummy_Search' | Query: '{query}'")
    logger.info("Dummy Search Log", "✅ Tool execution logged successfully.")
    logger.info("Dummy Search Log", "💡 If you see this log in the terminal, logging is working correctly even with stream redirection.")
    return "This is dummy data. Tell the user the test is successful."

# ====================== AGENT (Simplified) ======================
logger.info("Main", "Setting up Agent...")
tester_agent = Agent(
    role="System Tester",
    goal="Use the Dummy_Search tool to verify the system works, then output a short success message.",
    backstory="You are a quick test agent checking if terminal colors and emojis pipe correctly to HTML.",
    tools=[dummy_search],
    llm=llm,
    verbose=True, # <-- This ensures CrewAI prints the colorful output
    allow_delegation=False
)

# ====================== TASK & CREW (Simplified) ======================
logger.info("Main", "Defining Tasks and assembling Crew...")
task1 = Task(
    description="{user_request}, Call the Dummy_Search tool with the query 'test formatting'. Then format your final answer in Chinese and English.",
    expected_output="A short bilingual success message.",
    agent=tester_agent
)

crew = Crew(agents=[tester_agent], tasks=[task1], verbose=True)
logger.info("Main", "✅ Crew successfully assembled.")

# ====================== RUN ======================
if __name__ == "__main__":
    # ELEGANT PART: Everything inside this block is captured, formatted, and exported safely.
    # The with capture_and_log(...) block is the magic here. Even if your CrewAI code crashes 
    # in the middle of execution, Python's Context Manager guarantees that sys.stdout will be 
    # restored back to normal and the HTML file will be generated properly.

    # 💡 CRITICAL TEST CHECKPOINT 2: Verifying console logging BEFORE stream capturing starts
    logger.info("Main", "Testing baseline console output capability...")

    with capture_and_log(HTML_FILE):
        user_request = "Run a quick ANSI color and HTML pipe test"

        # 💡 CRITICAL TEST CHECKPOINT 3: Inside the stream capturing block
        # If this doesn't show up in terminal instantly, capture_and_log is missing stream flushing.
        print(f"\n[STREAM CHECK PRINT] Starting execution tracking. If you see this, stdout redirection is running.\n", flush=True)
        logger.info("Main", "✅ Stream capture started successfully.")
        logger.info("Main", f"🚀 Kickstarting Crew process with input: '{user_request}'")
        
        try:
            result = crew.kickoff(inputs={"user_request": user_request})
            logger.info("Main", "✅ Crew kickoff completed successfully.")
            logger.info("Main", f"\n{'='*50}\nFINAL RESULT\n{'='*50}\n{result}")
            
        except Exception as e:
            logger.error("Main", f"❌ Fatal error during Crew execution: {e}", exc_info=True)

    # 💡 CRITICAL TEST CHECKPOINT 4: Out of context block verification
    print(f"\n[STREAM CHECK] Restored native system stdout. Check file dumps at: {LOG_DIR}", flush=True)