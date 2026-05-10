import os
import sys
import datetime
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import API_KEY_MOONSHOT
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from utils.htmlcolor import capture_and_log
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
TRANSCRIPT_FILE = os.path.join(LOG_DIR, f"wqb_agent-{timestamp}.transcript.txt")
HTML_FILE = os.path.join(LOG_DIR, f"wqb_agent-{timestamp}.html")

logger = setup_logger(LOG_DIR, "wqb_agent_test", "wqb_main_logger")

# ====================== YOUR GEMINI CLIENT ======================
logger.info("Initializing LLM client...")
llm = LLM(
    model="moonshot/moonshot-v1-32k",   
    base_url="https://api.moonshot.cn/v1",
    api_key=API_KEY_MOONSHOT,
    temperature=0.6,          
    max_tokens=8192,
    timeout=180,              
    max_retries=5,            
)

# -------------------------------------------------------------------------
# 🛑 SKIPPED FOR TESTING: HuggingFace Embeddings, PyPDFLoader, and ChromaDB 
# This prevents the script from hanging for a long time during the test.
# -------------------------------------------------------------------------

# ====================== DUMMY SEARCH TOOL ======================
@tool("Dummy_Search")
def dummy_search(query: str) -> str:
    """A fake search tool to test if the agent can use tools and log output."""
    logger.info(f"🔍 Tool Called: 'Dummy_Search' | Query: '{query}'")
    return "This is dummy data. Tell the user the test is successful."

# ====================== AGENT (Simplified) ======================
logger.info("Setting up Agent...")
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
logger.info("Defining Tasks and assembling Crew...")
task1 = Task(
    description="Call the Dummy_Search tool with the query 'test formatting'. Then format your final answer in Chinese and English.",
    expected_output="A short bilingual success message.",
    agent=tester_agent
)

crew = Crew(agents=[tester_agent], tasks=[task1], verbose=True)
logger.info("✅ Crew successfully assembled.")

# ====================== RUN ======================
if __name__ == "__main__":
    # ELEGANT PART: Everything inside this block is captured, formatted, and exported safely.
    # The with capture_and_log(...) block is the magic here. Even if your CrewAI code crashes 
    # in the middle of execution, Python's Context Manager guarantees that sys.stdout will be 
    # restored back to normal and the HTML file will be generated properly.
    with capture_and_log(TRANSCRIPT_FILE, HTML_FILE):
        user_request = "Run a quick ANSI color and HTML pipe test."
        logger.info(f"🚀 Kickstarting Crew process with input: '{user_request}'")
        
        try:
            result = crew.kickoff(inputs={"user_request": user_request})
            logger.info("✅ Crew kickoff completed successfully.")
            print(f"\n{'='*50}\nFINAL RESULT\n{'='*50}\n{result}")
            
        except Exception as e:
            logger.error(f"❌ Fatal error during Crew execution: {e}", exc_info=True)