import os
import sys
from config.config import API_KEY

# ====================== FORCE RICH/CREWAi COLORS (MUST BE FIRST) ======================
# This runs BEFORE crewai imports rich internally
os.environ["FORCE_COLOR"] = "1"
os.environ["PY_COLORS"] = "1"
os.environ["NO_COLOR"] = ""          # important
os.environ["TERM"] = "xterm-256color"
os.environ["PYTHONUNBUFFERED"] = "1"

# Global reconfigure — this is the part that actually works when piped
import rich
rich.reconfigure(
    force_terminal=True,
    color_system="256",      # or "truecolor" if your terminal supports it
    width=140,
    soft_wrap=True
)

# Extra safety: force a console instance early
from rich.console import Console
Console(force_terminal=True, color_system="256", width=140)

# Import after setting environment variables to ensure they take effect for all libraries
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import logging
import datetime

console = Console(force_terminal=True, color_system="256")
console.print("[bold red]🚀 RICH FORCE TEST — IF YOU SEE RED, IT WORKS[/bold red]")
console.print("[bold green]✅ CrewAI colors should now appear below too[/bold green]")

def setup_logger(log_dir, log_name, logger_obj_name="logger_obj_name"):
    """
    Configure the logger system: output to both console and file simultaneously.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created: {log_dir}")

    log_filename = f"{log_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(logger_obj_name)
    logger.setLevel(logging.INFO) 
    logger.propagate = False # (Added this to stop duplicate printing)
    logger.handlers = [] 

    file_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout) # 
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"✅ logger System Started")
    logger.info(f"Log file path: {log_filepath}")
    return logger

# ====================== CONFIG ======================
BASE_DIR = "D:/AI_Data/Computer/Agent-WQB/"
CHROMA_DIR = BASE_DIR + "quant_forum_chroma/"
BGEM3_DIR = BASE_DIR + "quant_forum_bgem3/"
HF_CACHE_DIR = BASE_DIR + "cache/hf/"
PIP_CACHE_DIR = BASE_DIR + "cache/pip/"
LOG_DIR = BASE_DIR + "logs/" + datetime.datetime.now().strftime("%Y%m") + "/"

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(PIP_CACHE_DIR, exist_ok=True)
os.makedirs(BGEM3_DIR, exist_ok=True)

# ====================== INITIALIZE LOGGER ======================
logger = setup_logger(LOG_DIR, "wqb_agent_test", "wqb_main_logger")

# ====================== YOUR GEMINI CLIENT ======================
logger.info("Initializing LLM client...")
llm = LLM(
    model="gemini-3.1-pro",   
    base_url="http://100.127.232.48:8000/v1",
    api_key=API_KEY,
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

crew = Crew(
    agents=[tester_agent],
    tasks=[task1],
    verbose=True
)
logger.info("✅ Crew successfully assembled.")

# ====================== RUN ======================
if __name__ == "__main__":
    user_request = "Run a quick ANSI color and HTML pipe test."
    logger.info(f"🚀 Kickstarting Crew process with input: '{user_request}'")
    
    try:
        result = crew.kickoff(inputs={"user_request": user_request})
        logger.info("✅ Crew kickoff completed successfully.")
        
        print("\n\n" + "="*50)
        print("FINAL RESULT")
        print("="*50)
        print(result)
        
    except Exception as e:
        logger.error(f"❌ Fatal error during Crew execution: {e}", exc_info=True)
        print(f"An error occurred. Check logs at {LOG_DIR} for details.")