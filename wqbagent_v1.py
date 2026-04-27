import os
# from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai.tools import tool
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
# import gc
from config.config import FORUM_DATA_DIR, API_KEY
# load_dotenv()

# ====================== CONFIG: EVERYTHING ON YOUR OTHER DRIVE ======================
BASE_DIR = "D:/AI_Data/Computer/Agent-WQB/"
WQB_CONSULTANT_FORUM_PDF_PATH = "D:/AI_Data/WQB-Consultant-Data/Forums/wqb_consultant_pdf/"
CHROMA_DIR = BASE_DIR + "quant_forum_chroma/"
BGEM3_DIR = BASE_DIR + "quant_forum_bgem3/"
HF_CACHE_DIR = BASE_DIR + "cache/hf/"
PIP_CACHE_DIR = BASE_DIR + "cache/pip/"

# Redirect caches so nothing touches C:
# os.environ["HF_HOME"] = HF_CACHE_DIR
# os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
# os.environ["PIP_CACHE_DIR"] = PIP_CACHE_DIR

# Create the cache folder if it doesn't exist yet
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(PIP_CACHE_DIR, exist_ok=True)
os.makedirs(BGEM3_DIR, exist_ok=True)

# ====================== YOUR GEMINI CLIENT ======================
"""
CrewAI’s Agent class no longer accepts a raw langchain_openai.ChatOpenAI object directly in the llm= parameter (Pydantic validation fails).
Instead, you must use CrewAI’s own LLM wrapper (it internally uses LiteLLM and works perfectly with your reverse-engineered Gemini proxy).
"""

# Use your exact proxy settings
llm = LLM(
    model="gemini-3.1-pro",   # ← change if your proxy uses a different model name
    base_url="http://100.127.232.48:8000/v1",
    api_key=API_KEY,
    temperature=0.6,          # slightly lower = more stable
    max_tokens=8192,
    timeout=180,              # give it more time
    max_retries=5,            # extra retries
)

# llm.invoke([{"role": "user", "content": "Hello, how are you?"}]) # Test the client (for openai-compatible LLMs)
# llm.call("Hello, how are you?") # Simpler test method (for crewai's LLM wrapper)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", # excellent for Chinese
    cache_folder=HF_CACHE_DIR,  # Use the custom cache directory
    model_kwargs={"device": "cpu"},           # force CPU (your low GPU setup)
    encode_kwargs={"normalize_embeddings": True},  # best for Chroma similarity search
    show_progress=True
)

# ==================================== Ingest Forum (PDFs) (ONE-TIME) ====================================
def ingest_forum():
    # os.makedirs(FORUM_DATA_DIR, exist_ok=True)
    
    loader = DirectoryLoader(
        WQB_CONSULTANT_FORUM_PDF_PATH,
        glob="**/*.pdf",                    # ← changed for your PDFs
        loader_cls=PyPDFLoader,
        silent_errors=False,
        show_progress=True
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,               # defined in above cell
        persist_directory=BGEM3_DIR
    )
    print(f"✅ Ingested {len(splits)} chunks from {len(docs)} PDF files")
    return vectorstore

# First run only:
# vectorstore = ingest_forum()
vectorstore = Chroma(persist_directory=BGEM3_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ====================== FORUM SEARCH TOOL ======================
@tool("Forum_Search")
def forum_search(query: str) -> str:
    """Search the entire quant forum archive."""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# ====================== AGENTS (Your Quant Research Team) ======================
researcher = Agent(
    role="WorldQuant Forum Researcher & Master Analyst",
    goal="You MUST use the forum_search tool to extract real content from the WorldQuant Brain Consultant Forum PDFs. Never answer from general knowledge.",
    backstory="""You are a veteran WorldQuant Brain consultant who has read every single PDF in the forum archive.
    Your only source of truth is the `forum_search` tool. You always call it first.
    You never say you don't have access — you always use the tool.""",
    tools=[forum_search],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

ideator = Agent(
    role="Low-Correlation BRAIN Innovator",
    goal="Create truly innovative, submittable alphas that have low correlation to existing forum templates and common WorldQuant factors.",
    backstory="You are a contrarian quant who recombines master ideas with fresh edges while keeping everything simple enough for a clean BRAIN expression.",
    llm=llm,
    verbose=True
)

coder = Agent(
    role="WorldQuant BRAIN Expression Expert",
    goal="Convert the best idea into a clean, valid, and efficient WorldQuant BRAIN expression.",
    backstory="You are an ex-WorldQuant Brain coder who only writes correct, high-quality BRAIN expressions. You always include proper neutralization, delay, decay, and truncation settings.",
    llm=llm,
    verbose=True
)

validator = Agent(
    role="WorldQuant Submission Validator",
    goal="Ensure the alpha is innovative, low-correlation, and ready to submit. Output ONLY in the exact user-specified format.",
    backstory="You are the final gatekeeper. You check for simulator compatibility, low correlation, and economic soundness. You never add extra text outside the required format.",
    llm=llm,
    verbose=True
)

# ====================== TASKS & CREW ======================
task1 = Task(
    description="""
    You MUST use the forum_search tool at least once (ideally multiple times) to find relevant threads.
    Search for keywords from the user request.
    Extract real master opinions, alpha templates, crowded signals, and specific discussions from the actual forum PDFs.
    Do NOT use general knowledge. Cite the content you retrieve from the tool.
    """,
    expected_output="Structured summary of REAL content from the forum PDFs with direct quotes and PDF/thread references.",
    agent=researcher
)

task2 = Task(
    description="""
    Generate 3–5 genuinely innovative alpha ideas that are different from existing forum templates and your current portfolio.
    Focus on low correlation and economic rationale that can be expressed cleanly in BRAIN.
    """,
    expected_output="Numbered list of 3–5 alpha ideas with short name, hypothesis, and low-correlation justification.",
    agent=ideator
)

task3 = Task(
    description="""
    Take the BEST idea from Task 2 and write a clean, valid WorldQuant BRAIN expression.
    Choose realistic Target Settings (Universe, Neutralization, Delay, Decay, Truncation).
    """,
    expected_output="One complete alpha in the exact user format (Alpha Name + Economic Hypothesis + Target Settings + Full BRAIN Expression).",
    agent=coder
)

task4 = Task(
    description="""
    Act as strict WorldQuant reviewer.
    Critique the alpha from Task 3 for innovation, low correlation, simulator compatibility, and economic sense.
    If needed, improve it slightly.
    THEN output ONLY the final alpha in the EXACT format the user wants:
    
    **Alpha Name:** ...
    **Economic Hypothesis:** ...
    **Target Settings:** Universe: ___ | Neutralization: ___ | Delay: ___ | Decay: ___ | Truncation: ___
    **Full BRAIN Expression:** ...
    
    Do not add any extra explanation or text outside this format.
    """,
    expected_output="Final alpha in the exact markdown format requested by the user.",
    agent=validator
)

crew = Crew(
    agents=[researcher, ideator, coder, validator],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True
)

# ====================== RUN ======================
if __name__ == "__main__":
    result = crew.kickoff(inputs={"user_request": "Create innovative low-correlation alphas using ideas from the WorldQuant Brain Consultant Forum"})
    print(result)