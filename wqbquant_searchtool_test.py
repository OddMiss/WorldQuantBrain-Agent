import os
import sys
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crewai import Agent, Task, Crew, Process
from utils.htmlcolorlog import capture_and_log

def test_agents(
    retrieve_text_data, search_operators, 
    search_datafields, llm, transcript_file, 
    html_file, logger
):
    """Run a strict health check for the three search tools."""
    diagnostic_tester = Agent(
        role="API Formatting Assistant",
        goal="Format queries perfectly to test the text-based search interfaces.",
        backstory="""You are an AI assistant interacting with external text-search APIs provided in your environment.
    You do not need direct access to local databases; you simply output the strict `[ToolCalls]` formatting requested to trigger the external search.""",
        tools=[retrieve_text_data, search_operators, search_datafields],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    diagnostic_task = Task(
        description="""
    Perform a strict health check on the system's tools.

    STEP 1: Execute a test for each available search function:
    - search 'AllRightsReserved' using `retrieve_text_data`
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

    with capture_and_log(transcript_file, html_file):
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