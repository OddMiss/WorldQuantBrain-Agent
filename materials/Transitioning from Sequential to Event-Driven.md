## Concept: Transitioning from Sequential to Event-Driven

In your current architecture, you are using `Process.sequential`. This forces CrewAI to treat your agents like a single-direction conveyor belt:

$$\text{Researcher} \longrightarrow \text{Ideator} \longrightarrow \text{Coder} \longrightarrow \text{Validator}$$

When the Validator discovers an error, it is trapped at the end of the line. It cannot pass the alpha *backward* to the Coder because sequential processes have no memory or mechanism for backtracking.

**CrewAI Flows** shifts your pipeline from a rigid conveyor belt into an event-driven state machine. Instead of the agents managing the retry loop through prompt guessing, **Python code controls the macro-workflow**, while the agents focus strictly on their microscopic tasks.

---

## Core Building Blocks of a CrewAI Flow

Flows use Python class structures, state objects, and property decorators to manage execution:

* **The State (`BaseModel`):** A globally shared memory bucket accessible by all steps in the flow. It tracks variables like the current formula, error messages, and retry counters.
* **`@start()`:** Marks the entry point of your pipeline.
* **`@listen(target_method)`:** A trigger that fires automatically as soon as `target_method` finishes executing.
* **`@router(target_method)`:** A conditional gatekeeper. It evaluates the output of `target_method` and determines which Python function to trigger next based on your custom logic (e.g., successful validation vs. syntax failure).

---

## Step-by-Step Implementation

To implement Solution 3, keep your existing Agent and Tool declarations exactly as they are. You will separate your original `Crew` into smaller, specialized modules controlled by a `Flow` class.

### 1. Define the Flow State

Create a Pydantic model to track the health of your Alpha generation process across iterations.

```python
from crewai.flow.flow import Flow, start, listen, router
from pydantic import BaseModel

class AlphaGenerationState(BaseModel):
    user_request: str = ""
    research_report: str = ""
    alpha_ideas: str = ""
    coder_output: str = ""
    validation_report: str = ""
    error_log: str = ""
    retry_count: int = 0
    is_approved: bool = False

```

### 2. Update Tasks for the Loop

We need to modify the Coder's task slightly so that if a failure occurs, the Coder can ingest the `error_log` produced by the Validator.

```python
# Create a specialized task for fixing broken expressions
recoding_task = Task(
    description="""
    The previous BRAIN expression failed verification with the following error:
    {error_log}
    
    Your job is to fix it.
    1. Use `search_operators` to verify the syntax of the failing operator.
    2. Use `search_datafields` to verify the data fields.
    3. Output a corrected, complete expression based ONLY on these lookups.
    """,
    expected_output="One corrected complete alpha in the exact user format.",
    agent=coder
)

```

### 3. Build the Flow Class

Now, combine your components into a managed workflow graph.

```python
class WorldQuantAlphaFlow(Flow[AlphaGenerationState]):

    @start()
    def initialize_and_research(self):
        """Step 1: Run Research and Ideation sequentially to generate candidates."""
        print("🚀 Starting Research and Ideation Phase...")
        initial_crew = Crew(
            agents=[researcher, ideator],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=True
        )
        # Kickoff and capture outputs into state
        result = initial_crew.kickoff(inputs={"user_request": self.state.user_request})
        
        # Split outputs conceptually or let task interpolation handle it
        self.state.alpha_ideas = result.raw
        return result

    @listen(initialize_and_research)
    def code_initial_alpha(self):
        """Step 2: Coder builds the first iteration expression."""
        print("💻 Coding Initial Alpha Expression...")
        coding_crew = Crew(agents=[coder], tasks=[task3], verbose=True)
        
        # Pass the ideas from state directly into the coder's task context
        result = coding_crew.kickoff(inputs={"ideas_output": self.state.alpha_ideas})
        self.state.coder_output = result.raw
        return result

    @listen("retry_coding_path")
    def recode_broken_alpha(self):
        """Step 2b: Alternate entry point if the Validator caught an error."""
        print(f"🔧 Retrying Code Formulation (Attempt {self.state.retry_count}/4)...")
        recoding_crew = Crew(agents=[coder], tasks=[recoding_task], verbose=True)
        
        result = recoding_crew.kickoff(inputs={"error_log": self.state.error_log})
        self.state.coder_output = result.raw
        return result

    @listen(code_initial_alpha, recode_broken_alpha)
    def validate_expression(self):
        """Step 3: Validator tests expression via wqb_simulate_api."""
        print("🔬 Validating Expression against WQB Simulation API...")
        validation_crew = Crew(agents=[validator], tasks=[task4], verbose=True)
        
        result = validation_crew.kickoff(inputs={"coder_output": self.state.coder_output})
        self.state.validation_report = result.raw
        return result

    @router(validate_expression)
    def evaluate_validation_results(self):
        """Step 4: Smart Router determines pipeline trajectory."""
        report = self.state.validation_report.upper()
        
        # Check for error indicators in the validator's output text
        if "❌" in report or "ERROR" in report or "FAIL" in report:
            if self.state.retry_count < 4:
                self.state.retry_count += 1
                self.state.error_log = self.state.validation_report
                print("⚠️ Validation Failed. Routing back to Coder.")
                return "retry_coding_path" # Routes execution back to Step 2b
            else:
                print("🚨 Max retries reached. Terminating flow with failure state.")
                self.state.is_approved = False
                return "validation_failed_permanently"
        
        print("✅ Alpha successfully cleared all validation checks!")
        self.state.is_approved = True
        return "validation_passed"

    @listen("validation_passed")
    def finalize_flow(self):
        print("🎉 Flow completed successfully!")
        return self.state.validation_report

    @listen("validation_failed_permanently")
    def fail_flow(self):
        print("🛑 Flow failed to self-correct within iteration limits.")
        return f"Failed Alpha: {self.state.coder_output}\nLast Error Log: {self.state.error_log}"

```

### 4. Running the Flow

Replace your execution logic block at the bottom of your file with the following:

```python
if __name__ == "__main__":
    with capture_and_log(HTML_FILE):
        logger.info("Main", "🚀 Kickstarting Managed Flow Architecture")
        
        # Instantiate the flow state engine
        alpha_flow = WorldQuantAlphaFlow()
        alpha_flow.state.user_request = user_request
        
        # Execute the state machine graph
        final_output = alpha_flow.kickoff()
        
        logger.info("Main", f"\n{'='*50}\nFINAL FLOW OUTPUT\n{'='*50}\n{final_output}")

```

---

## Why This Fixes Your Problem

1. **Information Asymmetry Eliminated:** When `wqb_simulate_api` errors out, the precise syntax failure or missing data field message is stored cleanly inside `self.state.error_log`.
2. **Targeted Repair Work:** Instead of forcing the Validator to guess blind, control loops back to the **Coder**, who possesses the `search_operators` and `search_datafields` registry tools needed to cross-examine and fix the error.
3. **No Reset Overhead:** The Flow doesn't restart the entire program from scratch. Steps 1 (`initialize_and_research`) is skipped entirely on subsequent loops. Your expensive vector store lookups and ideation steps happen **exactly once**, preserving your context tokens and minimizing execution costs.

Would you like help customizing the Pydantic state schema further to track specific simulation metrics like Sharpe Ratio or Turnover?