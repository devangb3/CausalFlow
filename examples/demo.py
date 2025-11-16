"""
Demo Script: Demonstrates the complete CausalFlow pipeline.

This script:
1. Creates a simple agent that solves a math problem (and fails)
2. Analyzes the failure using CausalFlow
3. Generates reports showing causal attribution, repairs, and critiques
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_agent import MathReasoningAgent
from llm_client import LLMClient
from causal_flow import CausalFlow
from trace_logger import TraceLogger


def create_sample_failed_trace():

    problem_statement = "John has 5 apples and buys 3 more. How many apples does he have?"
    trace = TraceLogger(problem_statement=problem_statement)

    # Step 0: Initial reasoning (correct)
    step_0 = trace.log_reasoning(
        "John starts with 5 apples. He buys 3 more. We need to add these together.",
        dependencies=[]
    )

    # Step 1: Planning calculation (correct)
    step_1 = trace.log_reasoning(
        "OPERATION: addition\nEXPRESSION: 5 + 3",
        dependencies=[step_0]
    )

    # Step 2: Tool call (correct)
    step_2 = trace.log_tool_call(
        tool_name="calculator",
        tool_args={"expression": "5 + 3"},
        dependencies=[step_1]
    )

    # Step 3: Tool response (correct)
    step_3 = trace.log_tool_response(
        tool_output=8,
        dependencies=[step_2]
    )

    # Step 4: Final reasoning (ERROR: agent misinterprets the result)
    step_4 = trace.log_reasoning(
        "The calculation shows 8, but John gave away 2 apples to his friend, so the answer is 6.",
        dependencies=[step_3]
    )

    # Step 5: Final answer (WRONG - should be 8, but agent says 6)
    step_5 = trace.log_final_answer(
        "6",
        dependencies=[step_4]
    )

    # Record outcome
    trace.record_outcome(final_answer="6", gold_answer="8")

    return trace


def demo_basic_trace():

    trace = create_sample_failed_trace()

    print("\nTrace Summary:")
    print(f"  Total Steps: {len(trace.steps)}")
    print(f"  Success: {trace.success}")
    print(f"  Final Answer: {trace.final_answer}")
    print(f"  Gold Answer: {trace.gold_answer}")

    print("\nStep-by-step breakdown:")
    for step in trace.steps:
        print(f"  Step {step.step_id} [{step.step_type.value}]: deps={step.dependencies}")

    # Save trace
    trace.to_json("examples/sample_trace.json")
    print("\nTrace saved to examples/sample_trace.json")

    return trace


def demo_causal_flow_analysis():
    """Demonstrate full CausalFlow analysis."""
    print("\n" + "=" * 70)
    print("DEMO: CausalFlow Analysis")
    print("=" * 70)

    # Check if API key is set
    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\nOPENROUTER_SECRET_KEY not found in .env file")
        return

    trace = create_sample_failed_trace()

    print("\nInitializing CausalFlow...")
    flow = CausalFlow(api_key=api_key)

    print("\nRunning analysis...")

    try:
        results = flow.analyze_trace(
            trace,
            skip_repair=False,
        )

        report = flow.generate_full_report("examples/simple_causalflow_report.txt")
        print(report)

        # Export results
        flow.export_results("examples/simple_causalflow_results.json")

        print("\nAnalysis complete!")
    except Exception as e:
        print(f"\nError during analysis: {e}")


def demo_with_real_agent():
    """Demonstrate using CausalFlow with a real agent."""
    print("\n" + "=" * 70)
    print("DEMO 3: Real Agent with CausalFlow")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\nOPENROUTER_SECRET_KEY not found. Skipping this demo.")
        return

    # Create agent
    llm_client = LLMClient(api_key=api_key)
    agent = MathReasoningAgent(llm_client)

    # Solve a problem
    problem = "Sarah has 12 cookies and eats 4 of them. How many cookies does she have left?"
    gold_answer = "8"

    try:
        answer = agent.solve(problem, gold_answer)

        # Get the trace
        trace = agent.get_trace()

        # Save trace
        agent.save_trace("examples/agent_trace.json")

        # Analyze if it failed
        if not trace.success:
            print("\nAgent failed! Running CausalFlow analysis...")

            flow = CausalFlow(api_key=api_key)
            results = flow.analyze_trace(trace)

            flow.generate_full_report("examples/agent_analysis.txt")
            print("\nAnalysis saved to examples/agent_analysis.txt")
        else:
            print("\nAgent succeeded!")

    except Exception as e:
        print(f"\nError: {e}")


def main():
    #trace = demo_basic_trace()

    demo_causal_flow_analysis()


if __name__ == "__main__":
    main()
