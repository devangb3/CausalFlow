"""
Demo Script: Demonstrates the complete CausalFlow pipeline.

This script:
1. Creates a simple agent that solves a math problem (and fails)
2. Analyzes the failure using CausalFlow
3. Generates reports showing causal attribution, repairs, and critiques
"""

import os
from dotenv import load_dotenv
from example_agent import MathReasoningAgent
from llm_client import LLMClient
from causal_flow import CausalFlow
from trace_logger import TraceLogger


def create_sample_failed_trace():
    """
    More complex example: Multi-step inventory problem with percentages.

    Problem: A store has 120 items in stock. During a sale, they sell 35% of their
    inventory in the morning. In the afternoon, they receive a shipment of 40 new items.
    Then they sell 25 more items. How many items does the store have at the end of the day?

    Correct answer: 93
    - Morning sales: 35% of 120 = 42 items
    - After morning: 120 - 42 = 78
    - After shipment: 78 + 40 = 118
    - After afternoon sales: 118 - 25 = 93

    The agent will make an error in calculating the percentage.
    """
    trace = TraceLogger()

    # Step 0: Initial problem understanding (correct)
    step_0 = trace.log_reasoning(
        "Store starts with 120 items. Need to track: (1) morning sales of 35%, "
        "(2) shipment of 40 items, (3) afternoon sales of 25 items.",
        dependencies=[]
    )

    # Step 1: Calculate morning sales percentage (correct reasoning)
    step_1 = trace.log_reasoning(
        "First, calculate 35% of 120 to find morning sales.\nOPERATION: multiplication\nEXPRESSION: 120 * 0.35",
        dependencies=[step_0]
    )

    # Step 2: Tool call for percentage (correct)
    step_2 = trace.log_tool_call(
        tool_name="calculator",
        tool_args={"expression": "120 * 0.35"},
        dependencies=[step_1]
    )

    # Step 3: Tool response (correct)
    step_3 = trace.log_tool_response(
        tool_output=42.0,
        dependencies=[step_2]
    )

    # Step 4: Calculate remaining after morning (ERROR: agent uses wrong value)
    step_4 = trace.log_reasoning(
        "Morning sales were 42 items. However, I'll round this to 40 for simplicity.\n"
        "Remaining inventory: 120 - 40 = 80",
        dependencies=[step_3]
    )

    # Step 5: Tool call for subtraction (uses incorrect value due to step 4 error)
    step_5 = trace.log_tool_call(
        tool_name="calculator",
        tool_args={"expression": "120 - 40"},
        dependencies=[step_4]
    )

    # Step 6: Tool response (technically correct calculation, but based on wrong input)
    step_6 = trace.log_tool_response(
        tool_output=80,
        dependencies=[step_5]
    )

    # Step 7: Add shipment (correct operation, but starting from wrong base)
    step_7 = trace.log_reasoning(
        "Now add the shipment of 40 items.\nOPERATION: addition\nEXPRESSION: 80 + 40",
        dependencies=[step_6]
    )

    # Step 8: Tool call for addition
    step_8 = trace.log_tool_call(
        tool_name="calculator",
        tool_args={"expression": "80 + 40"},
        dependencies=[step_7]
    )

    # Step 9: Tool response
    step_9 = trace.log_tool_response(
        tool_output=120,
        dependencies=[step_8]
    )

    # Step 10: Subtract afternoon sales (correct operation, but chain is already wrong)
    step_10 = trace.log_reasoning(
        "Finally, subtract the 25 items sold in the afternoon.\nOPERATION: subtraction\nEXPRESSION: 120 - 25",
        dependencies=[step_9]
    )

    # Step 11: Tool call for final subtraction
    step_11 = trace.log_tool_call(
        tool_name="calculator",
        tool_args={"expression": "120 - 25"},
        dependencies=[step_10]
    )

    # Step 12: Tool response
    step_12 = trace.log_tool_response(
        tool_output=95,
        dependencies=[step_11]
    )

    # Step 13: Final reasoning
    step_13 = trace.log_reasoning(
        "The store has 95 items remaining at the end of the day.",
        dependencies=[step_12]
    )

    # Step 14: Final answer (WRONG - should be 93, but agent says 95 due to rounding error in step 4)
    step_14 = trace.log_final_answer(
        "95",
        dependencies=[step_13]
    )

    # Record outcome
    trace.record_outcome(final_answer="95", gold_answer="93")

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
    trace.to_json("sample_trace.json")
    print("\nTrace saved to sample_trace.json")

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

        report = flow.generate_full_report("causalflow_report.txt")
        print(report)

        # Export results
        flow.export_results("causalflow_results.json")

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
        agent.save_trace("agent_trace.json")

        # Analyze if it failed
        if not trace.success:
            print("\nAgent failed! Running CausalFlow analysis...")

            flow = CausalFlow(api_key=api_key)
            results = flow.analyze_trace(trace)

            flow.generate_full_report("agent_analysis.txt")
            print("\nAnalysis saved to agent_analysis.txt")
        else:
            print("\nAgent succeeded!")

    except Exception as e:
        print(f"\nError: {e}")


def main():
    #trace = demo_basic_trace()

    demo_causal_flow_analysis()


if __name__ == "__main__":
    main()
