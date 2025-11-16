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
from causal_flow import CausalFlow, analyze_failed_trace


def create_sample_failed_trace():
    """
    Create a sample failed execution trace for demonstration.

    This simulates an agent that makes an error in reasoning.
    """
    from trace_logger import TraceLogger

    trace = TraceLogger()

    # Problem: "If John has 5 apples and buys 3 more, how many does he have?"
    # Correct answer: 8

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
    """Demonstrate basic trace logging and inspection."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Trace Logging")
    print("=" * 70)

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
    print("\n✓ Trace saved to sample_trace.json")

    return trace


def demo_causal_flow_analysis():
    """Demonstrate full CausalFlow analysis."""
    print("\n" + "=" * 70)
    print("DEMO 2: CausalFlow Analysis")
    print("=" * 70)

    # Check if API key is set
    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\n⚠ OPENROUTER_SECRET_KEY not found in .env file")
        print("  To run this demo with actual LLM analysis:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenRouter API key")
        print("  3. Run this demo again")
        print("\n  For now, running with mock trace only...")

        # Still show the trace structure
        trace = create_sample_failed_trace()
        print("\nFailed trace created. Without API key, skipping LLM-based analysis.")
        return

    # Create trace
    trace = create_sample_failed_trace()

    # Run CausalFlow analysis
    print("\nInitializing CausalFlow...")
    flow = CausalFlow(api_key=api_key)

    print("\nRunning analysis...")
    print("(This may take a minute as multiple LLM calls are made)")

    try:
        # For demo, skip critique to reduce API calls
        results = flow.analyze_trace(
            trace,
            skip_repair=False,
            skip_critique=True  # Set to False for full analysis
        )

        # Generate report
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)

        report = flow.generate_full_report("causalflow_report.txt")
        print(report)

        # Export results
        flow.export_results("causalflow_results.json")

        print("\n✓ Analysis complete!")
        print("  - Report saved to: causalflow_report.txt")
        print("  - Results saved to: causalflow_results.json")

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        print("  This might be due to API key issues or network problems.")


def demo_with_real_agent():
    """Demonstrate using CausalFlow with a real agent."""
    print("\n" + "=" * 70)
    print("DEMO 3: Real Agent with CausalFlow")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\n⚠ OPENROUTER_SECRET_KEY not found. Skipping this demo.")
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
            print("\n🔍 Agent failed! Running CausalFlow analysis...")

            flow = CausalFlow(api_key=api_key)
            results = flow.analyze_trace(trace, skip_critique=True)

            flow.generate_full_report("agent_analysis.txt")
            print("\n✓ Analysis saved to agent_analysis.txt")
        else:
            print("\n✓ Agent succeeded!")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("CausalFlow Demo Suite")
    print("=" * 70)

    # Demo 1: Basic trace logging
    trace = demo_basic_trace()

    # Demo 2: CausalFlow analysis
    demo_causal_flow_analysis()

    # Demo 3: Real agent (optional)
    # Uncomment to run with a real agent
    # demo_with_real_agent()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check the generated files (sample_trace.json, etc.)")
    print("  2. Set up your .env file with OPENROUTER_SECRET_KEY")
    print("  3. Run demo_causal_flow_analysis() to see full analysis")
    print("  4. Explore the codebase and build your own agents!")


if __name__ == "__main__":
    main()
