import os
from dotenv import load_dotenv
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_logger import TraceLogger
from causal_flow import CausalFlow


def create_complex_failed_trace():
    """
    Complex multi-step inventory problem with percentages and error propagation.

    Problem: A store has 120 items in stock. During a sale, they sell 35% of their
    inventory in the morning. In the afternoon, they receive a shipment of 40 new items.
    Then they sell 25 more items. How many items does the store have at the end of the day?

    Correct solution:
    - Morning sales: 35% of 120 = 42 items
    - After morning: 120 - 42 = 78
    - After shipment: 78 + 40 = 118
    - After afternoon sales: 118 - 25 = 93

    Agent error: At step 4, the agent incorrectly rounds 42 to 40 "for simplicity",
    which causes all subsequent calculations to be off by 2, resulting in a final
    answer of 95 instead of 93.
    """
    problem_statement = (
        """A store has 120 items in stock. During a sale, they sell 35% of their 
        inventory in the morning. In the afternoon, they receive a shipment of 40 new items. 
        Then they sell 25 more items. How many items does the store have at the end of the day? """
    )
    trace = TraceLogger(problem_statement=problem_statement)

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
    # This is the critical failure point - agent rounds unnecessarily
    step_4 = trace.log_reasoning(
        "Morning sales were 42 items. However, I'll round this to 40 for simplicity.\n",
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


def demo_complex_analysis():
    print("EXAMPLE: Multi-step Inventory Problem")

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\nERROR: OPENROUTER_SECRET_KEY not found in .env file")
        return

    # Create the trace
    trace = create_complex_failed_trace()

    # Save trace
    trace.to_json("examples/complex_trace.json")
    try:
        # Initialize CausalFlow
        flow = CausalFlow(api_key=api_key)

        # Run analysis
        print("\n  Analyzing causal graph")
        results = flow.analyze_trace(trace, skip_repair=False)

        # Generate report
        report_path = "examples/complex_analysis_report.txt"
        report = flow.generate_full_report(report_path)
        print(f"\n  Report saved to: {report_path}")

        # Export results
        results_path = "examples/complex_analysis_results.json"
        flow.export_results(results_path)
        print(f"  Results saved to: {results_path}")

        print("ANALYSIS COMPLETE")
       
        if 'attribution' in results and results['attribution']:
            print("\nMost Causally Responsible Steps:")
            for step_id, score in sorted(results['attribution'].items(),
                                        key=lambda x: x[1],
                                        reverse=True)[:3]:
                step = trace.get_step(step_id)
                print(f"  Step {step_id} [{step.step_type.value}]: {score:.3f}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    demo_complex_analysis()


if __name__ == "__main__":
    main()
