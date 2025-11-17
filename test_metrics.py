"""
Test script for the new metrics JSON generation functionality.
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trace_logger import TraceLogger
from causal_flow import CausalFlow


def create_test_trace():
    """Create a simple failed trace for testing."""
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

    # Step 4: Final reasoning (ERROR: agent fabricates incorrect information)
    step_4 = trace.log_reasoning(
        "The calculation shows 8, but John gave away 2 apples to his friend, so the answer is 6.",
        dependencies=[step_3]
    )

    # Step 5: Final answer (WRONG - should be 8)
    step_5 = trace.log_final_answer(
        "6",
        dependencies=[step_4]
    )

    trace.record_outcome(final_answer="6", gold_answer="8")

    return trace


def test_metrics_without_ground_truth():
    """Test metrics generation without ground truth."""
    print("\n" + "=" * 70)
    print("TEST 1: Metrics Generation WITHOUT Ground Truth")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\nOPENROUTER_SECRET_KEY not found. Cannot run test.")
        return

    trace = create_test_trace()
    flow = CausalFlow(api_key=api_key)

    print("\nRunning analysis...")
    results = flow.analyze_trace(
        trace,
        skip_repair=False,
        metrics_output_file="test_metrics_no_gt.json",
        ground_truth_causal_steps=None
    )

    # Read and display the metrics
    with open("test_metrics_no_gt.json", 'r') as f:
        metrics = json.load(f)

    print("\n--- GENERATED METRICS ---")
    print(json.dumps(metrics, indent=2))

    print("\n✓ Test passed: Metrics generated without ground truth")


def test_metrics_with_ground_truth():
    """Test metrics generation with ground truth."""
    print("\n" + "=" * 70)
    print("TEST 2: Metrics Generation WITH Ground Truth")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\nOPENROUTER_SECRET_KEY not found. Cannot run test.")
        return

    trace = create_test_trace()
    flow = CausalFlow(api_key=api_key)

    # Define ground truth: Steps 4 and 5 are truly causal
    ground_truth = [4, 5]

    print("\nRunning analysis with ground truth...")
    print(f"Ground truth causal steps: {ground_truth}")

    results = flow.analyze_trace(
        trace,
        skip_repair=False,
        metrics_output_file="test_metrics_with_gt.json",
        ground_truth_causal_steps=ground_truth
    )

    # Read and display the metrics
    with open("test_metrics_with_gt.json", 'r') as f:
        metrics = json.load(f)

    print("\n--- GENERATED METRICS ---")
    print(json.dumps(metrics, indent=2))

    # Validate key metrics exist
    assert "causal_attribution_metrics" in metrics
    assert "repair_metrics" in metrics
    assert "minimality_metrics" in metrics
    assert "multi_agent_agreement" in metrics

    # Check precision/recall are calculated
    causal_metrics = metrics["causal_attribution_metrics"]
    assert causal_metrics["precision"] is not None
    assert causal_metrics["recall"] is not None
    assert causal_metrics["f1_score"] is not None

    print(f"\n✓ Precision: {causal_metrics['precision']}")
    print(f"✓ Recall: {causal_metrics['recall']}")
    print(f"✓ F1 Score: {causal_metrics['f1_score']}")

    # Check repair success rate
    repair_metrics = metrics["repair_metrics"]
    print(f"✓ Repair Success Rate: {repair_metrics['success_rate']}")

    # Check minimality scores
    minimality = metrics["minimality_metrics"]
    if minimality["average_minimality"] is not None:
        print(f"✓ Average Minimality: {minimality['average_minimality']}")

    # Check multi-agent agreement
    agreement = metrics["multi_agent_agreement"]
    if agreement["average_consensus_score"] is not None:
        print(f"✓ Average Consensus Score: {agreement['average_consensus_score']}")

    print("\n✓ Test passed: All metrics generated correctly with ground truth")


def test_manual_metrics_generation():
    """Test calling generate_metrics_json directly."""
    print("\n" + "=" * 70)
    print("TEST 3: Manual Metrics Generation")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")

    if not api_key:
        print("\nOPENROUTER_SECRET_KEY not found. Cannot run test.")
        return

    trace = create_test_trace()
    flow = CausalFlow(api_key=api_key)

    # Run analysis without auto-generating metrics
    print("\nRunning analysis without metrics export...")
    results = flow.analyze_trace(
        trace,
        skip_repair=False,
        metrics_output_file=None  # Skip automatic export
    )

    # Manually generate metrics
    print("\nManually generating metrics...")
    metrics = flow.generate_metrics_json(ground_truth_causal_steps=[4, 5])

    print("\n--- MANUALLY GENERATED METRICS ---")
    print(json.dumps(metrics, indent=2))

    print("\n✓ Test passed: Manual metrics generation works")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING CAUSAL METRICS JSON GENERATION")
    print("=" * 70)

    try:
        test_metrics_without_ground_truth()
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")

    try:
        test_metrics_with_ground_truth()
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")

    try:
        test_manual_metrics_generation()
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
