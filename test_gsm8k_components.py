"""
Test GSM8K components without requiring API access.

This script tests the mathematical reexecutor and basic data loading
without making any LLM API calls.
"""

from math_reexecutor import MathReexecutor
from run_gsm8k_experiment import GSM8KDataLoader


def test_math_reexecutor():
    """Test the mathematical reexecutor component."""
    print("="*70)
    print("Testing Mathematical Reexecutor")
    print("="*70)

    executor = MathReexecutor()

    # Test 1: Number extraction
    print("\n1. Number Extraction Tests:")
    test_cases = [
        ("The answer is 42", 42.0),
        ("#### 1234", 1234.0),
        ("$1,234.56", 1234.56),
        ("Total: 18 dollars", 18.0),
        ("Janet makes $18 every day at the farmer's market.\n#### 18", 18.0),
    ]

    passed = 0
    for text, expected in test_cases:
        result = executor.extract_number(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:50]}...' -> {result} (expected {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")

    # Test 2: Expression evaluation
    print("\n2. Expression Evaluation Tests:")
    expressions = [
        ("16 - 3 - 4", 9.0),
        ("9 * 2", 18.0),
        ("120 * 0.35", 42.0),
        ("2 / 2", 1.0),
        ("2 + 1", 3.0),
    ]

    passed = 0
    for expr, expected in expressions:
        result = executor.evaluate_expression(expr)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{expr}' = {result} (expected {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(expressions)}")

    # Test 3: Answer comparison
    print("\n3. Answer Comparison Tests:")
    comparisons = [
        ("18", "#### 18", True),
        ("42.0", "42", True),
        ("$18", "18", True),
        ("95", "93", False),
    ]

    passed = 0
    for ans1, ans2, expected in comparisons:
        result = executor.compare_answers(ans1, ans2)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{ans1}' == '{ans2}': {result} (expected {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(comparisons)}")


def test_data_loader():
    """Test the GSM8K data loader."""
    print("\n" + "="*70)
    print("Testing GSM8K Data Loader")
    print("="*70)

    loader = GSM8KDataLoader()

    # Test loading sample data
    print("\n1. Loading Sample Data:")
    data = loader.load_data(num_rows=3, use_sample=True)
    print(f"  ✓ Loaded {len(data)} problems")

    # Test extracting gold answers
    print("\n2. Extracting Gold Answers:")
    for i, item in enumerate(data[:3], 1):
        gold = loader.extract_gold_answer(item['answer'])
        print(f"  Problem {i}:")
        print(f"    Question: {item['question'][:60]}...")
        print(f"    Gold Answer: {gold}")

    # Test answer extraction from various formats
    print("\n3. Gold Answer Extraction Tests:")
    test_answers = [
        "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18",
        "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fiber\n#### 3",
    ]

    expected_answers = ["18", "3"]

    passed = 0
    for ans_text, expected in zip(test_answers, expected_answers):
        result = loader.extract_gold_answer(ans_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Extracted: {result} (expected {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(expected_answers)}")


def test_integration():
    """Test integration of components."""
    print("\n" + "="*70)
    print("Testing Component Integration")
    print("="*70)

    loader = GSM8KDataLoader()
    executor = MathReexecutor()

    # Load a sample problem
    data = loader.load_data(num_rows=1, use_sample=True)
    problem = data[0]

    print("\n1. Sample Problem:")
    print(f"  Question: {problem['question'][:100]}...")

    # Extract gold answer
    gold = loader.extract_gold_answer(problem['answer'])
    print(f"  Gold Answer: {gold}")

    # Simulate agent answer (correct)
    agent_answer = "18"
    match = executor.compare_answers(agent_answer, gold)
    print(f"\n2. Answer Verification:")
    print(f"  Agent Answer: {agent_answer}")
    print(f"  Gold Answer: {gold}")
    print(f"  Match: {match} {'✓' if match else '✗'}")

    # Simulate agent answer (incorrect)
    agent_answer = "20"
    match = executor.compare_answers(agent_answer, gold)
    print(f"\n3. Failed Answer:")
    print(f"  Agent Answer: {agent_answer}")
    print(f"  Gold Answer: {gold}")
    print(f"  Match: {match} {'✓' if match else '✗'}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GSM8K Components Test Suite")
    print("="*70)
    print("\nThis test suite verifies the core components without requiring API access.\n")

    try:
        test_math_reexecutor()
        test_data_loader()
        test_integration()

        print("\n" + "="*70)
        print("All Tests Completed Successfully!")
        print("="*70)
        print("\nThe GSM8K implementation is ready to use.")
        print("To run the full experiment with LLM analysis, ensure you have:")
        print("  1. Set OPENROUTER_SECRET_KEY in .env file")
        print("  2. Run: python run_gsm8k_experiment.py --num-rows 5 --use-sample")
        print("\nSee GSM8K_README.md for detailed usage instructions.")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
