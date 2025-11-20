#!/usr/bin/env python3
"""
Test script to verify structured outputs implementation compiles correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "gsm8k"))

# Test imports
try:
    from schemas import GSM8KSolution, GSM8KCalculationStep, LLMSchemas
    print("✓ Schema imports successful")
except Exception as e:
    print(f"✗ Schema import failed: {e}")
    sys.exit(1)

# Test schema registration
try:
    model = LLMSchemas.get_model("gsm8k_solution")
    assert model == GSM8KSolution
    print("✓ GSM8K schema registered correctly")
except Exception as e:
    print(f"✗ Schema registration failed: {e}")
    sys.exit(1)

# Test response format generation
try:
    response_format = LLMSchemas.get_response_format("gsm8k_solution")
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "gsm8k_solution"
    assert response_format["json_schema"]["strict"] == True
    print("✓ Response format generation successful")
except Exception as e:
    print(f"✗ Response format generation failed: {e}")
    sys.exit(1)

# Test schema structure
try:
    schema = response_format["json_schema"]["schema"]
    assert "properties" in schema
    assert "reasoning" in schema["properties"]
    assert "steps" in schema["properties"]
    assert "final_answer" in schema["properties"]
    print("✓ Schema structure is correct")
except Exception as e:
    print(f"✗ Schema structure validation failed: {e}")
    sys.exit(1)

# Test GSM8KAgent imports
try:
    from gsm8k_agent import GSM8KAgent
    print("✓ GSM8KAgent import successful")
except Exception as e:
    print(f"✗ GSM8KAgent import failed: {e}")
    sys.exit(1)

# Test that GSM8KAgent can be instantiated in test mode
try:
    # Don't actually create an agent since it would require an API key
    # Just verify the class exists and has the expected methods
    assert hasattr(GSM8KAgent, 'solve')
    assert hasattr(GSM8KAgent, '_solve_structured')
    assert hasattr(GSM8KAgent, '_solve_legacy')
    print("✓ GSM8KAgent has expected methods")
except Exception as e:
    print(f"✗ GSM8KAgent validation failed: {e}")
    sys.exit(1)

# Test sample data validation
try:
    sample_data = {
        "reasoning": "First calculate eggs remaining, then multiply by price",
        "steps": [
            {
                "description": "Calculate eggs remaining after breakfast and baking",
                "operation": "subtraction",
                "expression": "16 - 3 - 4"
            },
            {
                "description": "Calculate revenue from selling eggs",
                "operation": "multiplication",
                "expression": "9 * 2"
            }
        ],
        "final_answer": "18"
    }

    solution = GSM8KSolution(**sample_data)
    assert solution.reasoning == sample_data["reasoning"]
    assert len(solution.steps) == 2
    assert solution.final_answer == "18"
    print("✓ Sample data validation successful")
except Exception as e:
    print(f"✗ Sample data validation failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
print("\nStructured outputs implementation is ready.")
print("To run the full experiment, ensure OPENROUTER_SECRET_KEY is set in .env")
