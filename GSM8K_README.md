# GSM8K CausalFlow Implementation

This implementation demonstrates the CausalFlow architecture on the GSM8K (Grade School Math 8K) dataset, showing how causal attribution can help diagnose and repair failures in mathematical reasoning.

## Overview

The implementation consists of three main components:

1. **Mathematical Reexecutor** (`math_reexecutor.py`): A robust, non-LLM based calculator that:
   - Extracts numerical values from text (handling units, commas, etc.)
   - Safely evaluates mathematical expressions
   - Compares answers with floating-point tolerance
   - Handles various answer formats including GSM8K's `#### NUMBER` format

2. **GSM8K Agent** (`gsm8k_agent.py`): An agent that:
   - Breaks down math problems into step-by-step solutions
   - Uses LLM for reasoning and problem decomposition
   - Logs all steps using CausalFlow's TraceLogger
   - Verifies calculations using the mathematical reexecutor
   - Maintains full dependency tracking between steps

3. **Experiment Runner** (`run_gsm8k_experiment.py`): A script that:
   - Loads GSM8K dataset (configurable number of rows)
   - Runs the agent on each problem
   - Performs CausalFlow analysis on failures
   - Generates comprehensive reports
   - Shows the benefits of causal attribution

## Key Features

### 1. Robust Mathematical Verification

The mathematical reexecutor handles various answer formats:
```python
"42"                    # Simple number
"#### 1234"            # GSM8K format
"$1,234.56"            # Currency with formatting
"The answer is 42"     # Natural language
"3.5 kg"               # Numbers with units
```

### 2. Complete Trace Logging

Every step is logged with dependencies:
- Initial problem understanding
- LLM reasoning steps
- Calculator tool calls
- Tool responses
- Final answer

### 3. Causal Attribution on Failures

When the agent fails, CausalFlow automatically:
- Identifies which steps caused the failure
- Calculates Causal Responsibility Scores (CRS)
- Generates minimal counterfactual repairs
- Validates findings through multi-agent critique

## Installation

1. Install additional dependencies:
```bash
pip install datasets huggingface_hub
```

2. Ensure your `.env` file contains your OpenRouter API key:
```
OPENROUTER_SECRET_KEY=your_api_key_here
```

## Usage

### Basic Usage

Run with default settings (5 problems from sample data):
```bash
python run_gsm8k_experiment.py
```

### Custom Number of Rows

Specify how many problems to solve:
```bash
python run_gsm8k_experiment.py --num-rows 10
```

### Using HuggingFace Dataset

Load from HuggingFace (requires internet connection):
```bash
python run_gsm8k_experiment.py --num-rows 20
```

Or force sample data:
```bash
python run_gsm8k_experiment.py --num-rows 5 --use-sample
```

### Skip CausalFlow Analysis

Run without analyzing failures (faster):
```bash
python run_gsm8k_experiment.py --num-rows 10 --no-analysis
```

### Custom Output Directory

```bash
python run_gsm8k_experiment.py --num-rows 5 --output-dir my_results
```

## Output Files

The experiment generates several files in the output directory:

### Per-Problem Files
- `trace_{i}.json`: Complete execution trace for problem i
- `analysis_{i}.txt`: Human-readable causal analysis (for failures)
- `analysis_{i}.json`: Structured analysis results (for failures)

### Summary Files
- `experiment_summary.json`: Overall statistics and results
- `causalflow_benefits.txt`: Report highlighting CausalFlow benefits

## Example Output

```
Running experiment on 5 problems...

======================================================================
Problem 1/5
Question: Janet's ducks lay 16 eggs per day...
Gold Answer: 18
Agent Answer: 18
Success: True

======================================================================
Problem 2/5
Question: A robe takes 2 bolts of blue fiber...
Gold Answer: 3
Agent Answer: 3
Success: True

======================================================================
EXPERIMENT SUMMARY
======================================================================
Total problems: 5
Correct: 4 (80.0%)
Incorrect: 1 (20.0%)
Failures analyzed: 1/1

Results saved to: gsm8k_results/
Summary: gsm8k_results/experiment_summary.json
```

## Benefits of CausalFlow Architecture

### 1. **Automated Failure Diagnosis**
Instead of manually inspecting traces, CausalFlow automatically identifies:
- Which reasoning steps caused the error
- Why the agent made incorrect decisions
- How errors propagated through the reasoning chain

### 2. **Counterfactual Repairs**
For each failure, CausalFlow generates:
- Minimal edits that would fix the error
- Alternative reasoning that would lead to correct answers
- Actionable insights for model improvement

### 3. **Multi-Agent Validation**
Causal claims are validated by multiple LLMs:
- Reduces false positives in attribution
- Provides confidence scores
- Ensures robustness of findings

### 4. **Detailed Trace Analysis**
Complete visibility into agent behavior:
- Every reasoning step captured
- Full dependency graph
- Tool calls and responses
- Success/failure outcomes

## Architecture Benefits for GSM8K

The implementation demonstrates several key benefits:

1. **Step-by-Step Verification**: Each calculation is verified independently using the non-LLM reexecutor

2. **Error Localization**: When an agent fails, we can pinpoint exactly which step(s) caused the failure

3. **Robust Answer Extraction**: Handles various answer formats that LLMs might generate

4. **Dependency Tracking**: Maintains complete causal graph of reasoning steps

5. **Scalable Analysis**: Can run on any number of problems with configurable analysis

## Example Causal Analysis

When a problem fails, CausalFlow generates detailed reports:

```
CAUSAL ATTRIBUTION REPORT
=========================

Problem: A store has 120 items...
Expected Answer: 93
Agent Answer: 95

Causal Steps Identified:
- Step 4 [reasoning]: CRS = 1.0
  "Morning sales were 42 items. However, I'll round this to 40 for simplicity."

  Analysis: This rounding error propagated through all subsequent steps,
  causing the final answer to be off by 2.

Counterfactual Repair:
  Replace: "round this to 40 for simplicity"
  With: "use the exact value of 42"

  This minimal edit would have prevented the failure.
```

## Testing Individual Components

### Test Mathematical Reexecutor
```bash
python math_reexecutor.py
```

### Test GSM8K Agent
```bash
python gsm8k_agent.py
```

## Extending the Implementation

### Custom Reexecutor

You can create a custom reexecutor by extending `MathReexecutor`:
```python
class CustomReexecutor(MathReexecutor):
    def extract_number(self, text: str) -> Optional[float]:
        # Add custom extraction logic
        return super().extract_number(text)
```

### Custom Agent

Extend `GSM8KAgent` for different reasoning strategies:
```python
class CustomGSM8KAgent(GSM8KAgent):
    def solve(self, question: str, gold_answer: Optional[str] = None):
        # Implement custom solving logic
        return super().solve(question, gold_answer)
```

## Troubleshooting

### Issue: "OPENROUTER_SECRET_KEY not found"
**Solution**: Create a `.env` file with your OpenRouter API key:
```
OPENROUTER_SECRET_KEY=sk-or-v1-...
```

### Issue: "Failed to load from HuggingFace"
**Solution**: Use the `--use-sample` flag to use built-in sample data:
```bash
python run_gsm8k_experiment.py --use-sample
```

### Issue: "ModuleNotFoundError: No module named 'datasets'"
**Solution**: Install the datasets library:
```bash
pip install datasets huggingface_hub
```

## Performance Considerations

- **API Costs**: Each problem requires LLM calls for reasoning. Failed problems require additional calls for causal analysis.
- **Time**: Analysis can be slow for many problems. Consider using `--no-analysis` for quick testing.
- **Accuracy**: The agent's accuracy depends on the underlying LLM model. You can change models in `llm_client.py`.

## Citation

If you use this implementation in your research, please cite the CausalFlow framework:

```
@article{causalflow2024,
  title={CausalFlow: A Comprehensive Framework for Causal Attribution in LLM Agents},
  year={2024}
}
```

## License

See main repository LICENSE file.
