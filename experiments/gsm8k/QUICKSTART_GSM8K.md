# Quick Start Guide: GSM8K with CausalFlow

This guide will help you quickly get started with the GSM8K implementation of CausalFlow.

## Prerequisites

1. Python 3.8+
2. OpenRouter API key ([Get one here](https://openrouter.ai/keys))

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:
```
OPENROUTER_SECRET_KEY=sk-or-v1-your-actual-key-here
```

### 3. Verify Installation

Run the component tests (no API key required):

```bash
python test_gsm8k_components.py
```

You should see:
```
✓ All tests passed!
```

## Running Your First Experiment

### Option 1: Quick Test (5 problems, ~2 minutes)

```bash
python run_gsm8k_experiment.py --num-rows 5 --use-sample
```

This will:
- Solve 5 sample GSM8K math problems
- Analyze any failures with CausalFlow
- Generate detailed reports

### Option 2: Larger Test (20 problems from HuggingFace)

```bash
python run_gsm8k_experiment.py --num-rows 20
```

This requires internet access to download the GSM8K dataset.

## Understanding the Output

After running the experiment, you'll find in `gsm8k_results/`:

### Key Files

1. **`experiment_summary.json`** - Overall statistics
   - Total problems attempted
   - Accuracy rate
   - Number of failures analyzed

2. **`trace_{i}.json`** - Execution trace for problem i
   - Every reasoning step
   - Tool calls and responses
   - Dependencies between steps

3. **`analysis_{i}.txt`** - Human-readable analysis (for failures)
   - Causal steps identified
   - Why the agent failed
   - Suggested repairs

4. **`causalflow_benefits.txt`** - Summary of CausalFlow benefits

### Example Output

```
======================================================================
EXPERIMENT SUMMARY
======================================================================
Total problems: 5
Correct: 4 (80.0%)
Incorrect: 1 (20.0%)
Failures analyzed: 1/1

Results saved to: gsm8k_results/
```

## Viewing Analysis Results

### For a Successful Problem

```bash
cat gsm8k_results/trace_0.json
```

You'll see the full execution trace with all steps.

### For a Failed Problem

```bash
cat gsm8k_results/analysis_1.txt
```

You'll see:
- Which steps caused the failure (Causal Responsibility Score)
- What went wrong in the reasoning
- Suggested minimal repairs
- Multi-agent critique validation

## Common Use Cases

### Test with More Problems

```bash
python run_gsm8k_experiment.py --num-rows 50
```

### Skip Analysis (Faster Testing)

```bash
python run_gsm8k_experiment.py --num-rows 10 --no-analysis
```

### Custom Output Directory

```bash
python run_gsm8k_experiment.py --num-rows 5 --output-dir my_experiment
```

## Understanding CausalFlow Benefits

After running an experiment with failures, check:

```bash
cat gsm8k_results/causalflow_benefits.txt
```

This report shows:
1. **Automated Failure Diagnosis** - Root causes identified automatically
2. **Counterfactual Repairs** - Minimal edits that would fix errors
3. **Multi-Agent Validation** - Consensus across multiple LLMs
4. **Detailed Trace Analysis** - Complete reasoning chain captured

## Example: Analyzing a Specific Failure

Let's say problem 3 failed. Here's how to investigate:

```bash
# View the problem and agent's attempt
python -c "import json; data = json.load(open('gsm8k_results/trace_3.json')); print('Question:', data['problem_statement']); print('Agent Answer:', data['final_answer']); print('Gold Answer:', data['gold_answer'])"

# Read the causal analysis
cat gsm8k_results/analysis_3.txt

# View structured results
cat gsm8k_results/analysis_3.json | python -m json.tool
```

## Troubleshooting

### Error: "OPENROUTER_SECRET_KEY not found"

**Solution**: Make sure you created `.env` with your API key:
```bash
echo "OPENROUTER_SECRET_KEY=sk-or-v1-your-key-here" > .env
```

### Error: "Failed to load from HuggingFace"

**Solution 1**: Use sample data:
```bash
python run_gsm8k_experiment.py --use-sample
```

**Solution 2**: Check internet connection and try again

### Tests fail with import errors

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Read the Full Documentation**: See `GSM8K_README.md` for detailed information

2. **Explore the Code**:
   - `math_reexecutor.py` - Mathematical verification
   - `gsm8k_agent.py` - Agent implementation
   - `run_gsm8k_experiment.py` - Experiment orchestration

3. **Customize the Agent**: Modify `gsm8k_agent.py` to implement different reasoning strategies

4. **Analyze More Data**: Run on larger subsets of GSM8K to see patterns in failures

## Key Commands Summary

```bash
# Install
pip install -r requirements.txt

# Test components (no API key needed)
python test_gsm8k_components.py

# Quick experiment (5 problems, sample data)
python run_gsm8k_experiment.py --num-rows 5 --use-sample

# Larger experiment (20 problems, HuggingFace)
python run_gsm8k_experiment.py --num-rows 20

# Custom experiment
python run_gsm8k_experiment.py --num-rows 10 --output-dir results_v2

# Skip analysis (faster)
python run_gsm8k_experiment.py --num-rows 10 --no-analysis
```

## Getting Help

- Full documentation: `GSM8K_README.md`
- Component tests: `python test_gsm8k_components.py`
- Main README: `README.md`

## Architecture Overview

```
User Input (GSM8K Problem)
         ↓
   GSM8KAgent (with TraceLogger)
         ↓
   Step-by-step reasoning + calculations
         ↓
   Mathematical Reexecutor (verification)
         ↓
   Success/Failure determination
         ↓
   (If failed) CausalFlow Analysis
         ↓
   Reports: Causal steps, repairs, critique
```