# CausalFlow: A Comprehensive Framework for Causal Attribution in LLM Agents

CausalFlow is a unified framework for diagnosing agent failures using causal attribution, minimal counterfactual repair, and multi-agent critique. It helps you understand **why** your LLM agents fail by identifying causally responsible steps, proposing minimal fixes, and validating findings through multi-agent consensus.

## Features

- **Trace Extraction**: Capture every step of agent execution (reasoning, tool calls, memory access, etc.)
- **Causal Graph Construction**: Build directed acyclic graphs (DAGs) encoding step dependencies
- **Causal Attribution**: Use intervention-based causal inference to identify failure-causing steps
- **Counterfactual Repair**: Generate minimal edits that would fix failures
- **Multi-Agent Critique**: Validate causal claims through consensus of multiple LLMs
- **Comprehensive Reports**: Human-readable analysis with detailed insights

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Research Background](#research-background)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CF-Implementation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

   Your `.env` file should contain:
   ```
   OPENROUTER_SECRET_KEY=your_api_key_here
   ```

## Quick Start

### Basic Usage

```python
from trace_logger import TraceLogger
from causal_flow import CausalFlow

# 1. Create a trace logger
trace = TraceLogger()

# 2. Log agent execution steps
step_0 = trace.log_reasoning("I need to calculate 5 + 3", dependencies=[])
step_1 = trace.log_tool_call("calculator", {"expression": "5 + 3"}, dependencies=[step_0])
step_2 = trace.log_tool_response(8, dependencies=[step_1])
step_3 = trace.log_final_answer("8", dependencies=[step_2])

# 3. Record outcome
trace.record_outcome(final_answer="8", gold_answer="8")

# 4. Analyze with CausalFlow (if trace failed)
flow = CausalFlow()
results = flow.analyze_trace(trace)

# 5. Generate report
report = flow.generate_full_report("analysis_report.txt")
print(report)
```

### Run the Demo

```bash
# Simple example (beginner)
python examples/demo.py

# Complex example (advanced)
python examples/complex_example.py
```

This will:
1. Create a sample failed trace
2. Run CausalFlow analysis
3. Generate detailed reports
4. Save results to JSON files

## Architecture

CausalFlow consists of five major components:

### 1. Trace Extraction (`trace_logger.py`)

Captures every internal step of agent execution:
- **Reasoning steps**: Chain-of-thought, planning, deliberation
- **Tool calls**: Function invocations with arguments
- **Tool responses**: Outputs from tools
- **Memory access**: Retrieval of stored information
- **Environment actions**: Interactive task actions
- **Final answers**: Agent's declared solution

```python
from trace_logger import TraceLogger

logger = TraceLogger()
step_id = logger.log_reasoning("I need to solve this problem...", dependencies=[])
```

### 2. Causal Graph Construction (`causal_graph.py`)

Transforms trace sequences into directed acyclic graphs (DAGs):

```python
from causal_graph import CausalGraph

graph = CausalGraph(trace)
ancestors = graph.get_ancestors(step_id=5)
descendants = graph.get_descendants(step_id=2)
```

### 3. Causal Attribution (`causal_attribution.py`)

Uses interventions to determine which steps caused failures:

```python
from causal_attribution import CausalAttribution

attribution = CausalAttribution(trace, graph, llm_client)
crs_scores = attribution.compute_causal_responsibility()
causal_steps = attribution.get_causal_steps()
```

**Causal Responsibility Score (CRS)**:
- CRS(i) = 1 if intervening on step i flips outcome to success
- CRS(i) = 0 otherwise

### 4. Counterfactual Repair (`counterfactual_repair.py`)

Generates minimal edits that would have prevented failure:

```python
from counterfactual_repair import CounterfactualRepair

repair = CounterfactualRepair(trace, attribution, llm_client)
repairs = repair.generate_repairs(num_proposals=3)
best_repair = repair.get_best_repair(step_id)
```

**Minimality Score (MS)**:
```
MS = 1 - (tokens_changed / tokens_original)
```

### 5. Multi-Agent Critique (`multi_agent_critique.py`)

Validates causal attributions through multi-agent consensus:

```python
from multi_agent_critique import MultiAgentCritique

critique = MultiAgentCritique(trace, attribution, multi_agent_llm)
results = critique.critique_causal_attributions()
consensus_steps = critique.get_consensus_causal_steps()
```

## 📚 Usage Guide

### Creating an Agent with Trace Logging

```python
from trace_logger import TraceLogger
from llm_client import LLMClient

class MyAgent:
    def __init__(self):
        self.trace = TraceLogger()
        self.llm = LLMClient()

    def solve_task(self, task, gold_answer):
        # Log reasoning
        reasoning = self.llm.generate(f"Solve: {task}")
        step_1 = self.trace.log_reasoning(reasoning, dependencies=[])

        # Log tool call
        step_2 = self.trace.log_tool_call(
            tool_name="calculator",
            tool_args={"expr": "5+3"},
            dependencies=[step_1]
        )

        # Log tool response
        result = eval("5+3")
        step_3 = self.trace.log_tool_response(result, dependencies=[step_2])

        # Log final answer
        answer = str(result)
        step_4 = self.trace.log_final_answer(answer, dependencies=[step_3])

        # Record outcome
        self.trace.record_outcome(answer, gold_answer)

        return answer
```

### Analyzing Failed Traces

```python
from causal_flow import analyze_failed_trace

# Quick analysis with convenience function
results = analyze_failed_trace(
    trace=my_agent.trace,
    generate_report=True,
    report_file="failure_analysis.txt"
)
```

### Full Pipeline Example

```python
from causal_flow import CausalFlow

# Initialize CausalFlow
flow = CausalFlow(
    model="openai/gpt-4-turbo-preview",
    num_critique_agents=3
)

# Analyze trace
results = flow.analyze_trace(
    trace=failed_trace,
    skip_repair=False,   # Generate repairs
    skip_critique=False  # Run multi-agent critique
)

# Generate comprehensive report
report = flow.generate_full_report("full_analysis.txt")

# Export structured results
flow.export_results("results.json")
```

### Customizing LLM Models

```python
from llm_client import LLMClient, MultiAgentLLM

# Use different models
llm = LLMClient(model="anthropic/claude-3-sonnet")

# Multiple agents with different models
multi_agent = MultiAgentLLM(
    num_agents=3,
    models=[
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-sonnet",
        "google/gemini-pro"
    ]
)
```

## 🔧 API Reference

### TraceLogger

```python
trace = TraceLogger()

# Logging methods
step_id = trace.log_reasoning(text, dependencies=[])
step_id = trace.log_tool_call(tool_name, tool_args, dependencies=[])
step_id = trace.log_tool_response(tool_output, dependencies=[])
step_id = trace.log_memory_access(key, value, dependencies=[])
step_id = trace.log_environment_action(action, dependencies=[])
step_id = trace.log_environment_observation(observation, dependencies=[])
step_id = trace.log_final_answer(answer, dependencies=[])

# Recording outcome
trace.record_outcome(final_answer, gold_answer)

# Serialization
trace.to_json("trace.json")
trace = TraceLogger.from_json("trace.json")
```

### CausalFlow

```python
flow = CausalFlow(api_key=None, model="...", num_critique_agents=3)

# Analysis
results = flow.analyze_trace(trace, skip_repair=False, skip_critique=False)
results = flow.analyze_trace_from_file("trace.json")

# Reports
report = flow.generate_full_report(output_file="report.txt")
flow.export_results("results.json")
```

### CausalAttribution

```python
attribution = CausalAttribution(trace, graph, llm_client)

# Compute causal responsibility
crs_scores = attribution.compute_causal_responsibility()

# Get causal steps
causal_steps = attribution.get_causal_steps(threshold=0.5)
top_steps = attribution.get_top_causal_steps(n=3)

# Generate report
report = attribution.generate_report()
```

### CounterfactualRepair

```python
repair = CounterfactualRepair(trace, attribution, llm_client)

# Generate repairs
repairs = repair.generate_repairs(step_ids=None, num_proposals=3)

# Get best repairs
best = repair.get_best_repair(step_id)
all_best = repair.get_all_best_repairs()

# Generate report
report = repair.generate_report()
```

### MultiAgentCritique

```python
critique = MultiAgentCritique(trace, attribution, multi_agent_llm)

# Perform critique
results = critique.critique_causal_attributions(step_ids=None)

# Get consensus
consensus_steps = critique.get_consensus_causal_steps(threshold=0.5)

# Generate report
report = critique.generate_report()
```

## 📝 Examples

All examples are located in the `examples/` directory. See [examples/README.md](examples/README.md) for detailed documentation.

### Example 1: Simple Demo (`examples/demo.py`)

**Difficulty**: Beginner

A basic example with a simple arithmetic problem (5 + 3) where the agent fabricates information.
- 5 execution steps
- Easy-to-understand error pattern
- Perfect for learning CausalFlow basics

```bash
python examples/demo.py
```

### Example 2: Complex Multi-Step Problem (`examples/complex_example.py`)

**Difficulty**: Advanced

A realistic failure scenario with inventory tracking, percentages, and error propagation.
- 14 execution steps
- Multiple operations (multiplication, addition, subtraction)
- Demonstrates error cascading through correct intermediate steps
- Shows CausalFlow's power on longer reasoning chains

```bash
python examples/complex_example.py
```

### Example 3: Reusable Agent Class (`examples/example_agent.py`)

A complete implementation of `MathReasoningAgent` showing:
- Integration with TraceLogger
- Chain-of-thought reasoning
- Tool calls (calculator)
- Automatic trace logging

See [examples/README.md](examples/README.md) for usage details and comparisons.

### Example 4: Custom Agent Integration

```python
from trace_logger import TraceLogger
from causal_flow import CausalFlow

class MyCustomAgent:
    def __init__(self):
        self.trace = TraceLogger()

    def run(self, task):
        # Your agent logic here
        # Use self.trace.log_*() to capture steps
        pass

    def analyze_failure(self):
        if not self.trace.success:
            flow = CausalFlow()
            results = flow.analyze_trace(self.trace)
            return results
```

## 🔬 Research Background

CausalFlow is based on the research proposal:

**"CausalFlow: A Comprehensive Framework for Causal Attribution, Counterfactual Repair, and Multi-Agent Critique in LLM Agents"**

### Key Concepts

1. **Causal Attribution**: Uses intervention-based causal inference (do-operator) to identify which steps caused failures

2. **Minimality Principle**: Repairs should be as small as possible while correcting the failure

3. **Multi-Agent Triangulation**: Multiple LLMs critique causal claims to reduce variance and improve robustness

### Experimental Domains

The framework is designed for diverse agent tasks:
- **GSM8K**: Arithmetic reasoning
- **StrategyQA**: Commonsense logic
- **ALFWorld**: Embodied planning
- **WebShop**: Web browsing and decision-making
- **Tool-based tasks**: Python, calculator, API functions

## 📊 Output Files

Running CausalFlow generates several files:

- **`trace.json`**: Serialized execution trace
- **`causalflow_report.txt`**: Human-readable analysis report
- **`causalflow_results.json`**: Structured results in JSON format
