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
   # Edit .env and add your OpenRouter API key and MongoDB URI
   ```

   Your `.env` file should contain:
   ```
   OPENROUTER_SECRET_KEY=your_api_key_here
   MONGODB_URI=mongodb://localhost:27017/causalflow
   ```

   For MongoDB Atlas (cloud):
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/causalflow?retryWrites=true&w=majority
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

## MongoDB Integration

CausalFlow automatically saves all traces and analysis results to MongoDB for centralized storage and querying.

### Collections

- **`runs`**: Stores experiment runs with nested passing and failing traces

### Data Structure

Each run document contains:
```json
{
  "run_id": "run_GSM8K_2025-11-20T10:30:00",
  "experiment_name": "GSM8K",
  "timestamp": "2025-11-20T10:30:00",
  "num_problems": 10,
  "stats": {
    "total": 10,
    "passing": 7,
    "failing": 3
  },
  "passing_traces": [
    {
      "problem_id": 0,
      "timestamp": "2025-11-20T10:31:00",
      "success": true,
      "problem_statement": "...",
      "gold_answer": "42",
      "final_answer": "42",
      "trace": {
        "steps": [...],  // Complete trace as object - never stripped
        "success": true,
        "final_answer": "42",
        "num_steps": 5
      }
    }
  ],
  "failing_traces": [
    {
      "problem_id": 2,
      "timestamp": "2025-11-20T10:32:00",
      "success": false,
      "problem_statement": "...",
      "gold_answer": "42",
      "final_answer": "40",
      "trace": {
        "steps": [...],  // Complete trace as object
        "success": false
      },
      "analysis": {
        "causal_graph": {...},
        "causal_attribution": {...},
        "counterfactual_repairs": {...},
        "multi_agent_critique": {...}
      },
      "metrics": {
        "minimality": {...},
        "attribution": {...},
        "repairs": {...},
        "multi_agent": {...}
      },
      "reports": {
        "full_report": "...",
        "attribution_report": "...",
        "repair_report": "...",
        "critique_report": "..."
      }
    }
  ]
}
```

### Usage

```python
from mongodb_storage import MongoDBStorage
from causal_flow import CausalFlow

# Initialize MongoDB storage
mongo = MongoDBStorage()  # Uses MONGODB_URI from .env

# Create a new run
run_id = mongo.create_run(
    experiment_name="GSM8K",
    num_problems=10
)

# Pass to CausalFlow
flow = CausalFlow(mongo_storage=mongo)

# Add passing traces
mongo.add_passing_trace(
    run_id=run_id,
    trace_data=trace.to_json(),
    problem_id=0,
    problem_statement="What is 2+2?",
    gold_answer="4",
    final_answer="4"
)

# Failing traces are automatically saved during CausalFlow analysis

# Get run statistics
run_stats = mongo.get_run_statistics(run_id)
print(f"Total traces: {run_stats['total_traces']}")
print(f"Passing: {run_stats['passing_traces']}")
print(f"Failing: {run_stats['failing_traces']}")
print(f"Accuracy: {run_stats['accuracy']}")

# Get overall statistics across all runs
overall_stats = mongo.get_statistics()
print(f"Total runs: {overall_stats['total_runs']}")
print(f"Total traces: {overall_stats['total_traces']}")
```

### Querying Examples

```python
# Get a specific run
run = mongo.get_run(run_id)

# Get all passing traces from a run
passing = run['passing_traces']

# Get all failing traces from a run
failing = run['failing_traces']

# Get statistics for all runs
all_runs_stats = mongo.get_all_runs_statistics()
for run_stat in all_runs_stats:
    print(f"{run_stat['run_id']}: {run_stat['accuracy']:.2%}")

# Close connection when done
mongo.close()
```

### Running Without MongoDB

MongoDB is optional. To disable:

```python
# In experiments
experiment = GSM8KExperiment(api_key=api_key, use_mongodb=False)

# In code
flow = CausalFlow(mongo_storage=None)
```