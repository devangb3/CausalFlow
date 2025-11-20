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

- **`passing_traces`**: Stores successful execution traces
- **`failing_traces`**: Stores failed traces with complete CausalFlow analysis including:
  - Full trace data (never stripped or sliced)
  - All reports (attribution, repair, critique, full report)
  - All metrics (minimality scores, precision, recall, F1, consensus scores)
  - Complete analysis results

### Usage

```python
from mongodb_storage import MongoDBStorage
from causal_flow import CausalFlow

# Initialize MongoDB storage
mongo = MongoDBStorage()  # Uses MONGODB_URI from .env

# Pass to CausalFlow
flow = CausalFlow(mongo_storage=mongo)

# Traces are automatically saved during analysis
results = flow.analyze_trace(trace, problem_id=1)

# Get statistics
stats = mongo.get_statistics()
print(f"Total traces: {stats['total_traces']}")
print(f"Passing: {stats['total_passing_traces']}")
print(f"Failing: {stats['total_failing_traces']}")
```

### Data Structure

Each failing trace document contains:
```json
{
  "trace_id": "trace_1_2025-11-20T10:30:00",
  "problem_id": 1,
  "timestamp": "2025-11-20T10:30:00",
  "success": false,
  "problem_statement": "...",
  "gold_answer": "42",
  "final_answer": "40",
  "trace": { /* Complete trace with all steps */ },
  "analysis": {
    "causal_graph": { /* Graph statistics */ },
    "causal_attribution": { /* CRS scores, causal steps */ },
    "counterfactual_repairs": { /* Repair proposals */ },
    "multi_agent_critique": { /* Consensus results */ }
  },
  "metrics": {
    "minimality": { /* average, min, max, by_step */ },
    "attribution": { /* precision, recall, F1, TP, FP, FN */ },
    "repairs": { /* success_rate, repairs_by_step */ },
    "multi_agent": { /* consensus_score, steps_with_agreement */ }
  },
  "reports": {
    "full_report": "...",
    "attribution_report": "...",
    "repair_report": "...",
    "critique_report": "..."
  }
}
```

### Querying Examples

```python
# Get a specific trace
trace = mongo.get_trace("trace_1_2025-11-20T10:30:00")

# Check if trace exists
exists = mongo.trace_exists("trace_1_2025-11-20T10:30:00")

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