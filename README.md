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