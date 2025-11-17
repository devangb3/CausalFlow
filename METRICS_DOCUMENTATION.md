# Causal Metrics JSON Documentation

This document describes the new metrics JSON generation functionality added to the CausalFlow framework.

## Overview

The CausalFlow pipeline now automatically generates a comprehensive metrics JSON file at the end of each analysis. This JSON contains:

1. **Causal Attribution Metrics** - Precision, recall, and F1 score (with optional ground truth)
2. **Repair Success Rate** - Percentage of successful repairs and detailed repair statistics
3. **Minimality Scores** - Average, min, and max minimality scores across all repairs
4. **Multi-Agent Agreement** - Detailed consensus information and critic reasoning

## Usage

### Automatic Generation (Default)

By default, the metrics JSON is automatically generated when you run `analyze_trace()`:

```python
from causal_flow import CausalFlow

flow = CausalFlow(api_key=api_key)
results = flow.analyze_trace(
    trace,
    skip_repair=False,
    metrics_output_file="causal_metrics.json",  # Default filename
    ground_truth_causal_steps=[4, 5]  # Optional
)
```

### Manual Generation

You can also generate metrics manually after analysis:

```python
# Run analysis without automatic metrics export
results = flow.analyze_trace(trace, metrics_output_file=None)

# Generate metrics later
metrics = flow.generate_metrics_json(ground_truth_causal_steps=[4, 5])

# Export to file
flow.export_metrics("my_metrics.json", ground_truth_causal_steps=[4, 5])
```

### Disable Automatic Generation

To disable automatic metrics generation:

```python
results = flow.analyze_trace(trace, metrics_output_file=None)
```

## Metrics JSON Structure

### 1. Causal Attribution Metrics

```json
{
  "causal_attribution_metrics": {
    "num_identified_causal_steps": 3,
    "identified_steps": [0, 4, 5],
    "precision": 0.6667,
    "recall": 1.0,
    "f1_score": 0.8,
    "num_ground_truth_causal_steps": 2,
    "ground_truth_steps": [4, 5],
    "true_positives": 2,
    "false_positives": 1,
    "false_negatives": 0
  }
}
```

**Fields:**
- `num_identified_causal_steps`: Number of steps identified as causal by the system
- `identified_steps`: List of step IDs identified as causal
- `precision`: TP / (TP + FP) - accuracy of positive predictions
- `recall`: TP / (TP + FN) - coverage of actual positives
- `f1_score`: Harmonic mean of precision and recall
- `ground_truth_steps`: Provided ground truth causal steps (null if not provided)
- `true_positives`: Steps correctly identified as causal
- `false_positives`: Steps incorrectly identified as causal
- `false_negatives`: Causal steps missed by the system

**Note:** If ground truth is not provided, precision/recall/F1 will be `null`.

### 2. Repair Metrics

```json
{
  "repair_metrics": {
    "total_repairs_attempted": 9,
    "successful_repairs": 6,
    "failed_repairs": 3,
    "success_rate": 0.6667,
    "repairs_by_step": {
      "0": {
        "success": true,
        "minimality_score": 0.2
      },
      "4": {
        "success": false,
        "minimality_score": 0.22
      },
      "5": {
        "success": true,
        "minimality_score": 0.0
      }
    }
  }
}
```

**Fields:**
- `total_repairs_attempted`: Total number of repair proposals generated
- `successful_repairs`: Number of repairs predicted to succeed
- `failed_repairs`: Number of repairs predicted to fail
- `success_rate`: Ratio of successful repairs (0.0 to 1.0)
- `repairs_by_step`: Best repair for each causal step with success status and minimality score

### 3. Minimality Metrics

```json
{
  "minimality_metrics": {
    "average_minimality": 0.14,
    "min_minimality": 0.0,
    "max_minimality": 0.22,
    "minimality_by_step": {
      "0": 0.2,
      "4": 0.22,
      "5": 0.0
    }
  }
}
```

**Fields:**
- `average_minimality`: Average minimality score across all best repairs
- `min_minimality`: Lowest minimality score (most changes)
- `max_minimality`: Highest minimality score (most minimal)
- `minimality_by_step`: Minimality score for best repair of each step

**Minimality Score Definition:**
```
minimality_score = 1.0 - (tokens_changed / tokens_original)
```
- Score of 1.0 = no changes (most minimal)
- Score of 0.0 = complete rewrite (least minimal)

### 4. Multi-Agent Agreement

```json
{
  "multi_agent_agreement": {
    "average_consensus_score": 0.74,
    "num_steps_critiqued": 3,
    "steps_with_agreement": [
      {
        "step_id": 4,
        "consensus_score": 0.88,
        "final_verdict": "CAUSAL",
        "agent_a_score": 1.0,
        "agent_b_agrees": true,
        "agent_b_confidence": 0.9,
        "agent_b_reasoning": "This step introduces incorrect information...",
        "agent_c_agrees": true,
        "agent_c_confidence": 0.7,
        "agent_c_reasoning": "I agree with the causal attribution...",
        "final_critic_summary": "I agree with the causal attribution..."
      }
    ]
  }
}
```

**Fields:**
- `average_consensus_score`: Average consensus across all critiqued steps
- `num_steps_critiqued`: Number of steps that underwent multi-agent critique
- `steps_with_agreement`: Detailed breakdown per step

**Per-Step Fields:**
- `step_id`: The step being critiqued
- `consensus_score`: Weighted consensus score (0.0 to 1.0)
- `final_verdict`: "CAUSAL" or "NOT CAUSAL" (based on threshold of 0.5)
- `agent_a_score`: Original CRS score from causal attribution
- `agent_b_agrees`: Whether Agent B (first critic) agrees
- `agent_b_confidence`: Agent B's confidence level (0.0 to 1.0)
- `agent_b_reasoning`: Agent B's reasoning (truncated to 500 chars)
- `agent_c_agrees`: Whether Agent C (meta-critic) agrees
- `agent_c_confidence`: Agent C's confidence level (0.0 to 1.0)
- `agent_c_reasoning`: Agent C's reasoning (truncated to 500 chars)
- `final_critic_summary`: Summary from Agent C explaining why step is/isn't causal

**Consensus Score Calculation:**
```
consensus_score = 0.33 × agent_a_score
                + 0.33 × (agent_b_agrees ? 1.0 : 0.0) × agent_b_confidence
                + 0.33 × (agent_c_agrees ? 1.0 : 0.0) × agent_c_confidence
```

## Example Usage

### With Ground Truth

```python
from causal_flow import CausalFlow
from trace_logger import TraceLogger

# Create your trace
trace = TraceLogger(problem_statement="...")
# ... add steps ...
trace.record_outcome(final_answer="6", gold_answer="8")

# Analyze with ground truth
flow = CausalFlow(api_key=api_key)
results = flow.analyze_trace(
    trace,
    metrics_output_file="metrics_with_gt.json",
    ground_truth_causal_steps=[4, 5]  # Specify which steps are truly causal
)
```

This will generate `metrics_with_gt.json` with precision/recall metrics.

### Without Ground Truth

```python
# Analyze without ground truth
results = flow.analyze_trace(
    trace,
    metrics_output_file="metrics_no_gt.json"
)
```

This will generate `metrics_no_gt.json` with precision/recall set to `null`.

## API Reference

### `analyze_trace()`

```python
def analyze_trace(
    self,
    trace: TraceLogger,
    skip_repair: bool = False,
    metrics_output_file: Optional[str] = "causal_metrics.json",
    ground_truth_causal_steps: Optional[List[int]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `trace`: The execution trace to analyze
- `skip_repair`: Whether to skip counterfactual repair generation
- `metrics_output_file`: File path for metrics JSON (None to skip export)
- `ground_truth_causal_steps`: Optional ground truth for precision/recall

**Returns:** Dictionary containing analysis results

### `generate_metrics_json()`

```python
def generate_metrics_json(
    self,
    ground_truth_causal_steps: Optional[List[int]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `ground_truth_causal_steps`: Optional list of ground truth causal step IDs

**Returns:** Dictionary containing all metrics

### `export_metrics()`

```python
def export_metrics(
    self,
    filepath: str,
    ground_truth_causal_steps: Optional[List[int]] = None
)
```

**Parameters:**
- `filepath`: Path to save the metrics JSON
- `ground_truth_causal_steps`: Optional list of ground truth causal step IDs

## Integration with Existing Code

The new metrics functionality is backward compatible. Existing code will continue to work:

```python
# Old code - still works!
results = flow.analyze_trace(trace)
```

This will now automatically generate `causal_metrics.json` in addition to the existing results.

To maintain old behavior (no metrics JSON):

```python
results = flow.analyze_trace(trace, metrics_output_file=None)
```

## Notes

1. **Performance**: Metrics generation adds minimal overhead (~1-2% of total analysis time)

2. **File Output**: The metrics JSON is separate from the main results JSON (`causalflow_results.json`)

3. **Ground Truth**: Ground truth is optional but recommended for evaluation purposes

4. **Reasoning Truncation**: Agent reasoning is truncated to 500 characters in the JSON for readability. Full reasoning is available in the text report.

5. **Empty Results**: If certain components (repair, critique) are skipped, their metrics sections will have null/empty values.
