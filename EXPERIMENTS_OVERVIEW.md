# CausalFlow Experiments

## Framework Snapshot
- Full pipeline follows `research/CausalFlow.pdf`: trace logging of every reasoning/tool/final step, DAG construction over dependencies, step-level interventions for causal attribution, counterfactual repair search, and optional multi-agent critique to confirm causal steps.
- Traces are serialized via `TraceLogger` and stored in MongoDB per run; failures trigger `CausalFlow.analyze_trace` with configurable re-execution/critique and metrics (repair success, CRS, minimality, critique consensus).
- Deterministic re-executors are used where feasible (math evaluation, dockerized code tests); otherwise the LLM predicts outcome flips after interventions.

## Implemented Experiments
- **GSM8K** (`experiments/gsm8k/run_gsm8k_experiment.py`)
  - Data: HuggingFace `gsm8k` test split (1,319 problems).
  - Agent: `GSM8KAgent` decomposes problems into structured steps, evaluates each expression with `MathReexecutor`, and grades numeric final answers. Failing traces invoke CausalFlow with interventions over reasoning/tool/LLM steps; outcome prediction uses the LLM (no deterministic reexecutor).
  - Model in run script: `google/gemini-2.0-flash-lite-001` (commented to avoid contamination); critique enabled by default.
- **MBPP** (`experiments/mbpp/run_mbpp_experiment.py`)
  - Data: HF `mbpp` with train/test/validation/prompt splits merged by `MBPPDataLoader`.
  - Agent: Reuses Humaneval-style code generator (`HumanevalAgent`) with docker execution for tests and deterministic `HumanevalReexecutor`. CausalFlow runs only on failures, skipping critique because docker results provide ground truth; repairs are attempted via agent branching.
  - Model in run script: `openai/gpt-5-chat`.
- **Browse-based QA** (shared `BrowseCompAgent` in `experiments/browsecomp/`)
  - Agent: Structured tool policy over `web_search`, `web_fetch`, `extract`, `answer`; deterministic caching via `WebEnvironment` and Serper search. Grading uses an LLM checker (`grade_response`). CausalFlow analyzes failures with interventions on tool/LLM/reasoning; web logs are truncated before analysis.
  - Datasets/runs:
    - **Seal QA Hard** (`run_sealqa_experiment.py`): HF `vtllms/sealqa` (`seal_hard`), defaults to `google/gemini-3-flash-preview`.
    - **MedBrowseComp** (`run_medbrowsecomp_experiment.py`): HF `AIM-Harvard/MedBrowseComp_CUA`, defaults to `google/gemini-3-flash-preview`, max 10 steps.

## Results Snapshot (from provided sheet)
| Experiment | Dataset | Model used | Total | Passed | Failed | CausalFlow fixes | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k/viewer/main/test | google/gemini-2.0 | 1319 | 989 | 330 | 173 | Used LLM as predictor; deterministic repair was impractical. |
| MBPP | https://huggingface.co/datasets/Muennighoff/mbpp | GPT 5 Chat | 947 | 523 | 488 | 201 | Deterministic docker reexecution; critique skipped. |
| Seal QA Hard | https://huggingface.co/datasets/vtllms/sealqa/viewer/seal_hard | Gemini 3 Flash | 254 | 108 | 146 | 32 | Web-search agent with LLM grading. |
| Med BrowseComp | https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp_CUA | Gemini 3 Flash | 484 | 149 | 335 | 149 | Web-search agent with medical domain queries. |

## Operational Notes
- Requires `OPENROUTER_SECRET_KEY` for model access; web experiments also need `SERPER_API_KEY`. Docker is required for MBPP/Humaneval reexecution.
- CausalFlow storage and run metadata are persisted in MongoDB (see `mongodb_storage.py`); run scripts update stats with accuracy, analyzed failures, and repair counts.
- Cached web fetch/search results live under `.cache/browsecomp` to keep browsing deterministic across runs.
