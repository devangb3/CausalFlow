import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from causal_flow import CausalFlow
from llm_client import LLMClient
from math_reexecutor import MathReexecutor
from mongodb_storage import MongoDBStorage
from trace_logger import TraceLogger, StepType
from experiments.gsm8k.gsm8k_agent import GSM8KAgent


class GSM8KDataLoader:
    def __init__(self) -> None:
        self.reexecutor = MathReexecutor()

    def load_data(self, num_rows: Optional[int] = None) -> List[Dict[str, str]]:
        dataset = load_dataset("gsm8k", "main", split="test")
        data = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
        print(f"Loaded {len(data)} examples from HuggingFace")

        if num_rows is not None:
            data = data[:num_rows]

        return data

    def extract_gold_answer(self, answer_text: str) -> str:
        num = self.reexecutor.extract_number(answer_text)
        return str(num) if num is not None else answer_text


class GSM8KExperiment:
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash",
    ):
        self.api_key = api_key
        self.model = model
        self.data_loader = GSM8KDataLoader()

        self.agent = GSM8KAgent(
            llm_client=LLMClient(api_key=self.api_key, model=self.model),
            model=self.model,
        )

        self.mongo_storage: Optional[MongoDBStorage] = None
        try:
            self.mongo_storage = MongoDBStorage()
        except Exception as exc:
            raise RuntimeError(f"Could not initialize MongoDB storage: {exc}") from exc

        self.causal_flow = CausalFlow(
            api_key=self.api_key,
            model=self.model,
            mongo_storage=self.mongo_storage,
        )

    def run(
        self,
        num_rows: Optional[int] = None,
        skip_critique: bool = False,
    ) -> Dict[str, Any]:
        experiment_start_time = time.time()

        data = self.data_loader.load_data(num_rows)
        print(f"\nRunning GSM8K experiment on {len(data)} problems")

        run_id = self.mongo_storage.create_run(
            experiment_name="GSM8K",
            num_problems=len(data),
            model_used=self.model,
        )

        stats: Dict[str, int] = {
            "total": len(data),
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "analyzed": 0,
            "fixed": 0,
        }

        for idx, item in enumerate(tqdm(data, desc="Solving GSM8K problems")):
            question = item["question"]
            gold_answer = self.data_loader.extract_gold_answer(item["answer"])
            problem_id = f"gsm8k_{idx}"

            try:
                trace: TraceLogger = self.agent.solve(question, gold_answer)
            except Exception as exc:
                print(f"Error solving problem {problem_id}: {exc}")
                stats["errors"] += 1
                continue

            print(f"Agent Answer: {trace.final_answer}")
            print(f"Success: {trace.success}")

            if trace.success:
                stats["correct"] += 1
                try:
                    self.mongo_storage.add_passing_trace(
                        run_id=run_id,
                        trace_data=trace.to_json(),
                        problem_id=problem_id,
                        problem_statement=question,
                        gold_answer=gold_answer,
                        final_answer=trace.final_answer or "",
                    )
                except Exception as exc:
                    print(f"  Error saving passing trace to MongoDB: {exc}")
            else:
                stats["incorrect"] += 1

                # Run CausalFlow analysis on failing traces
                causal_flow_start_time = time.time()
                causal_flow_analysis_time_minutes: Optional[float] = None

                try:
                    # Build execution_context (causal_attribution.py assumes .get("logs") exists)
                    logs = "\n".join([
                        str(step.tool_output)
                        for step in trace.steps
                        if step.step_type == StepType.TOOL_RESPONSE and step.tool_output
                    ])
                    execution_context: Dict[str, Any] = {
                        "question": question,
                        "gold_answer": gold_answer,
                        "agent_final_answer": trace.final_answer or "",
                        "logs": logs,
                        "problem_id": problem_id,
                    }

                    analysis = self.causal_flow.analyze_trace(
                        trace,
                        reexecutor=None,  # Use LLM outcome prediction
                        execution_context=execution_context,
                        skip_critique=skip_critique,
                        intervene_step_types={StepType.TOOL_CALL, StepType.LLM_RESPONSE, StepType.REASONING, StepType.TOOL_RESPONSE},
                    )

                    causal_flow_analysis_time_seconds = time.time() - causal_flow_start_time
                    causal_flow_analysis_time_minutes = causal_flow_analysis_time_seconds / 60.0

                    stats["analyzed"] += 1

                    # Check if any repairs were successful
                    repair_metrics = analysis.get("metrics", {}).get("repair_metrics", {})
                    if repair_metrics.get("successful_repairs", 0) > 0:
                        stats["fixed"] += 1

                    self.mongo_storage.add_failing_trace(
                        run_id=run_id,
                        trace_data=trace.to_json(),
                        problem_id=problem_id,
                        problem_statement=question,
                        gold_answer=gold_answer,
                        final_answer=trace.final_answer or "",
                        analysis_results=analysis,
                        metrics=analysis.get("metrics", {}),
                        causal_flow_analysis_time_minutes=causal_flow_analysis_time_minutes,
                    )

                except Exception as exc:
                    if causal_flow_analysis_time_minutes is None:
                        causal_flow_analysis_time_seconds = time.time() - causal_flow_start_time
                        causal_flow_analysis_time_minutes = causal_flow_analysis_time_seconds / 60.0
                    print(f"  Error during CausalFlow analysis: {exc}")

        # Update run with final statistics
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        total_experiment_time = time.time() - experiment_start_time
        total_experiment_time_minutes = total_experiment_time / 60.0

        try:
            self.mongo_storage.update_run_statistics(
                run_id=run_id,
                fixed=stats["fixed"],
                analyzed=stats["analyzed"],
                accuracy=accuracy,
                total_experiment_time_minutes=total_experiment_time_minutes,
            )
        except Exception as exc:
            print(f"Error updating run statistics: {exc}")

        print("\n" + "=" * 50)
        print("EXPERIMENT COMPLETE")
        print("=" * 50)
        print(f"Total problems: {stats['total']}")
        print(f"Correct: {stats['correct']} ({accuracy:.1%})")
        print(f"Incorrect: {stats['incorrect']}")
        print(f"Errors: {stats['errors']}")
        print(f"Analyzed: {stats['analyzed']}")
        print(f"Fixed: {stats['fixed']}")
        print(f"Total experiment time: {total_experiment_time_minutes:.2f} minutes")

        return {
            "stats": stats,
            "accuracy": accuracy,
            "run_id": run_id,
            "total_experiment_time_minutes": total_experiment_time_minutes,
        }


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in .env file")

    experiment = GSM8KExperiment(
        api_key=api_key,
        model="google/gemini-2.0-flash-lite-001", #Using an earlier model to prevent Data Contamination from recent models
    )

    experiment.run(
        num_rows=None, #Run on all problems
        skip_critique=False,
    )

if __name__ == "__main__":
    main()
