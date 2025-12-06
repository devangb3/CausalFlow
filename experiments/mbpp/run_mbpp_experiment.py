import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from causal_flow import CausalFlow
from llm_client import LLMClient
from mongodb_storage import MongoDBStorage
from trace_logger import TraceLogger, StepType
from experiments.humaneval.docker_code_executor import DockerCodeExecutor
from experiments.humaneval.humaneval_agent import HumanevalAgent
from experiments.humaneval.humaneval_reexecutor import HumanevalReexecutor
from experiments.mbpp.mbpp_loader import MBPPDataLoader
from reexecution_utils import AgentBranchExecutor


class MBPPExperiment:
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        self.api_key = api_key
        self.data_loader = MBPPDataLoader(dataset_name="mbpp", split="train")

        self.executor = DockerCodeExecutor()
        self.reexecutor = HumanevalReexecutor(self.executor)
        self.agent = HumanevalAgent(LLMClient(api_key=self.api_key, model=model), self.reexecutor)
        self.branch_executor = AgentBranchExecutor(self.agent)

        self.mongo_storage = None
        try:
            self.mongo_storage = MongoDBStorage()
        except Exception as exc:
            raise RuntimeError(f"Could not initialize MongoDB storage: {exc}") from exc

        self.causal_flow = CausalFlow(api_key=self.api_key, model=model, mongo_storage=self.mongo_storage)

    def run(self, num_rows: int = 50):
        problems = self.data_loader.load_data(num_rows)
        print(f"Running MBPP on {len(problems)} tasks")
       
        run_id = self.mongo_storage.create_run(
            experiment_name="MBPP",
            num_problems=len(problems),
        )

        stats: Dict[str, int | List[Dict[str, object]]] = {
            "total": len(problems),
            "passed": 0,
            "failed": 0,
            "analyzed": 0,
            "results": [],
        }

        for idx, task in enumerate(tqdm(problems, desc="Solving MBPP tasks")):
            task_id = task["task_id"]
            prompt = task["prompt"]
            clean_tests = self.agent.cleanup_tests(task["tests"])
            entry_point = self.agent.resolve_entry_point(task["entry_point"], clean_tests)
            
            print(f"\nTask {idx + 1}/{len(problems)}: {task_id} ({entry_point})")
            try:
                trace: TraceLogger = self.agent.solve(task_id, prompt, clean_tests, entry_point)
            except Exception as exc:
                print(f"Error generating solution for {task_id}: {exc}")
                stats["failed"] += 1
                continue
            stats["results"].append(
                {
                    "task_id": task_id,
                    "success": trace.success,
                }
            )

            if trace.success:
                stats["passed"] += 1
                self.mongo_storage.add_passing_trace(
                    run_id=run_id,
                    trace_data=trace.to_json(),
                    problem_id=task_id,
                    problem_statement=prompt,
                    gold_answer="pass",
                    final_answer="pass",
                )
            else:
                stats["failed"] += 1
                print(f"Tests failed for {task_id}. Running CausalFlow analysis")
                try:
                    logs = "\n".join([step.tool_output for step in trace.steps if step.step_type == StepType.TOOL_RESPONSE])
                    execution_context = {
                        "prompt": prompt,
                        "tests": clean_tests,
                        "entry_point": entry_point,
                        "task_id": task_id,
                        "logs": logs
                    }
                    analysis = self.causal_flow.analyze_trace(
                        trace,
                        reexecutor=self.agent,
                        execution_context=execution_context,
                    )
                    metrics = analysis["metrics"]
                    stats["analyzed"] += 1

                    self.mongo_storage.add_failing_trace(
                        run_id=run_id,
                        trace_data=trace.to_json(),
                        problem_id=task_id,
                        problem_statement=prompt,
                        gold_answer="pass",
                        final_answer="fail",
                        analysis_results=analysis,
                        metrics=metrics,
                    )
                except Exception as exc:
                    print(f"  Error during CausalFlow analysis: {exc}")

        print("\nExperiment complete.")
        print(f"Passed: {stats['passed']} / {stats['total']}")
        print(f"Failed: {stats['failed']} / {stats['total']}")
        print(f"Accuracy: {stats['passed'] / stats['total']:.2%}")


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in .env file")

    experiment = MBPPExperiment(api_key=api_key, model="openai/gpt-3.5-turbo")
    experiment.run(num_rows=50)


if __name__ == "__main__":
    main()

