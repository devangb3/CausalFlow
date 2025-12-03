import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from causal_flow import CausalFlow
from llm_client import LLMClient
from mongodb_storage import MongoDBStorage
from trace_logger import TraceLogger
from experiments.humaneval.docker_code_executor import DockerCodeExecutor
from experiments.humaneval.humaneval_agent import HumanevalAgent
from experiments.humaneval.humaneval_reexecutor import HumanevalReexecutor


class HumanevalDataLoader:
    def load_data(self, num_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        dataset = load_dataset("openai_humaneval")["test"]
        items = []
        for row in dataset:
            items.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "tests": row["test"],
                    "entry_point": row["entry_point"],
                }
            )
        if num_rows is not None:
            items = items[:num_rows]
        return items


class HumanevalExperiment:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_loader = HumanevalDataLoader()

        self.executor = DockerCodeExecutor()
        self.reexecutor = HumanevalReexecutor(self.executor)
        self.agent = HumanevalAgent(LLMClient(api_key=self.api_key), self.reexecutor)

        self.mongo_storage = None
        try:
            self.mongo_storage = MongoDBStorage()
        except Exception as e:
            raise Exception(f"Could not initialize MongoDB storage: {e}")

        self.causal_flow = CausalFlow(api_key=self.api_key, mongo_storage=self.mongo_storage)

    def run(self, num_rows: int = 5):
        problems = self.data_loader.load_data(num_rows)
        print(f"Running Humaneval on {len(problems)} tasks")

        run_id = None
        if self.mongo_storage:
            run_id = self.mongo_storage.create_run(
                experiment_name="Humaneval",
                num_problems=len(problems),
            )

        stats = {
            "total": len(problems),
            "passed": 0,
            "failed": 0,
            "analyzed": 0,
            "results": [],
        }

        for idx, task in enumerate(tqdm(problems, desc="Solving tasks")):
            task_id = task["task_id"]
            prompt = task["prompt"]
            tests = task["tests"]
            entry_point = task["entry_point"]
            clean_tests = self.agent.cleanup_tests(tests)
            print(f"\nTask {idx + 1}/{len(problems)}: {task_id} ({entry_point})")
            try:
                result = self.agent.solve(task_id, prompt, clean_tests, entry_point)
            except Exception as e:
                print(f"Error generating solution for {task_id}: {e}")
                stats["failed"] += 1
                continue

            trace: TraceLogger = result["trace"]
            success = result["success"]
            logs = result["logs"]

            stats["results"].append(
                {
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "success": success,
                    "logs": logs,
                }
            )

            if success:
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
                print(f"  Tests failed for {task_id}. Running CausalFlow analysis.")
                try:
                    execution_context = {
                        "prompt": prompt,
                        "tests": clean_tests,
                        "entry_point": entry_point,
                        "task_id": task_id
                    }
                    analysis = self.causal_flow.analyze_trace(
                        trace,
                        reexecutor=self.reexecutor,
                        execution_context=execution_context
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
                except Exception as e:
                    print(f"  Error during CausalFlow analysis: {e}")

        print("\nExperiment complete.")
        print(f"Passed: {stats['passed']} / {stats['total']}")
        print(f"Failed: {stats['failed']} / {stats['total']}")
        print(f"Accuracy: {stats['passed'] / stats['total']:.2%}")


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_SECRET_KEY not found in .env file")
        return

    experiment = HumanevalExperiment(api_key=api_key)
    experiment.run(num_rows=164)


if __name__ == "__main__":
    main()
