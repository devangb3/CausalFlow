import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
import random

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from causal_flow import CausalFlow
from llm_client import LLMClient
from mongodb_storage import MongoDBStorage
from trace_logger import TraceLogger, StepType

from experiments.browsecomp.browsecomp_eval import grade_response
from experiments.browsecomp.browsecomp_agent import BrowseCompAgent
from experiments.browsecomp.web_env import WebEnvironment


class MedBrowseCompExperiment:
    def __init__(
        self,
        api_key: str,
        solver_model: str = "google/gemini-3-flash-preview",
        search_api_key: Optional[str] = None,
        max_steps: int = 15,
    ):
        self.api_key = api_key
        self.solver_model = solver_model
        
        self.solver_llm = LLMClient(api_key=api_key, model=solver_model, temperature=0.3)
        
        self.web_env = WebEnvironment(
            search_api_key=search_api_key,
        )
        
        self.agent = BrowseCompAgent(
            llm_client=self.solver_llm,
            web_env=self.web_env,
            max_steps=max_steps,
        )
        
        self.mongo_storage: Optional[MongoDBStorage] = None
        try:
            self.mongo_storage = MongoDBStorage()
        except Exception as exc:
            raise Exception(f"Could not initialize MongoDB storage: {exc}")
 
        self.causal_flow = CausalFlow(
            api_key=api_key,
            model=solver_model,
            mongo_storage=self.mongo_storage,
        )
    
    def load_medbrowsecomp_examples(
        self,
        num_examples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        try:
            dataset = load_dataset('AIM-Harvard/MedBrowseComp_CUA', split='MedBrowseComp_CUA')
            data = [{'question': item['prompt'], 'answer': item['gold']} for item in dataset]
            print(f"Loaded {len(data)} MedBrowseComp examples from HuggingFace")
        except Exception as e:
            print(f"Failed to load MedBrowseComp from HuggingFace: {e}")
            raise e

        if num_examples:
            rng = random.Random(42)
            data = rng.sample(data, min(num_examples, len(data)))

        return data

    def run(
        self,
        num_examples: Optional[int] = None,
        skip_critique: bool = True,
    ) -> Dict[str, Any]:
        experiment_start_time = time.time()
        
        examples = self.load_medbrowsecomp_examples(num_examples=num_examples)
        print(f"Running MedBrowseComp on {len(examples)} examples")
        
        run_id = self.mongo_storage.create_run(
            experiment_name="MedBrowseComp",
            num_problems=len(examples),
            model_used=self.solver_model,
        )
        
        stats: Dict[str, int] = {
            "total": len(examples),
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "analyzed": 0,
            "repair_success": 0,
        }
        
        results: List[Dict[str, Any]] = []
        
        for idx, row in enumerate(tqdm(examples, desc="Solving MedBrowseComp")):
            problem_id = f"medbrowsecomp_{idx}"
            
            try:
                question = row.get("question", "")
                gold_answer = row.get("answer", "")

                try:
                    trace: TraceLogger = self.agent.solve(
                        problem_id=problem_id,
                        question=question,
                        gold_answer=gold_answer,
                    )
                except Exception as e:
                    print(f"Error solving {problem_id}: {e}")
                    stats["errors"] += 1
                    continue
                
                agent_response = trace.final_answer or ""
                
                try:
                    trace.success = grade_response(gold_answer, agent_response, self.solver_llm) if agent_response else False
                except Exception as e:
                    print(f"Error grading response for {problem_id}: {e}")
                    trace.success = False
                
                result_entry: Dict[str, Any] = {
                    "problem_id": problem_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "agent_response": agent_response,
                    "is_correct": trace.success,
                    "num_steps": len(trace.steps),
                }
                
                if trace.success:
                    stats["correct"] += 1
                    print(f"CORRECT ({len(trace.steps)} steps)")
                    
                    try:
                        self.mongo_storage.add_passing_trace(
                            run_id=run_id,
                            trace_data=trace.to_json(),
                            problem_id=problem_id,
                            problem_statement=question,
                            gold_answer=gold_answer,
                            final_answer=agent_response,
                        )
                    except Exception as e:
                        print(f"Error storing passing trace for {problem_id}: {e}")
                else:
                    stats["incorrect"] += 1
                    
                    causal_flow_analysis_time_minutes: Optional[float] = None
                    causal_flow_start_time = time.time()
                    analysis: Optional[Dict[str, Any]] = None
                    
                    try:
                        web_logs = self._extract_web_logs(trace)
                        execution_context = {
                            "question": question,
                            "gold_answer": gold_answer,
                            "agent_final_answer": agent_response,
                            "logs": web_logs,
                            "problem_id": problem_id,
                        }
                        
                        analysis = self.causal_flow.analyze_trace(
                            trace,
                            reexecutor=self.agent,
                            execution_context=execution_context,
                            skip_critique=skip_critique,
                            intervene_step_types={StepType.TOOL_CALL, StepType.LLM_RESPONSE, StepType.REASONING},
                        )
                        causal_flow_analysis_time_seconds = time.time() - causal_flow_start_time
                        causal_flow_analysis_time_minutes = causal_flow_analysis_time_seconds / 60.0
                        
                        stats["analyzed"] += 1
                        
                        repair_metrics = analysis.get("metrics", {}).get("repair_metrics", {})
                        if repair_metrics.get("successful_repairs", 0) > 0:
                            stats["repair_success"] += 1
                        
                        result_entry["analysis"] = analysis
                        
                    except Exception as e:
                        if causal_flow_analysis_time_minutes is None:
                            causal_flow_analysis_time_seconds = time.time() - causal_flow_start_time
                            causal_flow_analysis_time_minutes = causal_flow_analysis_time_seconds / 60.0
                        print(f"CausalFlow analysis error for {problem_id}: {e}")
                        result_entry["analysis_error"] = str(e)
                    
                    try:
                        self.mongo_storage.add_failing_trace(
                            run_id=run_id,
                            trace_data=trace.to_json(),
                            problem_id=problem_id,
                            problem_statement=question,
                            gold_answer=gold_answer,
                            final_answer=agent_response,
                            analysis_results=analysis if analysis else {},
                            metrics=analysis.get("metrics") if analysis else {},
                            causal_flow_analysis_time_minutes=causal_flow_analysis_time_minutes,
                        )
                    except Exception as e:
                        print(f"Error storing failing trace for {problem_id}: {e}")
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"\n[CRITICAL] Unexpected error processing example {idx} ({problem_id}): {e}")
                print("Skipping this example and continuing with the rest")
                stats["errors"] += 1
                continue
        
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        total_experiment_time = time.time() - experiment_start_time
        total_experiment_time_minutes = total_experiment_time / 60.0
        
        try:
            self.mongo_storage.update_run_statistics(
                run_id=run_id,
                fixed=stats["repair_success"],
                analyzed=stats["analyzed"],
                accuracy=accuracy,
                total_experiment_time_minutes=total_experiment_time_minutes,
            )
        except Exception as e:
            print(f"Error updating run statistics: {e}")
        
        print("\n" + "=" * 50)
        print("EXPERIMENT COMPLETE")
        print("=" * 50)
        print(f"Total examples: {stats['total']}")
        print(f"Correct: {stats['correct']} ({accuracy:.1%})")
        print(f"Incorrect: {stats['incorrect']}")
        print(f"Errors: {stats['errors']}")
        print(f"Analyzed: {stats['analyzed']}")
        print(f"Repair success: {stats['repair_success']}")
        print(f"Total experiment time: {total_experiment_time_minutes:.2f} minutes")
        print(f"Cache stats: {self.web_env.get_cache_stats()}")
        
        return {
            "stats": stats,
            "accuracy": accuracy,
            "results": results,
            "run_id": run_id,
            "total_experiment_time_minutes": total_experiment_time_minutes,
        }
    
    def _extract_web_logs(self, trace: TraceLogger, max_chars: int = 15000) -> str:

        web_logs: List[Dict[str, Any]] = []
        
        for step in trace.steps:
            if step.step_type == StepType.TOOL_CALL:
                if step.tool_name in ("web_search", "web_fetch"):
                    web_logs.append({
                        "step_id": step.step_id,
                        "tool": step.tool_name,
                        "args": step.tool_args,
                    })
            elif step.step_type == StepType.TOOL_RESPONSE:
                if step.tool_name in ("web_search", "web_fetch"):
                    for log in reversed(web_logs):
                        if log["tool"] == step.tool_name and "result" not in log:
                            log["result"] = step.tool_output if step.tool_output else None
                            log["success"] = step.tool_call_result
                            break

        # Filter out None results and convert to strings
        log_strings: List[str] = []
        for log in web_logs:
            if log.get("result") is not None:
                result_str = str(log["result"])
                if result_str.strip():
                    log_strings.append(result_str)
        
        if not log_strings:
            return ""
        
        # Start from the end (most recent logs) and work backwards
        # to prioritize recent information
        total_chars = 0
        selected_logs: List[str] = []
        truncation_warning = ""
        processed_count = 0
        
        for log_str in reversed(log_strings):
            log_chars = len(log_str)
            if total_chars + log_chars <= max_chars:
                selected_logs.insert(0, log_str)  # Insert at beginning to maintain order
                total_chars += log_chars
                processed_count += 1
            else:
                # Try to include a partial log if there's room
                remaining_chars = max_chars - total_chars
                if remaining_chars > 100:  # Only include if meaningful space remains
                    truncated_log = log_str[:remaining_chars - 50] + "\n[...truncated...]"
                    selected_logs.insert(0, truncated_log)
                    processed_count += 1
                    truncated_count = len(log_strings) - processed_count
                    truncation_warning = f"\n[Note: {truncated_count} earlier log entries were truncated due to size limits]"
                else:
                    truncated_count = len(log_strings) - processed_count
                    truncation_warning = f"\n[Note: {truncated_count} log entries were truncated due to size limits]"
                break
        
        result = "\n".join(selected_logs)
        if truncation_warning:
            result += truncation_warning
        
        return result

def main():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in environment")
    
    search_api_key = os.getenv("SERPER_API_KEY")
    if not search_api_key:
        raise RuntimeError("SERPER_API_KEY not found. Web search will fail unless cache is warm.")
    
    experiment = MedBrowseCompExperiment(
        api_key=api_key,
        solver_model="google/gemini-3-flash-preview",
        search_api_key=search_api_key,
        max_steps=10,
    )
    
    results = experiment.run(
        num_examples=None,
        skip_critique=True,
    )
    
    print(f"\nFinal accuracy: {results['accuracy']:.1%}")

if __name__ == "__main__":
    main()
