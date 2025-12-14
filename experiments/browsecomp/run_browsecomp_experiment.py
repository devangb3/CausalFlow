import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any
import re
from dotenv import load_dotenv
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from causal_flow import CausalFlow
from llm_client import LLMClient
from mongodb_storage import MongoDBStorage
from trace_logger import TraceLogger, StepType

from experiments.browsecomp.browsecomp_eval import (
    decrypt,
    load_browsecomp_examples,
    GRADER_TEMPLATE,
)
from experiments.browsecomp.browsecomp_agent import BrowseCompAgent
from experiments.browsecomp.web_env import WebEnvironment
from experiments.browsecomp.types import SamplerBase, SamplerResponse


class LLMSampler(SamplerBase):
   
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        prompt_parts = []
        system_msg = None
        
        for msg in message_list:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                prompt_parts.append(msg["content"])
        
        prompt = "\n\n".join(prompt_parts)
        
        response_text = self.llm.generate(
            prompt,
            system_message=system_msg,
            temperature=0.0,
        )
        
        return SamplerResponse(
            response_text=response_text,
            actual_queried_message_list=message_list,
        )
    
    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": content}


class BrowseCompExperiment:
    def __init__(
        self,
        api_key: str,
        solver_model: str = "google/gemini-2.5-flash",
        grader_model: str = "google/gemini-2.5-flash",
        search_api_key: Optional[str] = None,
        max_steps: int = 15,
    ):
        self.api_key = api_key
        self.solver_model = solver_model
        self.grader_model = grader_model
        
        self.solver_llm = LLMClient(api_key=api_key, model=solver_model, temperature=0.0)
        self.grader_llm = LLMClient(api_key=api_key, model=grader_model, temperature=0.0)
        self.grader_sampler = LLMSampler(self.grader_llm)
        
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
    
    def grade_response(
        self,
        correct_answer: str,
        response: str,
    ) -> bool:      
        grader_prompt = GRADER_TEMPLATE.format(
            correct_answer=correct_answer,
            response=response,
        )
        
        sampler_response = self.grader_sampler([{"role": "user", "content": grader_prompt}])
        grading_response = sampler_response.response_text
        
        match = re.search(r"correct: (yes|no)", grading_response, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Failed to parse judge output. Expected 'correct: yes' or 'correct: no' "
                f"but got:\n{grading_response}"
            )
        
        return match.group(1).lower() == "yes"
    
    def run(
        self,
        num_examples: Optional[int] = None,
        skip_critique: bool = True,
    ) -> Dict[str, Any]:
        examples = load_browsecomp_examples(num_examples=num_examples)
        print(f"Running BrowseComp on {len(examples)} examples")
        
        run_id = self.mongo_storage.create_run(
            experiment_name="BrowseComp",
            num_problems=len(examples),
        )
        
        stats: Dict[str, int] = {
            "total": len(examples),
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "analyzed": 0,
            "repair_success_predicted": 0,
        }
        
        results: List[Dict[str, Any]] = []
        
        for idx, row in enumerate(tqdm(examples, desc="Solving BrowseComp")):
            problem_id = f"browsecomp_{idx}"
            
            try:
                question = decrypt(row.get("problem", ""), row.get("canary", ""))
                gold_answer = decrypt(row.get("answer", ""), row.get("canary", ""))
            except Exception as e:
                print(f"\nError decrypting example {idx}: {e}")
                stats["errors"] += 1
                continue
            
            print(f"\n[{idx + 1}/{len(examples)}]")
            
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
                is_correct = self.grade_response(
                    correct_answer=gold_answer,
                    response=agent_response,
                )
            except ValueError as e:
                print(f"Grading error: {e}")
                is_correct = False
                stats["errors"] += 1
            
            trace.success = is_correct
            
            result_entry = {
                "problem_id": problem_id,
                "question": question,
                "gold_answer": gold_answer,
                "agent_response": agent_response,
                "is_correct": is_correct,
                "num_steps": len(trace.steps),
            }
            
            if is_correct:
                stats["correct"] += 1
                print(f"CORRECT ({len(trace.steps)} steps)")
                
                self.mongo_storage.add_passing_trace(
                    run_id=run_id,
                    trace_data=trace.to_json(),
                    problem_id=problem_id,
                    problem_statement=question,
                    gold_answer=gold_answer,
                    final_answer=agent_response,
                )
            else:
                stats["incorrect"] += 1
                
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
                    )
                    
                    stats["analyzed"] += 1
                    
                    repair_metrics = analysis.get("metrics", {}).get("repair_metrics", {})
                    if repair_metrics.get("successful_repairs", 0) > 0:
                        stats["repair_success_predicted"] += 1
                    
                    result_entry["analysis"] = analysis
                    
                    self.mongo_storage.add_failing_trace(
                        run_id=run_id,
                        trace_data=trace.to_json(),
                        problem_id=problem_id,
                        problem_statement=question,
                        gold_answer=gold_answer,
                        final_answer=agent_response,
                        analysis_results=analysis,
                        metrics=analysis.get("metrics"),
                    )
                    
                except Exception as e:
                    print(f"CausalFlow analysis error for {problem_id}: {e}")
                    result_entry["analysis_error"] = str(e)
            
            results.append(result_entry)
        
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        try:
            self.mongo_storage.update_run_statistics(
                run_id=run_id,
                fixed=stats["repair_success_predicted"],
                analyzed=stats["analyzed"],
                accuracy=accuracy,
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
        print(f"Repair success predicted: {stats['repair_success_predicted']}")
        print(f"Cache stats: {self.web_env.get_cache_stats()}")
        
        return {
            "stats": stats,
            "accuracy": accuracy,
            "results": results,
            "run_id": run_id,
        }
    
    def _extract_web_logs(self, trace: TraceLogger) -> str:
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

        return "\n".join([log["result"] for log in web_logs])

def main():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_SECRET_KEY not found in environment")
    
    search_api_key = os.getenv("SERPER_API_KEY")
    if not search_api_key:
        raise RuntimeError("SERPER_API_KEY not found. Web search will fail unless cache is warm.")
    
    experiment = BrowseCompExperiment(
        api_key=api_key,
        solver_model="openai/gpt-5.1",
        grader_model="openai/gpt-5.2",
        search_api_key=search_api_key,
        max_steps=15,
    )
    
    results = experiment.run(
        num_examples=5,
        skip_critique=True,
    )
    
    print(f"\nFinal accuracy: {results['accuracy']:.1%}")

if __name__ == "__main__":
    main()
