import copy
import json
import re
from typing import Dict, List, Any, Optional, Set
from trace_logger import TraceLogger, Step, StepType
from causal_graph import CausalGraph
from llm_client import LLMClient
from text_processor import convert_text_to_jsonl
from utils import summarize_step


class CausalAttribution:

    def __init__(
        self,
        trace: TraceLogger,
        causal_graph: CausalGraph,
        llm_client: LLMClient,
        re_executor: Optional[Any] = None
    ):

        self.trace = trace #The failed execution trace
        self.causal_graph = causal_graph #The causal graph constructed from the trace
        self.llm_client = llm_client #LLM client for generating interventions
        self.re_executor = re_executor #Function to re-execute agent from a given step

        self.crs_scores: Dict[int, float] = {} #Causal Responsibility Scores for each step
        self.intervention_results: Dict[int, Dict[str, Any]] = {} #Store intervention results

    def compute_causal_responsibility(
        self,
        execution_context: Optional[Dict[str, Any]] = None,
        intervene_step_types: Optional[Set[StepType]] = None,
    ) -> Dict[int, float]:
        steps_to_check = self.trace.steps[:-1]  # Exclude the final step
        
        for step in steps_to_check:
            if intervene_step_types is not None and step.step_type not in intervene_step_types:
                self.crs_scores[step.step_id] = 0.0
                self.intervention_results[step.step_id] = {
                    "success": False,
                    "reason": f"Skipped: step type {step.step_type.value} not in intervene_step_types"
                }
                continue
            
            self.crs_scores[step.step_id] = self._intervene_on_step(step, execution_context=execution_context)

        return self.crs_scores

    def _intervene_on_step(self, step: Step, execution_context: Optional[Dict[str, Any]] = None) -> float:
        """
        Perform intervention on a single step and compute CRS.

        Process:
        1. Copy the trace
        2. Apply intervention to step i
        3. Re-execute from step i forward
        4. Compare new outcome to original

        Args:
            step_id: The step to intervene on

        Returns:
            CRS score (1.0 if intervention flipped outcome to success, 0.0 otherwise)
        """
        end_feedback = "The execution in this run failed. The following logs were generated: " + execution_context.get("logs") or "No end feedback provided"
        intervened_step = self._generate_intervention(step, end_feedback=end_feedback)

        if intervened_step is None:
            self.intervention_results[step.step_id] = {
                "success": False,
                "reason": "Could not generate intervention"
            }
            return 0.0
        new_outcome = self._reexecute(step.step_id, intervened_step)

        self.intervention_results[step.step_id] = {
            "original_step": step.to_dict(),
            "intervened_step": intervened_step.to_dict(),
            "new_outcome": new_outcome,
            "flipped_to_success": new_outcome
        }

        return 1.0 if new_outcome else 0.0 # CRS = 1 if outcome flipped to success, 0 otherwise

    def _generate_intervention(self, step: Step, end_feedback: str) -> Optional[Step]:
        intervention_prompt = self._create_intervention_prompt(step, end_feedback)

        try:
            result = self.llm_client.generate_structured(
                intervention_prompt,
                schema_name="intervention",
                system_message="You are an expert at debugging and correcting agent reasoning steps. Always respond using the provided schema in JSON format.",
                model_name="google/gemini-3-flash-preview"
            )

            intervened_step = copy.deepcopy(step)

            if step.step_type == StepType.REASONING:
                intervened_step.text = result.corrected_reasoning or step.text
            elif step.step_type == StepType.TOOL_CALL:
                # Parse tool args from JSON string
                if result.corrected_tool_args_json is not None:
                    parsed = convert_text_to_jsonl(result.corrected_tool_args_json)
                    if not parsed:
                        raise ValueError(f"Failed to parse corrected_tool_args_json: {result.corrected_tool_args_json}")
                    intervened_step.tool_args = parsed[0]
                if result.corrected_tool_name is not None:
                    intervened_step.tool_name = result.corrected_tool_name
            elif step.step_type == StepType.MEMORY_ACCESS:
                intervened_step.memory_value = result.corrected_reasoning or step.memory_value
            else:
                # For other types, update the text field
                intervened_step.text = result.corrected_reasoning or result.corrected_text or step.text

            return intervened_step

        except Exception as e:
            print(f"Error generating intervention for step {step.step_id}: {e}")
            return None

    def _create_intervention_prompt(self, step: Step, end_feedback: str) -> str:
        context = self._get_step_context(step)

        prompt = f"""You are analyzing a failed agent execution. The agent produced an incorrect final answer.

Problem Statement: {self.trace.problem_statement or "No problem statement provided"}
Gold Answer (correct answer): {self.trace.gold_answer or "No gold answer provided"}
Environment feedback at the point of failure: {end_feedback}

Context from previous steps:
{context if context else "No earlier context"}

Current step (Step {step.step_id}, Type: {step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Original Reasoning: {step.text}\n\n"
            prompt += "Provide a corrected version of this reasoning step that would lead to the correct answer.\n"
            prompt += "CRITICAL: Compare the reasoning against the Problem Statement. If the logic contradicts the problem statement (e.g. ignoring a constraint like 'restart from beginning'), you MUST correct it.\n"
            prompt += "Fill in the 'corrected_reasoning' field."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Arguments: {json.dumps(step.tool_args)}\n\n"
            prompt += "Provide corrected tool name and arguments. Fill in 'corrected_tool_name' and 'corrected_tool_args_json' (as a valid JSON string, e.g. '{\"key\": \"value\"}')."

        elif step.step_type == StepType.LLM_RESPONSE:
            prompt += f"LLM Response: {step.text}\n\n"
            prompt += "Provide a corrected version of this LLM response in the 'corrected_reasoning' field."

        elif step.step_type == StepType.TOOL_RESPONSE:
            prompt += f"Tool Call Result: {step.tool_call_result}\n"
            prompt += f"Tool Output: {step.tool_output}\n\n"
            prompt += "Provide a corrected version of this tool response in the 'corrected_reasoning' field."

        elif step.step_type == StepType.MEMORY_ACCESS:
            prompt += f"Memory Key: {step.memory_key}\n"
            prompt += f"Memory Value: {step.memory_value}\n\n"
            prompt += "Provide the correct memory value in the 'corrected_reasoning' field."

        else:
            prompt += f"Content: {step.text or step.action or step.observation}\n\n"
            prompt += "Provide a corrected version of this step in the 'corrected_reasoning' field."

        prompt += "\n\nAlso provide a brief explanation of what was corrected and why in the 'explanation' field."

        return prompt

    def _get_step_context(self, step: Step, max_context_steps: int = 3) -> str:
        dependencies = step.dependencies
        if not dependencies:
            return "No previous dependencies."

        context_lines = []
        for dep_id in dependencies[-max_context_steps:]:
            dep_step = self.trace.get_step(dep_id)
            if dep_step:
                summary = summarize_step(dep_step)
                context_lines.append(f"Step {dep_id}: {summary}")

        return "\n".join(context_lines)

    def _reexecute(self, step_id: int, intervened_step: Step) -> bool:
        if self.re_executor is None:
            return self._llm_predict_outcome(step_id, intervened_step, self.trace.problem_statement)

        history = [copy.deepcopy(step) for step in self.trace.steps if step.step_id < step_id]
        history.append(intervened_step)

        new_trace = self.re_executor.run_remaining_steps(history)
        return new_trace.success

    def _llm_predict_outcome(self, step_id: int, intervened_step: Step, problem_statement: str) -> bool:
        prompt = f"""You are analyzing an agent execution trace.

Problem Statement: {problem_statement}
Original Final Answer: {self.trace.final_answer}
Correct Answer: {self.trace.gold_answer}
Original Outcome: FAILED

An intervention was made at Step {step_id}:
Original: {summarize_step(self.trace.get_step(step_id))}
Intervened: {summarize_step(intervened_step)}

Descendants of this step (affected by the intervention):
"""
        descendants = self.causal_graph.get_descendants(step_id)
        for desc_id in sorted(descendants):
            desc_step = self.trace.get_step(desc_id)
            if desc_step:
                prompt += f"  Step {desc_id}: {summarize_step(desc_step)}\n"

        prompt += f"\nWould this intervention cause the final answer to change to the correct answer ({self.trace.gold_answer})?"
        prompt += "\nIMPORTANT: Be conservative. If the intervention is trivial (e.g. only formatting changes, removing units) or does not address the root logical error, answer FALSE."
        prompt += "\n\nProvide your prediction, confidence level, and reasoning."

        try:
            result = self.llm_client.generate_structured(
                prompt,
                schema_name="outcome_prediction",
                temperature=0.0,
                model_name="openai/gpt-5.2"
            )
            return result.would_succeed
        except Exception as e:
            print(f"Error predicting outcome: {e}")
            return False

    def get_causal_steps(self) -> List[int]:
        return [
            step_id for step_id, score in self.crs_scores.items()
            if score >= 0.5
        ]

    def get_top_causal_steps(self) -> List[tuple]:

        return [
            (step_id, score) for step_id, score in self.crs_scores.items()
            if score >= 0.5
        ]