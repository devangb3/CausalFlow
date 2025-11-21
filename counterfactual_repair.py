"""
CounterfactualRepair: Generates minimal edits to fix agent failures.
Generates minimal edits to causal steps that would correct failures.

Implements the minimality principle: repairs should be as small as possible
while still correcting the failure.
"""

import copy
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from trace_logger import TraceLogger, Step, StepType
from causal_attribution import CausalAttribution
from llm_client import LLMClient
from utils import extract_step_text, calculate_minimality_score


class Repair:
    def __init__(
        self,
        step_id: int,
        original_step: Step,
        repaired_step: Step,
        minimality_score: float,
        success_predicted: bool
    ):

        self.step_id = step_id #The step being repaired
        self.original_step = original_step #The original (faulty) step
        self.repaired_step = repaired_step #The repaired step
        self.minimality_score = minimality_score #Score indicating how minimal the edit is (0-1)
        self.success_predicted = success_predicted #Whether this repair is predicted to succeed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "original_step": self.original_step.to_dict(),
            "repaired_step": self.repaired_step.to_dict(),
            "minimality_score": self.minimality_score,
            "success_predicted": self.success_predicted
        }


class CounterfactualRepair:
    def __init__(
        self,
        trace: TraceLogger,
        causal_attribution: CausalAttribution,
        llm_client: LLMClient
    ):

        self.trace = trace #The failed execution trace
        self.causal_attribution = causal_attribution #Causal attribution analysis results
        self.llm_client = llm_client

        self.repairs: Dict[int, List[Repair]] = {}

    def generate_repairs(
        self,
        step_ids: List[int],
        num_proposals: int = 3
    ) -> Dict[int, List[Repair]]:

        for step_id in step_ids:
            self.repairs[step_id] = self._generate_repairs_for_step(
                step_id,
                num_proposals,
                self.trace.steps
            )

        return self.repairs

    def _generate_repairs_for_step(
        self,
        step_id: int,
        num_proposals: int,
        all_steps: List[Step]
    ) -> List[Repair]:
        original_step = self.trace.get_step(step_id)
        if not original_step:
            return []

        proposals = []
        previous_steps = all_steps[:step_id]
        for i in range(num_proposals):
            prompt = self._create_repair_prompt(original_step, previous_steps)

            try:
                result = self.llm_client.generate_structured(
                    prompt,
                    schema_name="repair",
                    system_message="You are an expert at debugging and fixing agent reasoning. Generate minimal, targeted edits. Always respond using the provided schema in JSON format.",
                    temperature=0.7
                )

                # Create repaired step
                repaired_step = self._apply_repair(original_step, result)

                minimality = calculate_minimality_score(
                    extract_step_text(original_step),
                    extract_step_text(repaired_step)
                )

                # Predict if this repair would succeed
                success_predicted = self._predict_repair_success(step_id, repaired_step)

                repair = Repair(
                    step_id=step_id,
                    original_step=original_step,
                    repaired_step=repaired_step,
                    minimality_score=minimality,
                    success_predicted=success_predicted
                )

                proposals.append(repair)

            except Exception as e:
                print(f"Error generating repair proposal {i} for step {step_id}: {e}")
                continue

        # Sort by minimality (prefer more minimal repairs)
        proposals.sort(key=lambda r: r.minimality_score, reverse=True)

        return proposals

    def _create_repair_prompt(self, step: Step, previous_steps: List[Step]) -> str:
        
        prompt = f"""You are debugging a failed agent execution. The agent's final answer was incorrect.

Problem Statement: {self.trace.problem_statement}
Correct Answer: {self.trace.gold_answer}
Agent's Incorrect Answer: {self.trace.final_answer}

Recent previous steps:
{json.dumps([step.to_dict() for step in previous_steps], indent=2)}

Below is the step that has been identified as causally responsible for the failure.

Step {step.step_id} ({step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Original Reasoning: {step.text}\n\n"
            prompt += "Provide a MINIMAL correction to this reasoning that would lead to the correct answer.\n"
            prompt += "Fill in 'repaired_text' with the corrected reasoning.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal in 'minimality_justification'."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Original Arguments: {step.tool_args}\n\n"
            prompt += "Provide corrected tool name and arguments.\n"
            prompt += "Fill in 'repaired_tool_name' and 'repaired_tool_args'.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal in 'minimality_justification'."

        elif step.step_type == StepType.MEMORY_ACCESS:
            prompt += f"Memory Key: {step.memory_key}\n"
            prompt += f"Original Value: {step.memory_value}\n\n"
            prompt += "Provide the correct memory value in 'repaired_text'.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        else:
            content = step.text or step.action or step.observation
            prompt += f"Original Content: {content}\n\n"
            prompt += "Provide a minimal correction in 'repaired_text'.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        prompt += "\n\nRemember: Change ONLY what is absolutely necessary to fix the error."

        return prompt

    def _apply_repair(self, original_step: Step, repair_result: BaseModel) -> Step:
        repaired_step = copy.deepcopy(original_step)

        if original_step.step_type == StepType.REASONING:
            repaired_step.text = repair_result.repaired_text or original_step.text

        elif original_step.step_type == StepType.TOOL_CALL:
            # Use structured tool args and name directly
            if repair_result.repaired_tool_args is not None:
                repaired_step.tool_args = repair_result.repaired_tool_args
            if repair_result.repaired_tool_name is not None:
                repaired_step.tool_name = repair_result.repaired_tool_name

        elif original_step.step_type == StepType.MEMORY_ACCESS:
            repaired_step.memory_value = repair_result.repaired_text or original_step.memory_value

        elif original_step.step_type == StepType.ENVIRONMENT_ACTION:
            repaired_step.action = repair_result.repaired_text or original_step.action

        else:
            repaired_step.text = repair_result.repaired_text or original_step.text

        return repaired_step

    def _predict_repair_success(self, step_id: int, repaired_step: Step) -> bool:

        return self.causal_attribution._llm_predict_outcome(
            step_id, repaired_step, self.trace.problem_statement
        )

    def get_best_repair(self, step_id: int) -> Optional[Repair]:
        if step_id not in self.repairs:
            return None

        successful = [r for r in self.repairs[step_id] if r.success_predicted]

        if not successful:
            return None

        return max(successful, key=lambda r: r.minimality_score)

    def get_all_best_repairs(self) -> Dict[int, Repair]:

        best_repairs = {}

        for step_id in self.repairs.keys():
            best_repair = self.get_best_repair(step_id)
            if best_repair is not None:
                best_repairs[step_id] = best_repair
        
        return best_repairs