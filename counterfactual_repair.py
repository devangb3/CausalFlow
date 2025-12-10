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
        llm_client: LLMClient,
        reexecutor: Optional[Any] = None,
        execution_context: Optional[Dict[str, Any]] = None
    ):

        self.trace = trace #The failed execution trace
        self.causal_attribution = causal_attribution #Causal attribution analysis results
        self.llm_client = llm_client
        self.reexecutor = reexecutor #Optional reexecutor for deterministic repair evaluation
        self.execution_context = execution_context or {} #Context needed for execution (prompt, tests, entry_point, etc.)

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
                    system_message="""You are an expert at debugging and fixing agent failures. 
                                    Analyze the execution error logs carefully and generate MINIMAL, TARGETED edits that directly address the specific error.
                                    Make the smallest possible change that fixes the issue.
                                    Always respond using the provided schema in JSON format.
                                    """,
                    temperature=0.7
                )

                repaired_step = self._apply_repair(original_step, result)

                minimality = calculate_minimality_score(
                    extract_step_text(original_step),
                    extract_step_text(repaired_step)
                )

                # Evaluate repair success deterministically if reexecutor is available, otherwise predict
                success_predicted = self._evaluate_repair_success(step_id, repaired_step, execution_context=self.execution_context)

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

        proposals.sort(key=lambda r: r.minimality_score, reverse=True)

        return proposals

    def _create_repair_prompt(self, step: Step, previous_steps: List[Step]) -> str:
        execution_logs = self._extract_execution_logs()
        
        prompt = f"""You are debugging a failed agent execution. The agent's final answer was incorrect.

Problem Statement: {self.trace.problem_statement}
Correct Answer: {self.trace.gold_answer}
Agent's Incorrect Answer: {self.trace.final_answer}
"""

        if execution_logs:
            prompt += f"""
EXECUTION ERROR LOGS (what went wrong):
{execution_logs}

Use these error logs to understand the specific failure and create a targeted fix.
"""

        prompt += f"""
Recent previous steps:
{json.dumps([step.to_dict() for step in previous_steps], indent=2)}

Below is the step that has been identified as causally responsible for the failure.

Step {step.step_id} ({step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Original Reasoning: {step.text}\n\n"
            prompt += "Based on the execution error logs above, provide a MINIMAL, TARGETED correction to this reasoning.\n"
            prompt += "The correction should directly address the specific error shown in the logs.\n"
            prompt += "Fill in 'repaired_text' with the corrected reasoning.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal and targeted in 'minimality_justification'."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Original Arguments: {step.tool_args}\n\n"
            prompt += "Based on the execution error logs, provide corrected tool name and arguments that address the failure.\n"
            prompt += "Fill in 'repaired_tool_name' and 'repaired_tool_args'.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal and targeted in 'minimality_justification'."

        elif step.step_type == StepType.LLM_RESPONSE:
            prompt += f"Original LLM Response (code): {step.text}\n\n"
            prompt += "Based on the execution error logs, provide a MINIMAL, TARGETED code fix.\n"
            prompt += "The fix should directly address the specific error shown in the logs (e.g., fix the exact line causing the error).\n"
            prompt += "Fill in 'repaired_text' with the corrected code.\n"
            prompt += "List the specific changes in 'changes_made'.\n"
            prompt += "Explain why this is minimal and targeted in 'minimality_justification'."

        elif step.step_type == StepType.MEMORY_ACCESS:
            prompt += f"Memory Key: {step.memory_key}\n"
            prompt += f"Original Value: {step.memory_value}\n\n"
            prompt += "Based on the execution error logs, provide the correct memory value.\n"
            prompt += "Fill in 'repaired_text' with the corrected value.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        elif step.step_type == StepType.TOOL_RESPONSE:
            prompt += f"Tool Output: {step.tool_output}\n"
            prompt += f"Tool Call Result: {step.tool_call_result}\n\n"
            prompt += "Based on the execution error logs, provide a corrected tool response.\n"
            prompt += "Fill in 'repaired_text' with the corrected output.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        else:
            content = step.text or step.action or step.observation
            prompt += f"Original Content: {content}\n\n"
            prompt += "Based on the execution error logs, provide a minimal, targeted correction.\n"
            prompt += "Fill in 'repaired_text'.\n"
            prompt += "List changes in 'changes_made'.\n"
            prompt += "Explain minimality in 'minimality_justification'."

        prompt += "\n\nCRITICAL: Make MINIMAL, TARGETED edits that directly address the specific error shown in the execution logs. Change ONLY what is necessary to fix the error."

        return prompt

    def _extract_execution_logs(self) -> str:
        """Extract execution error logs from the trace."""
        logs_parts: List[str] = []
        
        # First, try to get logs from execution_context
        if self.execution_context and "logs" in self.execution_context:
            logs = self.execution_context.get("logs")
            if logs and isinstance(logs, str) and logs.strip():
                logs_parts.append(logs.strip())
        
        # Also extract from TOOL_RESPONSE steps in the trace
        for step in self.trace.steps:
            if step.step_type == StepType.TOOL_RESPONSE:
                if step.tool_output and isinstance(step.tool_output, str):
                    output = step.tool_output.strip()
                    if output and output not in logs_parts:
                        logs_parts.append(output)
        
        if not logs_parts:
            return "No execution logs available."
        
        return "\n".join(logs_parts)

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

        elif original_step.step_type == StepType.LLM_RESPONSE:
            if repair_result.repaired_text is not None:
                repaired_step.text = repair_result.repaired_text

        elif original_step.step_type == StepType.MEMORY_ACCESS:
            repaired_step.memory_value = repair_result.repaired_text or original_step.memory_value

        elif original_step.step_type == StepType.ENVIRONMENT_ACTION:
            repaired_step.action = repair_result.repaired_text or original_step.action

        elif original_step.step_type == StepType.TOOL_RESPONSE:
            # For tool responses, repair goes into tool_output
            if repair_result.repaired_text is not None:
                repaired_step.tool_output = repair_result.repaired_text
            # Keep original tool_output if no repair provided

        else:
            repaired_step.text = repair_result.repaired_text or original_step.text

        return repaired_step

    def _evaluate_repair_success(self, step_id: int, repaired_step: Step, execution_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Evaluate repair success by re-executing with the repaired step.

        If an agent is provided, build a history up to and including the repaired
        step, then call agent.run_remaining_steps() to continue execution and
        observe the actual outcome.
        """
        if self.reexecutor is None:
            return self.causal_attribution._llm_predict_outcome(
                step_id, repaired_step, self.trace.problem_statement
            )

        # Build history: all steps before step_id + the repaired step
        history = [copy.deepcopy(step) for step in self.trace.steps if step.step_id < step_id]
        history.append(repaired_step)

        new_trace = self.reexecutor.run_remaining_steps(history)
        return new_trace.success

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