"""
CausalAttribution: Identifies which steps caused agent failures through interventions.
Performs causal attribution by intervening on steps and observing outcome changes.

    Uses the do-operator framework: for each step i, we ask "Did modifying step i
    change the final outcome from failure to success?"
"""

import copy
from typing import Dict, List, Any, Callable, Optional
from trace_logger import TraceLogger, Step, StepType
from causal_graph import CausalGraph
from llm_client import LLMClient
from utils import summarize_step, format_step_context


class CausalAttribution:

    def __init__(
        self,
        trace: TraceLogger,
        causal_graph: CausalGraph,
        llm_client: LLMClient,
        re_executor: Optional[Callable] = None
    ):

        self.trace = trace #The failed execution trace
        self.causal_graph = causal_graph #The causal graph constructed from the trace
        self.llm_client = llm_client #LLM client for generating interventions
        self.re_executor = re_executor #Optional function to re-execute agent from a given step

        self.crs_scores: Dict[int, float] = {} #Causal Responsibility Scores for each step
        self.intervention_results: Dict[int, Dict[str, Any]] = {} #Store intervention results

    def compute_causal_responsibility(
        self
    ) -> Dict[int, float]:
       
        step_ids = [step.step_id for step in self.trace.steps]

        for step_id in step_ids:
            self.crs_scores[step_id] = self._intervene_on_step(step_id)

        return self.crs_scores

    def _intervene_on_step(self, step_id: int) -> float:
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
        original_step = self.trace.get_step(step_id)
        if not original_step:
            return 0.0

        # Generate intervention for this step
        intervened_step = self._generate_intervention(original_step)

        if intervened_step is None:
            self.intervention_results[step_id] = {
                "success": False,
                "reason": "Could not generate intervention"
            }
            return 0.0

        # Simulate re-execution with intervention
        new_outcome = self._simulate_reexecution(step_id, intervened_step)

        self.intervention_results[step_id] = {
            "original_step": original_step.to_dict(),
            "intervened_step": intervened_step.to_dict(),
            "new_outcome": new_outcome,
            "flipped_to_success": new_outcome
        }

        # CRS = 1 if outcome flipped to success, 0 otherwise
        return 1.0 if new_outcome else 0.0

    def _generate_intervention(self, step: Step) -> Optional[Step]:
        intervention_prompt = self._create_intervention_prompt(step)

        try:
            # Use structured output to generate corrected version
            result = self.llm_client.generate_structured(
                intervention_prompt,
                schema_name="intervention",
                system_message="You are an expert at debugging and correcting agent reasoning steps."
            )

            # Create new step with corrected content
            intervened_step = copy.deepcopy(step)

            if step.step_type == StepType.REASONING:
                intervened_step.text = result.corrected_reasoning or step.text
            elif step.step_type == StepType.TOOL_CALL:
                # Use structured tool args directly
                if result.corrected_tool_args is not None:
                    intervened_step.tool_args = result.corrected_tool_args
                if result.corrected_tool_name is not None:
                    intervened_step.tool_name = result.corrected_tool_name
            elif step.step_type == StepType.MEMORY_ACCESS:
                intervened_step.memory_value = result.corrected_reasoning or step.memory_value
            else:
                # For other types, update the text field
                intervened_step.text = result.corrected_reasoning or step.text

            return intervened_step

        except Exception as e:
            print(f"Error generating intervention for step {step.step_id}: {e}")
            return None

    def _create_intervention_prompt(self, step: Step) -> str:
        context = self._get_step_context(step)

        prompt = f"""You are analyzing a failed agent execution. The agent produced an incorrect final answer.

Problem Statement: {self.trace.problem_statement}
Gold Answer (correct answer): {self.trace.gold_answer}

Context from previous steps:
{context}

Current step (Step {step.step_id}, Type: {step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Original Reasoning: {step.text}\n\n"
            prompt += "Provide a corrected version of this reasoning step that would lead to the correct answer. Fill in the 'corrected_reasoning' field."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Arguments: {step.tool_args}\n\n"
            prompt += "Provide corrected tool name and arguments. Fill in 'corrected_tool_name' and 'corrected_tool_args' fields."

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

    def _simulate_reexecution(self, step_id: int, intervened_step: Step) -> bool:
        """
        Simulate re-execution of the trace with an intervened step.

        In a full implementation, this would actually re-run the agent.
        For now, we use heuristics or LLM to predict the outcome.

        Args:
            step_id: The step that was intervened on
            intervened_step: The modified step

        Returns:
            True if predicted to succeed, False otherwise
        """
        # If a re-executor function is provided, use it
        if self.re_executor:
            try:
                return self.re_executor(step_id, intervened_step)
            except Exception as e:
                print(f"Re-execution failed: {e}")
                return False

        # Otherwise, use LLM to predict outcome
        return self._llm_predict_outcome(step_id, intervened_step, self.trace.problem_statement)

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
        prompt += "\n\nProvide your prediction, confidence level, and reasoning."

        try:
            result = self.llm_client.generate_structured(
                prompt,
                schema_name="outcome_prediction",
                temperature=0.0
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

    def generate_report(self) -> str:

        lines = ["=" * 60]
        lines.append("CAUSAL ATTRIBUTION REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Steps Analyzed: {len(self.crs_scores)}")
        lines.append(f"Original Outcome: {'SUCCESS' if self.trace.success else 'FAILURE'}")
        lines.append(f"Final Answer: {self.trace.final_answer}")
        lines.append(f"Gold Answer: {self.trace.gold_answer}")
        lines.append("")

        causal_steps = self.get_causal_steps()
        lines.append(f"Causally Responsible Steps: {len(causal_steps)}")
        lines.append("")

        top_steps = self.get_top_causal_steps()
        lines.append("Top Causal Steps (by CRS):")
        lines.append("-" * 60)

        for step_id, crs in top_steps:
            if crs > 0:
                step = self.trace.get_step(step_id)
                lines.append(f"\nStep {step_id} (CRS = {crs:.2f}):")
                lines.append(f"  Type: {step.step_type.value}")
                lines.append(f"  Summary: {summarize_step(step)}")

                if step_id in self.intervention_results:
                    result = self.intervention_results[step_id]
                    if "intervened_step" in result:
                        step_data = result["intervened_step"].copy()
                        step_data["step_type"] = StepType(step_data["step_type"])
                        intervened = Step(**step_data)
                        lines.append(f"  Intervention: {summarize_step(intervened)}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
