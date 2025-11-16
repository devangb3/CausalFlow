"""
CausalAttribution: Identifies which steps caused agent failures through interventions.

This module implements the causal attribution via interventions component of CausalFlow
as specified in Section 5 of the research proposal.
"""

import copy
from typing import Dict, List, Any, Callable, Optional
from trace_logger import TraceLogger, Step, StepType
from causal_graph import CausalGraph
from llm_client import LLMClient


class CausalAttribution:
    """
    Performs causal attribution by intervening on steps and observing outcome changes.

    Uses the do-operator framework: for each step i, we ask "Did modifying step i
    change the final outcome from failure to success?"
    """

    def __init__(
        self,
        trace: TraceLogger,
        causal_graph: CausalGraph,
        llm_client: LLMClient,
        re_executor: Optional[Callable] = None
    ):
        """
        Initialize causal attribution analyzer.

        Args:
            trace: The failed execution trace
            causal_graph: The causal graph constructed from the trace
            llm_client: LLM client for generating interventions
            re_executor: Optional function to re-execute agent from a given step
        """
        self.trace = trace
        self.causal_graph = causal_graph
        self.llm_client = llm_client
        self.re_executor = re_executor

        # Causal Responsibility Scores for each step
        self.crs_scores: Dict[int, float] = {}

        # Store intervention results
        self.intervention_results: Dict[int, Dict[str, Any]] = {}

    def compute_causal_responsibility(
        self,
        step_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute Causal Responsibility Score (CRS) for each step.

        Args:
            step_ids: Optional list of step IDs to analyze (if None, analyzes all)

        Returns:
            Dictionary mapping step_id to CRS score (0 or 1)
        """
        if step_ids is None:
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
            # Could not generate valid intervention
            self.intervention_results[step_id] = {
                "success": False,
                "reason": "Could not generate intervention"
            }
            return 0.0

        # Simulate re-execution with intervention
        new_outcome = self._simulate_reexecution(step_id, intervened_step)

        # Store results
        self.intervention_results[step_id] = {
            "original_step": original_step.to_dict(),
            "intervened_step": intervened_step.to_dict(),
            "new_outcome": new_outcome,
            "flipped_to_success": new_outcome
        }

        # CRS = 1 if outcome flipped to success, 0 otherwise
        return 1.0 if new_outcome else 0.0

    def _generate_intervention(self, step: Step) -> Optional[Step]:
        """
        Generate an corrected version for a step.

        Args:
            step: The original step

        Returns:
            Modified step, or None if intervention cannot be generated
        """
        intervention_prompt = self._create_intervention_prompt(step)

        try:
            # Use LLM to generate corrected version
            corrected_content = self.llm_client.generate(
                intervention_prompt,
                system_message="You are an expert at debugging and correcting agent reasoning steps."
            )

            # Create new step with corrected content
            intervened_step = copy.deepcopy(step)

            if step.step_type == StepType.REASONING:
                intervened_step.text = corrected_content
            elif step.step_type == StepType.TOOL_CALL:
                # Parse corrected tool arguments
                intervened_step.tool_args = self._parse_tool_args(corrected_content)
            elif step.step_type == StepType.MEMORY_ACCESS:
                intervened_step.memory_value = corrected_content
            else:
                # For other types, update the text field
                intervened_step.text = corrected_content

            return intervened_step

        except Exception as e:
            print(f"Error generating intervention for step {step.step_id}: {e}")
            return None

    def _create_intervention_prompt(self, step: Step) -> str:

        context = self._get_step_context(step)

        prompt = f"""You are analyzing a failed agent execution. The agent produced an incorrect final answer.

Context from previous steps:
{context}

Current step (Step {step.step_id}, Type: {step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Reasoning: {step.text}\n\n"
            prompt += "Provide a corrected version of this reasoning step that would lead to the correct answer."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Arguments: {step.tool_args}\n\n"
            prompt += "Provide corrected tool arguments in JSON format."

        elif step.step_type == StepType.MEMORY_ACCESS:
            prompt += f"Memory Key: {step.memory_key}\n"
            prompt += f"Memory Value: {step.memory_value}\n\n"
            prompt += "Provide the correct memory value."

        else:
            prompt += f"Content: {step.text or step.action or step.observation}\n\n"
            prompt += "Provide a corrected version of this step."

        prompt += f"\n\nGold Answer (correct answer): {self.trace.gold_answer}"
        prompt += "\n\nProvide ONLY the corrected content without explanation."

        return prompt

    def _get_step_context(self, step: Step, max_context_steps: int = 3) -> str:
        """
        Get context from previous steps.

        Args:
            step: The step to get context for
            max_context_steps: Maximum number of previous steps to include

        Returns:
            Context string
        """
        dependencies = step.dependencies
        if not dependencies:
            return "No previous dependencies."

        context_lines = []
        for dep_id in dependencies[-max_context_steps:]:
            dep_step = self.trace.get_step(dep_id)
            if dep_step:
                summary = self._summarize_step(dep_step)
                context_lines.append(f"Step {dep_id}: {summary}")

        return "\n".join(context_lines)

    def _summarize_step(self, step: Step) -> str:
        """
        Create a brief summary of a step.

        Args:
            step: The step to summarize

        Returns:
            Summary string
        """
        if step.step_type == StepType.REASONING:
            text = step.text[:1000] + "..." if len(step.text) > 1000 else step.text
            return f"[Reasoning] {text}"
        elif step.step_type == StepType.TOOL_CALL:
            return f"[Tool Call] {step.tool_name}({step.tool_args})"
        elif step.step_type == StepType.TOOL_RESPONSE:
            output = str(step.tool_output)
            return f"[Tool Response] {output}"
        elif step.step_type == StepType.MEMORY_ACCESS:
            return f"[Memory] {step.memory_key} = {step.memory_value}"
        elif step.step_type == StepType.ENVIRONMENT_ACTION:
            return f"[Action] {step.action}"
        elif step.step_type == StepType.ENVIRONMENT_OBSERVATION:
            return f"[Observation] {step.observation}"
        elif step.step_type == StepType.FINAL_ANSWER:
            return f"[Final Answer] {step.text}"
        else:
            return f"[{step.step_type.value}]"

    def _parse_tool_args(self, text: str) -> Dict[str, Any]:
        """
        Parse tool arguments from LLM response.

        Args:
            text: Text containing tool arguments

        Returns:
            Parsed arguments dictionary
        """
        import json
        import re

        # Try to find JSON in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return as single argument
        return {"value": text.strip()}

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
        return self._llm_predict_outcome(step_id, intervened_step)

    def _llm_predict_outcome(self, step_id: int, intervened_step: Step) -> bool:
        """
        Use LLM to predict if the intervention would lead to success.

        Args:
            step_id: The step that was intervened on
            intervened_step: The modified step

        Returns:
            True if predicted to succeed, False otherwise
        """
        prompt = f"""You are analyzing an agent execution trace.

Original Final Answer: {self.trace.final_answer}
Correct Answer: {self.trace.gold_answer}
Original Outcome: FAILED

An intervention was made at Step {step_id}:
Original: {self._summarize_step(self.trace.get_step(step_id))}
Intervened: {self._summarize_step(intervened_step)}

Descendants of this step (affected by the intervention):
"""
        descendants = self.causal_graph.get_descendants(step_id)
        for desc_id in sorted(descendants):
            desc_step = self.trace.get_step(desc_id)
            if desc_step:
                prompt += f"  Step {desc_id}: {self._summarize_step(desc_step)}\n"

        prompt += f"\nWould this intervention cause the final answer to change to the correct answer ({self.trace.gold_answer})?"
        prompt += "\n\nRespond with ONLY 'YES' or 'NO'."

        try:
            response = self.llm_client.generate(prompt, temperature=0.0)
            return "YES" in response.upper()
        except Exception:
            return False

    def get_causal_steps(self, threshold: float = 0.5) -> List[int]:
        """
        Get step IDs that are causally responsible for failure.

        Args:
            threshold: Minimum CRS score to be considered causal

        Returns:
            List of step IDs with CRS >= threshold
        """
        return [
            step_id for step_id, score in self.crs_scores.items()
            if score >= threshold
        ]

    def get_top_causal_steps(self, n: int = 3) -> List[tuple]:
        """
        Get top N causal steps by CRS score.

        Args:
            n: Number of top steps to return

        Returns:
            List of (step_id, crs_score) tuples, sorted by score
        """
        sorted_steps = sorted(
            self.crs_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_steps[:n]

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

        top_steps = self.get_top_causal_steps(5)
        lines.append("Top Causal Steps (by CRS):")
        lines.append("-" * 60)

        for step_id, crs in top_steps:
            if crs > 0:
                step = self.trace.get_step(step_id)
                lines.append(f"\nStep {step_id} (CRS = {crs:.2f}):")
                lines.append(f"  Type: {step.step_type.value}")
                lines.append(f"  Summary: {self._summarize_step(step)}")

                if step_id in self.intervention_results:
                    result = self.intervention_results[step_id]
                    if "intervened_step" in result:
                        step_data = result["intervened_step"].copy()
                        step_data["step_type"] = StepType(step_data["step_type"])
                        intervened = Step(**step_data)
                        lines.append(f"  Intervention: {self._summarize_step(intervened)}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        num_causal = len(self.get_causal_steps())
        return f"CausalAttribution(steps_analyzed={len(self.crs_scores)}, causal_steps={num_causal})"
