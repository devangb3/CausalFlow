"""
CounterfactualRepair: Generates minimal edits to fix agent failures.

This module implements the counterfactual repair component of CausalFlow
as specified in Section 6 of the research proposal.
"""

import copy
from typing import Dict, List, Any, Optional, Tuple
from trace_logger import TraceLogger, Step, StepType
from causal_attribution import CausalAttribution
from llm_client import LLMClient


class Repair:
    """Represents a proposed repair to a step."""

    def __init__(
        self,
        step_id: int,
        original_step: Step,
        repaired_step: Step,
        minimality_score: float,
        success_predicted: bool
    ):
        """
        Initialize a repair.

        Args:
            step_id: The step being repaired
            original_step: The original (faulty) step
            repaired_step: The repaired step
            minimality_score: Score indicating how minimal the edit is (0-1)
            success_predicted: Whether this repair is predicted to succeed
        """
        self.step_id = step_id
        self.original_step = original_step
        self.repaired_step = repaired_step
        self.minimality_score = minimality_score
        self.success_predicted = success_predicted

    def to_dict(self) -> Dict[str, Any]:
        """Convert repair to dictionary."""
        return {
            "step_id": self.step_id,
            "original_step": self.original_step.to_dict(),
            "repaired_step": self.repaired_step.to_dict(),
            "minimality_score": self.minimality_score,
            "success_predicted": self.success_predicted
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"Repair(step_id={self.step_id}, "
                f"minimality={self.minimality_score:.2f}, "
                f"success={self.success_predicted})")


class CounterfactualRepair:
    """
    Generates minimal edits to causal steps that would correct failures.

    Implements the minimality principle: repairs should be as small as possible
    while still correcting the failure.
    """

    def __init__(
        self,
        trace: TraceLogger,
        causal_attribution: CausalAttribution,
        llm_client: LLMClient
    ):
        """
        Initialize counterfactual repair generator.

        Args:
            trace: The failed execution trace
            causal_attribution: Causal attribution analysis results
            llm_client: LLM client for generating repairs
        """
        self.trace = trace
        self.causal_attribution = causal_attribution
        self.llm_client = llm_client

        self.repairs: Dict[int, List[Repair]] = {}

    def generate_repairs(
        self,
        step_ids: Optional[List[int]] = None,
        num_proposals: int = 3
    ) -> Dict[int, List[Repair]]:
        """
        Generate repairs for causally responsible steps.

        Args:
            step_ids: Optional list of step IDs to repair (if None, uses causal steps)
            num_proposals: Number of repair proposals to generate per step

        Returns:
            Dictionary mapping step_id to list of repair proposals
        """
        if step_ids is None:
            # Get causal steps from attribution
            step_ids = self.causal_attribution.get_causal_steps()

        for step_id in step_ids:
            self.repairs[step_id] = self._generate_repairs_for_step(
                step_id,
                num_proposals
            )

        return self.repairs

    def _generate_repairs_for_step(
        self,
        step_id: int,
        num_proposals: int
    ) -> List[Repair]:
        """
        Generate multiple repair proposals for a single step.

        Args:
            step_id: The step to repair
            num_proposals: Number of proposals to generate

        Returns:
            List of repair proposals
        """
        original_step = self.trace.get_step(step_id)
        if not original_step:
            return []

        proposals = []

        for i in range(num_proposals):
            prompt = self._create_repair_prompt(original_step, proposal_num=i)

            try:
                repaired_content = self.llm_client.generate(
                    prompt,
                    system_message="You are an expert at debugging and fixing agent reasoning. Generate minimal, targeted edits.",
                    temperature=0.7  # Some variety in proposals
                )

                # Create repaired step
                repaired_step = self._apply_repair(original_step, repaired_content)

                # Calculate minimality score
                minimality = self._calculate_minimality(original_step, repaired_step)

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

    def _create_repair_prompt(self, step: Step, proposal_num: int = 0) -> str:
        """
        Create a prompt for generating a repair.

        Args:
            step: The step to repair
            proposal_num: The proposal number (for variety)

        Returns:
            Prompt string
        """
        prompt = f"""You are debugging a failed agent execution. The agent's final answer was incorrect.

This step has been identified as causally responsible for the failure.

Step {step.step_id} ({step.step_type.value}):
"""

        if step.step_type == StepType.REASONING:
            prompt += f"Original Reasoning: {step.text}\n\n"
            prompt += "Provide a MINIMAL correction to this reasoning that would lead to the correct answer.\n"
            prompt += "Change only what is necessary."

        elif step.step_type == StepType.TOOL_CALL:
            prompt += f"Tool: {step.tool_name}\n"
            prompt += f"Original Arguments: {step.tool_args}\n\n"
            prompt += "Provide corrected tool arguments (JSON format).\n"
            prompt += "Change only the incorrect arguments."

        elif step.step_type == StepType.MEMORY_ACCESS:
            prompt += f"Memory Key: {step.memory_key}\n"
            prompt += f"Original Value: {step.memory_value}\n\n"
            prompt += "Provide the correct memory value."

        else:
            content = step.text or step.action or step.observation
            prompt += f"Original Content: {content}\n\n"
            prompt += "Provide a minimal correction."

        prompt += f"\n\nCorrect Answer: {self.trace.gold_answer}"
        prompt += f"\nAgent's Incorrect Answer: {self.trace.final_answer}"

        if proposal_num > 0:
            prompt += f"\n\nThis is proposal #{proposal_num + 1}. Consider alternative minimal fixes."

        prompt += "\n\nProvide ONLY the corrected content, nothing else."

        return prompt

    def _apply_repair(self, original_step: Step, repaired_content: str) -> Step:
        """
        Apply a repair to create a modified step.

        Args:
            original_step: The original step
            repaired_content: The repaired content from LLM

        Returns:
            Modified step
        """
        repaired_step = copy.deepcopy(original_step)

        if original_step.step_type == StepType.REASONING:
            repaired_step.text = repaired_content.strip()

        elif original_step.step_type == StepType.TOOL_CALL:
            # Parse tool arguments
            import json
            import re

            json_match = re.search(r'\{.*\}', repaired_content, re.DOTALL)
            if json_match:
                try:
                    repaired_step.tool_args = json.loads(json_match.group())
                except json.JSONDecodeError:
                    repaired_step.tool_args = {"value": repaired_content.strip()}
            else:
                repaired_step.tool_args = {"value": repaired_content.strip()}

        elif original_step.step_type == StepType.MEMORY_ACCESS:
            repaired_step.memory_value = repaired_content.strip()

        elif original_step.step_type == StepType.ENVIRONMENT_ACTION:
            repaired_step.action = repaired_content.strip()

        else:
            repaired_step.text = repaired_content.strip()

        return repaired_step

    def _calculate_minimality(self, original_step: Step, repaired_step: Step) -> float:
        """
        Calculate minimality score: how small is the edit?

        Formula: MS = 1 - (tokens_changed / tokens_original)

        Args:
            original_step: Original step
            repaired_step: Repaired step

        Returns:
            Minimality score (0-1, higher is more minimal)
        """
        # Extract text content from both steps
        original_text = self._extract_step_text(original_step)
        repaired_text = self._extract_step_text(repaired_step)

        if not original_text:
            return 0.0

        # Simple token-based calculation
        original_tokens = original_text.split()
        repaired_tokens = repaired_text.split()

        # Count different tokens
        tokens_changed = sum(
            1 for o, r in zip(original_tokens, repaired_tokens) if o != r
        )
        tokens_changed += abs(len(original_tokens) - len(repaired_tokens))

        if len(original_tokens) == 0:
            return 0.0

        minimality = 1.0 - (tokens_changed / len(original_tokens))
        return max(0.0, min(1.0, minimality))  # Clamp to [0, 1]

    def _extract_step_text(self, step: Step) -> str:
        """
        Extract text content from a step for comparison.

        Args:
            step: The step

        Returns:
            Text content
        """
        if step.step_type == StepType.REASONING:
            return step.text or ""
        elif step.step_type == StepType.TOOL_CALL:
            return str(step.tool_args)
        elif step.step_type == StepType.MEMORY_ACCESS:
            return str(step.memory_value)
        elif step.step_type == StepType.ENVIRONMENT_ACTION:
            return step.action or ""
        elif step.step_type == StepType.ENVIRONMENT_OBSERVATION:
            return step.observation or ""
        else:
            return step.text or ""

    def _predict_repair_success(self, step_id: int, repaired_step: Step) -> bool:
        """
        Predict if a repair would lead to success.

        Args:
            step_id: The step being repaired
            repaired_step: The proposed repair

        Returns:
            True if predicted to succeed
        """
        # Use the same prediction method as causal attribution
        return self.causal_attribution._llm_predict_outcome(step_id, repaired_step)

    def get_best_repair(self, step_id: int) -> Optional[Repair]:
        """
        Get the best (most minimal, successful) repair for a step.

        Args:
            step_id: The step ID

        Returns:
            Best repair, or None if no successful repairs exist
        """
        if step_id not in self.repairs:
            return None

        # Filter to successful repairs
        successful = [r for r in self.repairs[step_id] if r.success_predicted]

        if not successful:
            # If no successful repairs, return most minimal one
            return self.repairs[step_id][0] if self.repairs[step_id] else None

        # Return most minimal successful repair
        return max(successful, key=lambda r: r.minimality_score)

    def get_all_best_repairs(self) -> Dict[int, Repair]:
        """
        Get the best repair for each step.

        Returns:
            Dictionary mapping step_id to best repair
        """
        return {
            step_id: self.get_best_repair(step_id)
            for step_id in self.repairs.keys()
            if self.get_best_repair(step_id) is not None
        }

    def generate_report(self) -> str:
        """
        Generate a human-readable report of repairs.

        Returns:
            Report string
        """
        lines = ["=" * 60]
        lines.append("COUNTERFACTUAL REPAIR REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Steps with Repairs: {len(self.repairs)}")
        lines.append("")

        best_repairs = self.get_all_best_repairs()
        lines.append(f"Steps with Successful Repairs: {len(best_repairs)}")
        lines.append("")

        for step_id in sorted(self.repairs.keys()):
            step = self.trace.get_step(step_id)
            lines.append(f"\nStep {step_id} ({step.step_type.value}):")
            lines.append("-" * 60)

            original_text = self._extract_step_text(step)
            lines.append(f"Original: {original_text[:100]}...")

            best = self.get_best_repair(step_id)
            if best:
                repaired_text = self._extract_step_text(best.repaired_step)
                lines.append(f"Best Repair: {repaired_text[:100]}...")
                lines.append(f"Minimality: {best.minimality_score:.2f}")
                lines.append(f"Success Predicted: {best.success_predicted}")

            num_proposals = len(self.repairs[step_id])
            num_successful = sum(1 for r in self.repairs[step_id] if r.success_predicted)
            lines.append(f"Total Proposals: {num_proposals} ({num_successful} successful)")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        total_repairs = sum(len(repairs) for repairs in self.repairs.values())
        return f"CounterfactualRepair(steps={len(self.repairs)}, total_proposals={total_repairs})"
