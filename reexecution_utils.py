from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Protocol

from trace_logger import Step, TraceLogger


class BranchRunnableAgent(Protocol):
    """Minimal interface an agent must satisfy to support branching reexecution."""

    def run_remaining_steps(self, history: List[Step]) -> TraceLogger:
        """Continue the agent loop from an injected history and return the full trace."""
        raise NotImplementedError


@dataclass
class BranchExecutionResult:
    """Outcome of a branched execution attempt."""

    success: bool
    trace: TraceLogger


class AgentBranchExecutor:
    """
    Generic branching utility to retry an agent run from an intervened step.

    The executor clones the original history up to the target step, injects a
    fixed step (e.g., corrected plan/reasoning), and resumes the agent's run
    loop from that state.
    """

    def __init__(self, agent: BranchRunnableAgent):
        if not hasattr(agent, "run_remaining_steps"):
            raise TypeError("Agent must expose run_remaining_steps(history: List[Step]) -> TraceLogger.")
        if not callable(getattr(agent, "run_remaining_steps")):
            raise TypeError("run_remaining_steps must be callable.")
        self.agent = agent

    def branch_and_reexecute(
        self,
        original_trace: TraceLogger,
        step_id_to_fix: int,
        intervened_step: Step,
    ) -> BranchExecutionResult:
        """
        Branch the trace at the specified step, inject the intervened step, and
        rerun the agent from that point.

        Returns:
            BranchExecutionResult with the new trace and whether the gold answer
            was achieved.
        """
        self._validate_inputs(original_trace, step_id_to_fix, intervened_step)
        seed_history = self._build_seed_history(original_trace, step_id_to_fix, intervened_step)

        branched_trace = self.agent.run_remaining_steps(history=seed_history)
        success = self._verify_success(branched_trace)

        return BranchExecutionResult(success=success, trace=branched_trace)

    def _validate_inputs(
        self,
        original_trace: TraceLogger,
        step_id_to_fix: int,
        intervened_step: Step,
    ) -> None:
        if step_id_to_fix < 0:
            raise ValueError("step_id_to_fix must be non-negative.")
        if step_id_to_fix >= len(original_trace.steps):
            raise ValueError(
                f"step_id_to_fix {step_id_to_fix} is out of bounds for a trace with {len(original_trace.steps)} steps."
            )
        if intervened_step is None:
            raise ValueError("intervened_step is required.")

    def _build_seed_history(
        self,
        original_trace: TraceLogger,
        step_id_to_fix: int,
        intervened_step: Step,
    ) -> List[Step]:
        """
        Clone the prefix history and inject the intervened step as the new head.
        """
        history_prefix = deepcopy(original_trace.steps[:step_id_to_fix])
        patched_step = deepcopy(intervened_step)
        patched_step.step_id = step_id_to_fix
        history_prefix.append(patched_step)
        return history_prefix

    def _verify_success(self, trace: TraceLogger) -> bool:
        """
        Return True when the branched run matches the gold answer.
        """
        if trace.gold_answer is None:
            raise ValueError("gold_answer is missing on the branched trace; cannot verify success.")
        if trace.final_answer is None:
            raise ValueError("final_answer is missing on the branched trace; cannot verify success.")

        normalized_gold = str(trace.gold_answer).strip().lower()
        normalized_final = str(trace.final_answer).strip().lower()

        success = normalized_final == normalized_gold
        trace.success = success
        return success

