"""
TraceLogger: Captures every internal step of an agent's decision-making process.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class StepType(Enum):
    """Types of steps that can occur during agent execution."""
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    MEMORY_ACCESS = "memory_access"
    ENVIRONMENT_ACTION = "environment_action"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class Step:
    step_id: int
    step_type: StepType
    dependencies: List[int] = field(default_factory=list)

    # Optional fields depending on step type
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    memory_key: Optional[str] = None
    memory_value: Optional[Any] = None
    action: Optional[str] = None
    observation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        result = {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "dependencies": self.dependencies
        }

        # Add non-None optional fields
        for field_name in ["text", "tool_name", "tool_args", "tool_output",
                          "memory_key", "memory_value", "action", "observation"]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        return result


class TraceLogger:
    """
    Logs every step of an agent's execution to enable causal analysis.

    The trace logger captures:
    - Reasoning steps (chain-of-thought, planning)
    - Tool calls and responses
    - Memory accesses
    - Environment actions and observations
    - Final answers
    - Dependencies between steps
    """

    def __init__(self):
        """Initialize an empty trace logger."""
        self.steps: List[Step] = []
        self.current_step_id: int = 0
        self.success: Optional[bool] = None
        self.final_answer: Optional[str] = None
        self.gold_answer: Optional[str] = None

    def log_reasoning(self, text: str, dependencies: List[int] = None) -> int:
        """
        Log a reasoning step (chain-of-thought, planning, deliberation).

        Args:
            text: The reasoning text generated
            dependencies: List of step IDs this reasoning depends on

        Returns:
            The step ID of this reasoning step
        """
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.REASONING,
            text=text,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_tool_call(self, tool_name: str, tool_args: Dict[str, Any],
                      dependencies: List[int] = None) -> int:
        """
        Log a tool invocation.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            dependencies: List of step IDs this tool call depends on

        Returns:
            The step ID of this tool call
        """
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.TOOL_CALL,
            tool_name=tool_name,
            tool_args=tool_args,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_tool_response(self, tool_output: Any, dependencies: List[int]) -> int:
        """
        Log a tool's response.

        Args:
            tool_output: The output returned by the tool
            dependencies: List of step IDs (typically the tool call)

        Returns:
            The step ID of this tool response
        """
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.TOOL_RESPONSE,
            tool_output=tool_output,
            dependencies=dependencies
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_memory_access(self, memory_key: str, memory_value: Any,
                          dependencies: List[int] = None) -> int:
        """
        Log a memory retrieval operation.

        Args:
            memory_key: The key used to access memory
            memory_value: The value retrieved from memory
            dependencies: List of step IDs this memory access depends on

        Returns:
            The step ID of this memory access
        """
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.MEMORY_ACCESS,
            memory_key=memory_key,
            memory_value=memory_value,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_environment_action(self, action: str, dependencies: List[int] = None) -> int:
        """
        Log an action taken in an environment (e.g., ALFWorld).

        Args:
            action: The action taken (e.g., "open fridge")
            dependencies: List of step IDs this action depends on

        Returns:
            The step ID of this action
        """
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.ENVIRONMENT_ACTION,
            action=action,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_environment_observation(self, observation: str,
                                   dependencies: List[int]) -> int:
        """
        Log an observation from the environment.

        Args:
            observation: The environment's response
            dependencies: List of step IDs (typically the action)

        Returns:
            The step ID of this observation
        """
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.ENVIRONMENT_OBSERVATION,
            observation=observation,
            dependencies=dependencies
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_final_answer(self, answer: str, dependencies: List[int] = None) -> int:
        """
        Log the agent's final answer.

        Args:
            answer: The final answer provided by the agent
            dependencies: List of step IDs that led to this answer

        Returns:
            The step ID of the final answer
        """
        self.final_answer = answer
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.FINAL_ANSWER,
            text=answer,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def record_outcome(self, final_answer: str, gold_answer: str):
        """
        Record the final outcome of the agent's execution.

        Args:
            final_answer: The agent's final answer
            gold_answer: The correct/expected answer
        """
        self.final_answer = final_answer
        self.gold_answer = gold_answer
        self.success = self._compare_answers(final_answer, gold_answer)

    def _compare_answers(self, final_answer: str, gold_answer: str) -> bool:
        """
        Compare the final answer with the gold answer.

        Args:
            final_answer: The agent's answer
            gold_answer: The correct answer

        Returns:
            True if answers match, False otherwise
        """
        # Simple exact match (can be enhanced with fuzzy matching)
        return str(final_answer).strip().lower() == str(gold_answer).strip().lower()

    def get_step(self, step_id: int) -> Optional[Step]:
        """
        Retrieve a step by its ID.

        Args:
            step_id: The ID of the step to retrieve

        Returns:
            The step if found, None otherwise
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire trace to a dictionary.

        Returns:
            Dictionary representation of the trace
        """
        return {
            "steps": [step.to_dict() for step in self.steps],
            "success": self.success,
            "final_answer": self.final_answer,
            "gold_answer": self.gold_answer,
            "num_steps": len(self.steps)
        }

    def to_json(self, filepath: str = None) -> str:
        """
        Serialize the trace to JSON.

        Args:
            filepath: Optional path to save the JSON file

        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceLogger':
        """
        Reconstruct a TraceLogger from a dictionary.

        Args:
            data: Dictionary representation of a trace

        Returns:
            Reconstructed TraceLogger instance
        """
        logger = cls()

        for step_data in data.get("steps", []):
            step = Step(
                step_id=step_data["step_id"],
                step_type=StepType(step_data["step_type"]),
                dependencies=step_data.get("dependencies", []),
                text=step_data.get("text"),
                tool_name=step_data.get("tool_name"),
                tool_args=step_data.get("tool_args"),
                tool_output=step_data.get("tool_output"),
                memory_key=step_data.get("memory_key"),
                memory_value=step_data.get("memory_value"),
                action=step_data.get("action"),
                observation=step_data.get("observation")
            )
            logger.steps.append(step)

        logger.current_step_id = len(logger.steps)
        logger.success = data.get("success")
        logger.final_answer = data.get("final_answer")
        logger.gold_answer = data.get("gold_answer")

        return logger

    @classmethod
    def from_json(cls, filepath: str) -> 'TraceLogger':
        """
        Load a TraceLogger from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Reconstructed TraceLogger instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation of the trace logger."""
        return (f"TraceLogger(steps={len(self.steps)}, "
                f"success={self.success}, "
                f"final_answer='{self.final_answer}')")
