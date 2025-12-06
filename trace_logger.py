import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    LLM_RESPONSE = "llm_response"
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
    tool_call_result: Optional[bool] = None
    memory_key: Optional[str] = None
    memory_value: Optional[Any] = None
    action: Optional[str] = None
    observation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "dependencies": self.dependencies
        }

        # Add non-None optional fields
        for field_name in ["text", "tool_name", "tool_args", "tool_output", "tool_call_result",
                          "memory_key", "memory_value", "action", "observation"]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        return result


class TraceLogger:
    def __init__(self, problem_statement: Optional[str] = None, gold_answer: Optional[str] = None):
        self.steps: List[Step] = []
        self.current_step_id: int = 0
        self.success: Optional[bool] = None
        self.final_answer: Optional[str] = None
        self.gold_answer: Optional[str] = gold_answer
        self.problem_statement: Optional[str] = problem_statement

    def log_reasoning(self, text: str, dependencies: List[int] = None) -> int:
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
                      dependencies: List[int] = None, logs: Optional[str] = None) -> int:
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

    def log_tool_response(self, tool_name: str, dependencies: List[int], tool_call_result: bool, tool_output: Optional[str] = None) -> int:
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.TOOL_RESPONSE,
            tool_name=tool_name,
            tool_call_result=tool_call_result,
            tool_output=tool_output,
            dependencies=dependencies,
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id
        
    def log_llm_response(self, llm_response: str, dependencies: List[int]) -> int:
        step = Step(
            step_id=self.current_step_id,
            step_type=StepType.LLM_RESPONSE,
            text=llm_response,
            dependencies=dependencies
        )
        self.steps.append(step)
        self.current_step_id += 1
        return step.step_id

    def log_memory_access(self, memory_key: str, memory_value: Any,
                          dependencies: List[int] = None) -> int:
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

        self.final_answer = final_answer
        self.gold_answer = gold_answer
        self.success = self._compare_answers(final_answer, gold_answer)

    def _compare_answers(self, final_answer: str, gold_answer: str) -> bool:
        # Simple exact match (can be enhanced with fuzzy matching)
        return str(final_answer).strip().lower() == str(gold_answer).strip().lower()

    def get_step(self, step_id: int) -> Optional[Step]:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "success": self.success,
            "final_answer": self.final_answer,
            "gold_answer": self.gold_answer,
            "problem_statement": self.problem_statement,
            "num_steps": len(self.steps)
        }

    def to_json(self, filepath: str = None) -> str:
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceLogger':
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
                tool_call_result=step_data.get("tool_call_result"),
                memory_key=step_data.get("memory_key"),
                memory_value=step_data.get("memory_value"),
                action=step_data.get("action"),
                observation=step_data.get("observation"),
            )
            logger.steps.append(step)

        logger.current_step_id = len(logger.steps)
        logger.success = data.get("success")
        logger.final_answer = data.get("final_answer")
        logger.gold_answer = data.get("gold_answer")
        logger.problem_statement = data.get("problem_statement")

        return logger

    @classmethod
    def from_json(cls, filepath: str) -> 'TraceLogger':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
