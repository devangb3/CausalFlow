import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from llm_client import LLMClient
from trace_logger import Step, StepType, TraceLogger
from .humaneval_reexecutor import HumanevalReexecutor

@dataclass
class AgentExecutionContext:
    task_id: str
    prompt: str
    tests: str
    entry_point: str
    resolved_entry_point: str

class HumanevalAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        reexecutor: HumanevalReexecutor,
    ):
        self.llm = llm_client
        self.reexecutor = reexecutor
        self._current_context: Optional[AgentExecutionContext] = None

    def solve(
        self,
        task_id: str,
        prompt: str,
        tests: str,
        entry_point: str,
    ) -> TraceLogger:
        resolved_entry_point = self.resolve_entry_point(entry_point, tests)
        context = AgentExecutionContext(
            task_id=task_id,
            prompt=prompt,
            tests=tests,
            entry_point=entry_point,
            resolved_entry_point=resolved_entry_point,
        )
        self._current_context = context

        trace = self._build_trace_with_history(history=None, context=context)
        return self._run(trace=trace, context=context, include_reasoning=True)

    def run_remaining_steps(self, history: List[Step]) -> TraceLogger:
        if not history:
            raise ValueError("History cannot be empty for run_remaining_steps")

        context = self._ensure_context()
        trace = self._build_trace_with_history(history=history, context=context)
        last_step = history[-1]

        # Determine what to execute based on the last step type
        if last_step.step_type == StepType.REASONING:
            # Need: LLM call -> LLM response -> docker call -> docker response -> final
            return self._continue_from_reasoning(trace, context)

        elif last_step.step_type == StepType.TOOL_CALL:
            if last_step.tool_name == "llm_code_generation":
                # Need: LLM response -> docker call -> docker response -> final
                return self._continue_from_llm_tool_call(trace, context)
            elif last_step.tool_name == "docker_code_execution":
                # Need: docker response -> final
                return self._continue_from_docker_tool_call(trace, context)
            else:
                raise ValueError(f"Unknown tool_call type: {last_step.tool_name}")

        elif last_step.step_type == StepType.LLM_RESPONSE:
            # Have the code, need: docker call -> docker response -> final
            return self._continue_from_llm_response(trace, context, last_step)

        elif last_step.step_type == StepType.TOOL_RESPONSE:
            # Have docker result, need: final answer
            return self._continue_from_tool_response(trace, last_step)

        else:
            raise ValueError(f"Cannot continue from step type: {last_step.step_type}")

    def resolve_entry_point(self, entry_point: str, tests: str) -> str:
        normalized = entry_point.strip() if entry_point else ""
        if normalized and normalized != "__init__":
            return normalized

        inferred = self._first_function_name_from_tests(tests)
        if inferred:
            return inferred

        raise ValueError("Unable to infer entry point from tests")

    def _extract_code(self, response: str) -> str:
        code_blocks = re.findall(r"```python(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        if code_blocks:
            return code_blocks[-1].strip()

        fallback_blocks = re.findall(r"```(.*?)```", response, flags=re.DOTALL)
        if fallback_blocks:
            return fallback_blocks[-1].strip()

        return response.strip()
    def cleanup_tests(self, test: str) -> str:
        test = test.strip()
        check_pattern = r"def check\(candidate\):"
        match = re.search(check_pattern, test)
        if match:
            test = test[match.start():]

        test = re.sub(r"\n\s*METADATA\s*=\s*\{[^}]*\}", "", test, flags=re.DOTALL)
        return test

    def _first_function_name_from_tests(self, tests: str) -> Optional[str]:
        candidates = re.findall(r"([a-zA-Z_][\w]*)\s*\(", tests)
        for name in candidates:
            if name not in {"assert", "print", "check"}:
                return name
        return None

    def _run(
        self,
        trace: TraceLogger,
        context: AgentExecutionContext,
        include_reasoning: bool,
    ) -> TraceLogger:
        dependencies: List[int] = []

        if include_reasoning:
            reasoning_step = trace.log_reasoning(
                f"Implement {context.resolved_entry_point} for task {context.task_id}. Run official tests to verify.",
                dependencies=[],
            )
            dependencies = [reasoning_step]
        elif trace.steps:
            dependencies = [trace.steps[-1].step_id]

        generation_prompt = self._build_generation_prompt(context, trace)

        step_tool_call = trace.log_tool_call(
            "llm_code_generation",
            {"task_id": context.task_id, "entry_point": context.resolved_entry_point},
            dependencies=dependencies,
        )

        raw_response = self.llm.generate(
            generation_prompt,
            system_message="You are a precise Python expert. Respond with code only.",
            temperature=0.2,
        )

        step_llm_response = trace.log_llm_response(raw_response, dependencies=[step_tool_call])
        completion = self._extract_code(raw_response)

        step_exec_call = trace.log_tool_call(
            "docker_code_execution",
            {"entry_point": context.resolved_entry_point},
            dependencies=[step_llm_response],
        )

        success, full_code, logs = self.reexecutor.run_solution(context.prompt, completion, context.tests)
        step_exec_response = trace.log_tool_response(
            tool_name="docker_code_execution_result",
            dependencies=[step_exec_call],
            tool_call_result=success,
            tool_output=logs,
        )

        final_answer = "pass" if success else "fail"
        trace.log_final_answer(final_answer, dependencies=[step_exec_response])
        trace.record_outcome(final_answer=final_answer, gold_answer="pass")

        return trace

    def _build_trace_with_history(
        self,
        history: Optional[List[Step]],
        context: AgentExecutionContext,
    ) -> TraceLogger:
        trace = TraceLogger(problem_statement=context.prompt, gold_answer="pass")

        if history is None:
            return trace

        if not history:
            return trace

        validated_history = self._validate_history(history)
        trace.steps = validated_history
        trace.current_step_id = len(validated_history)
        trace.gold_answer = "pass"
        trace.problem_statement = context.prompt
        return trace

    def _ensure_context(self) -> AgentExecutionContext:
        if self._current_context is None:
            raise RuntimeError("No execution context available. Run solve(...) before branching.")
        return self._current_context

    def _validate_history(self, history: List[Step]) -> List[Step]:
        cloned_history = deepcopy(history)
        for idx, step in enumerate(cloned_history):
            if step.step_id != idx:
                raise ValueError(
                    f"History step_id sequence is non-contiguous at index {idx}: found {step.step_id}."
                )
            if step.step_type == StepType.FINAL_ANSWER:
                raise ValueError("History already contains a FINAL_ANSWER step; cannot resume execution.")
        return cloned_history

    def _continue_from_reasoning(self, trace: TraceLogger, context: AgentExecutionContext) -> TraceLogger:
        """Continue after a REASONING step: generate code, execute, finalize."""
        last_step_id = trace.steps[-1].step_id

        generation_prompt = self._build_generation_prompt(context, trace)
        step_tool_call = trace.log_tool_call(
            "llm_code_generation",
            {"task_id": context.task_id, "entry_point": context.resolved_entry_point},
            dependencies=[last_step_id],
        )

        raw_response = self.llm.generate(
            generation_prompt,
            system_message="You are a precise Python expert. Respond with code only.",
            temperature=0.2,
        )

        step_llm_response = trace.log_llm_response(raw_response, dependencies=[step_tool_call])

        return self._execute_and_finalize(trace, context, raw_response, step_llm_response)

    def _continue_from_llm_tool_call(self, trace: TraceLogger, context: AgentExecutionContext) -> TraceLogger:
        """Continue after TOOL_CALL(llm_code_generation): call LLM, execute, finalize."""
        last_step_id = trace.steps[-1].step_id

        generation_prompt = self._build_generation_prompt(context, trace)
        raw_response = self.llm.generate(
            generation_prompt,
            system_message="You are a precise Python expert. Respond with code only.",
            temperature=0.2,
        )

        step_llm_response = trace.log_llm_response(raw_response, dependencies=[last_step_id])
        completion = self._extract_code(raw_response)

        return self._execute_and_finalize(trace, context, completion, step_llm_response)

    def _continue_from_llm_response(self, trace: TraceLogger, context: AgentExecutionContext, llm_step: Step) -> TraceLogger:
        """Continue after LLM_RESPONSE: extract code, execute in docker, finalize."""
        completion = self._extract_code(llm_step.text or "")
        return self._execute_and_finalize(trace, context, completion, llm_step.step_id)

    def _continue_from_docker_tool_call(self, trace: TraceLogger, context: AgentExecutionContext) -> TraceLogger:
        """Continue after TOOL_CALL(docker_code_execution): find code, run docker, finalize."""
        # Find the LLM_RESPONSE step to get the code
        llm_response_step = None
        for step in reversed(trace.steps):
            if step.step_type == StepType.LLM_RESPONSE:
                llm_response_step = step
                break

        if llm_response_step is None:
            raise ValueError("No LLM_RESPONSE step found in trace history")

        completion = self._extract_code(llm_response_step.text or "")
        last_step_id = trace.steps[-1].step_id

        success, full_code, logs = self.reexecutor.run_solution(context.prompt, completion, context.tests)
        step_exec_response = trace.log_tool_response(
            tool_name="docker_code_execution_result",
            dependencies=[last_step_id],
            tool_call_result=success,
            tool_output=logs,
        )

        final_answer = "pass" if success else "fail"
        trace.log_final_answer(final_answer, dependencies=[step_exec_response])
        trace.record_outcome(final_answer=final_answer, gold_answer="pass")
        return trace

    def _continue_from_tool_response(self, trace: TraceLogger, tool_response_step: Step) -> TraceLogger:
        """Continue after TOOL_RESPONSE: just log final answer based on result."""
        success = tool_response_step.tool_call_result or False
        final_answer = "pass" if success else "fail"
        trace.log_final_answer(final_answer, dependencies=[tool_response_step.step_id])
        trace.record_outcome(final_answer=final_answer, gold_answer="pass")
        return trace

    def _execute_and_finalize(
        self,
        trace: TraceLogger,
        context: AgentExecutionContext,
        completion: str,
        dependency_step_id: int,
    ) -> TraceLogger:
        """Execute code in docker and log final answer."""
        step_exec_call = trace.log_tool_call(
            "docker_code_execution",
            {"entry_point": context.resolved_entry_point},
            dependencies=[dependency_step_id],
        )

        success, full_code, logs = self.reexecutor.run_solution(context.prompt, completion, context.tests)
        step_exec_response = trace.log_tool_response(
            tool_name="docker_code_execution_result",
            dependencies=[step_exec_call],
            tool_call_result=success,
            tool_output=logs,
        )

        final_answer = "pass" if success else "fail"
        trace.log_final_answer(final_answer, dependencies=[step_exec_response])
        trace.record_outcome(final_answer=final_answer, gold_answer="pass")
        return trace

    def _build_generation_prompt(self, context: AgentExecutionContext, trace: TraceLogger) -> str:
        reasoning_context = self._extract_reasoning_from_history(trace)

        prompt_parts = [
            "Complete the following Python function. Return only executable Python code with the implementation.",
            "",
            context.prompt,
            "",
        ]

        if reasoning_context:
            prompt_parts.extend([
                "Additional guidance from prior reasoning:",
                reasoning_context,
                "",
            ])

        prompt_parts.extend([
            "Rules:",
            f"- The function name must be exactly `{context.resolved_entry_point}` so tests can call it.",
            f"- Preserve the provided signature and docstring if present; otherwise define `{context.resolved_entry_point}` per the prompt.",
            "- Include any required imports.",
            "- No extra prints or explanations.",
        ])

        return "\n".join(prompt_parts)

    def _extract_reasoning_from_history(self, trace: TraceLogger) -> str:
        """Extract reasoning steps from trace history to include in generation prompt."""
        reasoning_parts: List[str] = []
        for step in trace.steps:
            if step.step_type == StepType.REASONING and step.text:
                reasoning_parts.append(step.text)
        return "\n".join(reasoning_parts)