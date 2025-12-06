import re
from typing import Optional

from llm_client import LLMClient
from trace_logger import TraceLogger
from .humaneval_reexecutor import HumanevalReexecutor

class HumanevalAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        reexecutor: HumanevalReexecutor,
    ):
        self.llm = llm_client
        self.reexecutor = reexecutor

    def solve(
        self,
        task_id: str,
        prompt: str,
        tests: str,
        entry_point: str,
    ) -> TraceLogger:
        trace = TraceLogger(problem_statement=prompt, gold_answer="pass")
        resolved_entry_point = self.resolve_entry_point(entry_point, tests)

        step_0 = trace.log_reasoning(
            f"Implement {resolved_entry_point} for task {task_id}. Run official tests to verify.",
            dependencies=[],
        )

        generation_prompt = f"""Complete the following Python function. Return only executable Python code with the implementation.

{prompt}

Rules:
- The function name must be exactly `{resolved_entry_point}` so tests can call it.
- Preserve the provided signature and docstring if present; otherwise define `{resolved_entry_point}` per the prompt.
- Include any required imports.
- No extra prints or explanations."""

        step_1 = trace.log_tool_call(
            "llm_code_generation",
            {"task_id": task_id, "entry_point": resolved_entry_point},
            dependencies=[step_0],
        )

        raw_response = self.llm.generate(
            generation_prompt,
            system_message="You are a precise Python expert. Respond with code only.",
            temperature=0.2,
        )
        
        step_2 = trace.log_llm_response(raw_response, dependencies=[step_1])

        completion = self._extract_code(raw_response)

        step_3 = trace.log_tool_call(
            "docker_code_execution",
            {"entry_point": resolved_entry_point},
            dependencies=[step_2],
        )
        success, full_code, logs = self.reexecutor.run_solution(prompt, completion, tests)
        step_4 = trace.log_tool_response(
            tool_name="docker_code_execution_result",
            dependencies=[step_3],
            tool_call_result=success,
            tool_output=logs,
        )

        final_answer = "pass" if success else "fail"
        trace.log_final_answer(final_answer, dependencies=[step_4])
        trace.record_outcome(final_answer=final_answer, gold_answer="pass")

        return trace

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