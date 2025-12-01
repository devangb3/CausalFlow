import re
from typing import Any, Dict, Optional, List

from llm_client import LLMClient
from trace_logger import TraceLogger
from .humaneval_reexecutor import HumanevalReexecutor


class HumanevalAgent:
    """
    Generates Python solutions for Humaneval tasks and executes them in Docker.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        reexecutor: HumanevalReexecutor,
    ):
        self.llm = llm_client
        self.reexecutor = reexecutor
        self.trace: Optional[TraceLogger] = None

    def solve(
        self,
        task_id: str,
        prompt: str,
        tests: str,
        entry_point: str,
    ) -> Dict[str, Any]:
        self.trace = TraceLogger(problem_statement=prompt)

        step_0 = self.trace.log_reasoning(
            f"Implement {entry_point} for task {task_id}. Run official tests to verify.",
            dependencies=[],
        )

        generation_prompt = f"""Complete the following Python function. Return only executable Python code with the implementation.

{prompt}

Rules:
- Preserve the provided signature and docstring.
- Include any required imports.
- No extra prints or explanations."""

        step_1 = self.trace.log_tool_call(
            "llm_code_generation",
            {"task_id": task_id, "entry_point": entry_point},
            dependencies=[step_0],
        )

        raw_response = self.llm.generate(
            generation_prompt,
            system_message="You are a precise Python expert. Respond with code only.",
            temperature=0.2,
        )
        step_2 = self.trace.log_tool_response(raw_response, dependencies=[step_1])

        completion = self._extract_code(raw_response)

        step_3 = self.trace.log_tool_call(
            "docker_code_execution",
            {"entry_point": entry_point},
            dependencies=[step_2],
        )
        success, full_code, logs = self.reexecutor.run_solution(prompt, completion, tests)
        step_4 = self.trace.log_tool_response(
            {"success": success, "logs": logs},
            dependencies=[step_3],
        )

        final_answer = "pass" if success else "fail"
        self.trace.log_final_answer(final_answer, dependencies=[step_4])
        self.trace.record_outcome(final_answer=final_answer, gold_answer="pass")

        return {
            "task_id": task_id,
            "entry_point": entry_point,
            "completion": completion,
            "full_code": full_code,
            "logs": logs,
            "success": success,
            "trace": self.trace,
            "error": None,
        }

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