from typing import Tuple
import re
from .docker_code_executor import DockerCodeExecutor

class HumanevalReexecutor:

    def __init__(
        self,
        executor: DockerCodeExecutor,
    ):
        self.executor = executor

    def run_solution(
        self,
        prompt: str,
        completion: str,
        tests: str,
    ) -> Tuple[bool, str, str]:
        """
        Returns (success, full_code, logs).
        """
        full_code = self._build_script(prompt, completion, tests)
        success, logs = self.executor.run_code(full_code)
        return success, full_code, logs

    def _build_script(self, prompt: str, completion: str, tests: str) -> str:
        prompt = prompt.rstrip()
        completion = self._extract_code(completion)
        tests = tests.strip()

        parts = [
            completion,
            "",
            "if __name__ == '__main__':",
            "    " + tests.replace("\n", "\n    "),
        ]
        return "\n".join(parts)
    
    def _extract_code(self, response: str) -> str:
        code_blocks = re.findall(r"```python(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        if code_blocks:
            return code_blocks[-1].strip()

        fallback_blocks = re.findall(r"```(.*?)```", response, flags=re.DOTALL)
        if fallback_blocks:
            return fallback_blocks[-1].strip()

        return response.strip()