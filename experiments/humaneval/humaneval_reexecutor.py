from typing import Tuple

from .docker_code_executor import DockerCodeExecutor


class HumanevalReexecutor:
    """
    Builds a runnable script from Humaneval prompt, completion, and tests,
    then executes it inside Docker to determine pass/fail.
    """

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
        completion = completion.strip()
        tests = tests.strip()

        parts = [
            prompt,
            "",
            completion,
            "",
            "if __name__ == '__main__':",
            "    " + tests.replace("\n", "\n    "),
        ]
        return "\n".join(parts)
