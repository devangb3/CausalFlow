import re
from typing import Dict, List, Optional, Sequence, TypedDict

from datasets import load_dataset


class MBPPTask(TypedDict):
    task_id: str
    prompt: str
    tests: str
    entry_point: str


class MBPPDataLoader:
    def __init__(self, dataset_name: str = "mbpp", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split

    def load_data(self, num_rows: int = 50) -> List[MBPPTask]:
        dataset = load_dataset(self.dataset_name)[self.split]
        tasks: List[MBPPTask] = []

        for row in dataset:
            prompt = self._extract_prompt(row)
            tests = self._build_tests(row)
            entry_point = self._infer_entry_point(row, tests)
            task_id = self._extract_task_id(row)

            tasks.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "tests": tests,
                    "entry_point": entry_point,
                }
            )

            if len(tasks) >= num_rows:
                break

        return tasks

    def _extract_prompt(self, row: Dict[str, object]) -> str:
        for key in ("prompt", "text", "question"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError("Row missing prompt/text/question")

    def _extract_task_id(self, row: Dict[str, object]) -> str:
        if "task_id" in row:
            value = row["task_id"]
            if isinstance(value, str):
                return value
            if isinstance(value, int):
                return f"mbpp-{value}"
        raise ValueError("Row missing task_id")

    def _get_code(self, row: Dict[str, object]) -> str:
        for key in ("code", "solution", "code_string"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _build_tests(self, row: Dict[str, object]) -> str:
        tests_field = row.get("test_list")
        if isinstance(tests_field, Sequence) and not isinstance(tests_field, str):
            lines: List[str] = []
            for test in tests_field:
                if not isinstance(test, str):
                    raise ValueError("test_list entries must be strings")
                stripped = test.strip()
                if not stripped:
                    raise ValueError("Empty test case encountered")
                lines.append(self._ensure_assert(stripped))
            return "\n".join(lines)

        test_str = row.get("test")
        if isinstance(test_str, str) and test_str.strip():
            return test_str.strip()

        raise ValueError("Row missing test_list or test field")

    def _ensure_assert(self, test_line: str) -> str:
        if test_line.startswith("assert"):
            return test_line
        return f"assert {test_line}"

    def _infer_entry_point(self, row: Dict[str, object], tests: str) -> str:
        code_text = self._get_code(row)
        if code_text:
            name = self._first_function_name(code_text)
            if name:
                return name

        test_name = self._first_function_name_from_tests(tests)
        if test_name:
            return test_name

        raise ValueError("Unable to infer entry point from code or tests")

    def _first_function_name(self, code_text: str) -> Optional[str]:
        # Search for function definitions: "def function_name(" pattern
        match = re.search(r"def\s+([a-zA-Z_][\w]*)\s*\(", code_text)
        if match:
            return match.group(1)
        return None

    def _first_function_name_from_tests(self, tests: str) -> Optional[str]:
        # Find all function calls: "function_name(" pattern to extract function names
        candidates = re.findall(r"([a-zA-Z_][\w]*)\s*\(", tests)
        filtered = [name for name in candidates if name not in {"assert", "print"}]
        if filtered:
            return filtered[0]
        return None

