import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trace_logger import TraceLogger
from llm_client import LLMClient
from math_reexecutor import MathReexecutor


class GSM8KAgent:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None
    ):

        self.llm = llm_client or LLMClient()
        self.reexecutor = MathReexecutor()
        self.trace: Optional[TraceLogger] = None

    def solve(
        self,
        question: str,
        gold_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a GSM8K problem and return the answer with trace.

        Args:
            question: The math problem to solve
            gold_answer: The correct answer (for evaluation)

        Returns:
            Dictionary with 'answer', 'trace', and 'success' keys
        """
        self.trace = TraceLogger(problem_statement=question)

        step_0 = self.trace.log_reasoning(
            f"Problem: {question}\nI need to solve this step by step.",
            dependencies=[]
        )

        breakdown_prompt = f"""Solve this math problem step by step. For each calculation, write it in the format:
STEP: <description>
OPERATION: <operation type (addition/subtraction/multiplication/division)>
EXPRESSION: <mathematical expression>

Problem: {question}

Provide your solution:"""

        solution = self.llm.generate(
            breakdown_prompt,
            system_message="You are a helpful math tutor. Solve problems step by step, showing all calculations clearly."
        )

        step_1 = self.trace.log_reasoning(
            f"Breaking down the problem:\n{solution}",
            dependencies=[step_0]
        )

        calculations = self._extract_calculations(solution)

        if not calculations:
            final_answer = self._extract_final_answer(solution)
            step_2 = self.trace.log_reasoning(
                f"Direct answer extracted: {final_answer}",
                dependencies=[step_1]
            )
            step_final = self.trace.log_final_answer(
                final_answer,
                dependencies=[step_2]
            )
        else:
            current_deps = [step_1]
            last_result = None

            for i, calc in enumerate(calculations):
                expr = calc.get('expression', '')
                desc = calc.get('description', f'Calculation {i+1}')

                step_reasoning = self.trace.log_reasoning(
                    f"Step {i+1}: {desc}\nExpression: {expr}",
                    dependencies=current_deps
                )

                step_tool_call = self.trace.log_tool_call(
                    tool_name="calculator",
                    tool_args={"expression": expr},
                    dependencies=[step_reasoning]
                )

                if self.reexecutor:
                    result = self.reexecutor.evaluate_expression(expr)
                else:
                    # Fallback: try to evaluate with Python eval (unsafe but simple)
                    try:
                        if '=' in expr:
                            rhs = expr.split('=', 1)[1].strip() # Try to extract number from RHS
                            try:
                                result = float(rhs)
                            except ValueError:
                                result = eval(rhs)
                        else:
                            result = eval(expr)
                    except:
                        result = None

                step_tool_response = self.trace.log_tool_response(
                    tool_output=result,
                    dependencies=[step_tool_call]
                )

                current_deps = [step_tool_response]
                last_result = result

            final_answer = self._extract_final_answer(solution)

            if last_result is not None:
                final_answer = str(last_result)

            step_final = self.trace.log_final_answer(
                final_answer,
                dependencies=current_deps
            )

        if gold_answer:
            gold_num = self.reexecutor.extract_number(gold_answer) if self.reexecutor else None

            if self.reexecutor:
                success = self.reexecutor.compare_answers(final_answer, gold_answer)
            else:
                success = str(final_answer).strip() == str(gold_num).strip()

            self.trace.record_outcome(
                final_answer=final_answer,
                gold_answer=str(gold_num) if gold_num is not None else gold_answer
            )
        else:
            success = None

        return {
            'answer': final_answer,
            'trace': self.trace,
            'success': success
        }

    def _extract_calculations(self, solution: str) -> List[Dict[str, str]]:

        calculations = []

        expr_pattern = r'EXPRESSION:\s*(.+?)(?:\n|$)'
        matches = re.finditer(expr_pattern, solution, re.MULTILINE)

        for match in matches:
            expr = match.group(1).strip()
            start_pos = max(0, match.start() - 200)
            context = solution[start_pos:match.start()]

            step_matches = list(re.finditer(r'STEP:\s*(.+?)(?:\n|$)', context))
            desc = step_matches[-1].group(1).strip() if step_matches else f"Calculate {expr}"

            calculations.append({
                'expression': expr,
                'description': desc
            })

        if not calculations:
            math_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)'
            matches = re.finditer(math_pattern, solution)

            for match in matches:
                expr = match.group(0).split('=')[0].strip()
                calculations.append({
                    'expression': expr,
                    'description': f"Calculate {expr}"
                })

        return calculations

    def _extract_final_answer(self, solution: str) -> str:
        if self.reexecutor:
            num = self.reexecutor.extract_number(solution)
            if num is not None:
                return str(num)

        answer_patterns = [
            r'(?:final answer|answer|result)(?:\s+is)?[:\s]+(\d+(?:\.\d+)?)',
            r'####\s*(\d+(?:\.\d+)?)',
            r'= (\d+(?:\.\d+)?)\s*$'
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        numbers = re.findall(r'\d+(?:\.\d+)?', solution)
        if numbers:
            return numbers[-1]

        return "unknown"

    def get_trace(self) -> Optional[TraceLogger]:
        return self.trace


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    agent = GSM8KAgent()

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gold_answer = "18"  # 16 - 3 - 4 = 9 eggs, 9 * $2 = $18

    print(f"Question: {question}\n")
    result = agent.solve(question, gold_answer)

    print(f"Agent's Answer: {result['answer']}")
    print(f"Correct Answer: {gold_answer}")
    print(f"Success: {result['success']}")
    print(f"\nTrace has {len(result['trace'].steps)} steps")

    result['trace'].to_json("experiments/gsm8k/gsm8k_test_trace.json")
    print("Trace saved to experiments/gsm8k/gsm8k_test_trace.json")
