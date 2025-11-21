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
        llm_client: Optional[LLMClient] = None,
        use_structured_outputs: bool = True
    ):

        if use_structured_outputs and llm_client is None:
            self.llm = LLMClient(model="google/gemini-2.5-flash")
        else:
            self.llm = llm_client or LLMClient()

        self.use_structured_outputs = use_structured_outputs
        self.reexecutor = MathReexecutor()
        self.trace: Optional[TraceLogger] = None

    def solve(
        self,
        question: str,
        gold_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.use_structured_outputs:
            return self._solve_structured(question, gold_answer)
        else:
            print("Using legacy solver")
            return self._solve_legacy(question, gold_answer)

    def _solve_structured(
        self,
        question: str,
        gold_answer: Optional[str] = None
    ) -> Dict[str, Any]:

        self.trace = TraceLogger(problem_statement=question)

        step_0 = self.trace.log_reasoning(
            f"Problem: {question}\nI need to solve this step by step.",
            dependencies=[]
        )

        breakdown_prompt = f"""Solve this math problem step by step.

Problem: {question}

Provide:
1. A brief reasoning about your approach
2. Each calculation step with:
   - A clear description
   - The operation type (addition, subtraction, multiplication, division, or other)
   - The exact mathematical expression to evaluate
3. The final numerical answer

Be precise with expressions - they should be evaluatable (e.g., "16 - 3 - 4" not "16 eggs - 3 - 4")."""

        try:
            solution = self.llm.generate_structured(
                breakdown_prompt,
                schema_name="gsm8k_solution",
                system_message="You are a helpful math tutor. Solve problems step by step with clear, evaluatable mathematical expressions.",
                model_name="google/gemini-2.5-flash"
            )

            step_1 = self.trace.log_reasoning(
                f"Overall approach: {solution.reasoning}",
                dependencies=[step_0]
            )

            current_deps = [step_1]
            last_result = None

            # Process each step
            for i, step in enumerate(solution.steps):
                desc = step.description
                expr = step.expression

                step_reasoning = self.trace.log_reasoning(
                    f"Step {i+1}: {desc}\nOperation: {step.operation}\nExpression: {expr}",
                    dependencies=current_deps
                )

                step_tool_call = self.trace.log_tool_call(
                    tool_name="calculator",
                    tool_args={"expression": expr},
                    dependencies=[step_reasoning]
                )

                # Evaluate the expression
                result = self.reexecutor.evaluate_expression(expr)

                step_tool_response = self.trace.log_tool_response(
                    tool_output=result,
                    dependencies=[step_tool_call]
                )

                current_deps = [step_tool_response]
            final_answer = solution.final_answer

            self.trace.log_final_answer(
                final_answer,
                dependencies=current_deps
            )

            if gold_answer:
                gold_num = self.reexecutor.extract_number(gold_answer)
                success = self.reexecutor.compare_answers(final_answer, gold_answer)

                self.trace.record_outcome(
                    final_answer=final_answer,
                    gold_answer=str(gold_num) if gold_num is not None else gold_answer
                )
            else:
                success = None

            return {
                'answer': final_answer,
                'trace': self.trace,
                'success': success,
                'error': None
            }

        except Exception as e:
            return {
                'answer': None,
                'trace': self.trace,
                'success': False,
                'error': str(e)
            }

    def _solve_legacy(
        self,
        question: str,
        gold_answer: Optional[str] = None
    ) -> Dict[str, Any]:

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
            'success': success,
            'error': None
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

