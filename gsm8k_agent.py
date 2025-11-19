"""
GSM8K Agent: An agent that solves grade school math problems with trace logging.

This agent uses the CausalFlow framework to log its reasoning steps,
enabling causal analysis of failures.
"""

import re
from typing import Optional, Dict, Any, List
from trace_logger import TraceLogger
from llm_client import LLMClient
from math_reexecutor import MathReexecutor


class GSM8KAgent:
    """
    An agent that solves GSM8K math problems with detailed trace logging.

    The agent:
    1. Breaks down problems into steps
    2. Performs calculations using a calculator tool
    3. Logs all reasoning and tool calls
    4. Uses a mathematical reexecutor to verify calculations
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_reexecutor: bool = True
    ):
        """
        Initialize the GSM8K agent.

        Args:
            llm_client: LLM client for reasoning (if None, creates default)
            use_reexecutor: Whether to use the math reexecutor for verification
        """
        self.llm = llm_client or LLMClient()
        self.use_reexecutor = use_reexecutor
        self.reexecutor = MathReexecutor() if use_reexecutor else None
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
        # Initialize trace
        self.trace = TraceLogger(problem_statement=question)

        # Step 0: Initial understanding
        step_0 = self.trace.log_reasoning(
            f"Problem: {question}\nI need to solve this step by step.",
            dependencies=[]
        )

        # Step 1: Get LLM to break down the problem
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

        # Step 2: Extract and execute calculations
        calculations = self._extract_calculations(solution)

        if not calculations:
            # If no calculations found, try to extract final answer
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
            # Execute calculations step by step
            current_deps = [step_1]
            last_result = None

            for i, calc in enumerate(calculations):
                expr = calc.get('expression', '')
                desc = calc.get('description', f'Calculation {i+1}')

                # Log the reasoning for this step
                step_reasoning = self.trace.log_reasoning(
                    f"Step {i+1}: {desc}\nExpression: {expr}",
                    dependencies=current_deps
                )

                # Log tool call to calculator
                step_tool_call = self.trace.log_tool_call(
                    tool_name="calculator",
                    tool_args={"expression": expr},
                    dependencies=[step_reasoning]
                )

                # Execute the calculation
                if self.use_reexecutor and self.reexecutor:
                    result = self.reexecutor.evaluate_expression(expr)
                else:
                    # Fallback: try to evaluate with Python eval (unsafe but simple)
                    try:
                        result = eval(expr)
                    except:
                        result = None

                # Log tool response
                step_tool_response = self.trace.log_tool_response(
                    tool_output=result,
                    dependencies=[step_tool_call]
                )

                current_deps = [step_tool_response]
                last_result = result

            # Step 3: Extract final answer
            final_answer = self._extract_final_answer(solution)

            # If we have a last result, use it
            if last_result is not None:
                final_answer = str(last_result)

            step_final = self.trace.log_final_answer(
                final_answer,
                dependencies=current_deps
            )

        # Record outcome if gold answer provided
        if gold_answer:
            # Extract numerical answer from gold answer
            gold_num = self.reexecutor.extract_number(gold_answer) if self.reexecutor else None
            final_num = self.reexecutor.extract_number(final_answer) if self.reexecutor else None

            # Use reexecutor for comparison if available
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
        """
        Extract calculations from LLM solution.

        Looks for patterns like:
        EXPRESSION: 120 * 0.35
        or mathematical expressions in the text

        Args:
            solution: LLM's solution text

        Returns:
            List of calculation dictionaries
        """
        calculations = []

        # Look for EXPRESSION: pattern
        expr_pattern = r'EXPRESSION:\s*(.+?)(?:\n|$)'
        matches = re.finditer(expr_pattern, solution, re.MULTILINE)

        for match in matches:
            expr = match.group(1).strip()
            # Try to find description before expression
            start_pos = max(0, match.start() - 200)
            context = solution[start_pos:match.start()]

            step_match = re.search(r'STEP:\s*(.+?)(?:\n|$)', context)
            desc = step_match.group(1).strip() if step_match else f"Calculate {expr}"

            calculations.append({
                'expression': expr,
                'description': desc
            })

        # If no EXPRESSION patterns found, look for mathematical expressions
        if not calculations:
            # Look for expressions like "120 * 0.35 = 42"
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
        """
        Extract the final numerical answer from solution.

        Args:
            solution: LLM's solution text

        Returns:
            Final answer string
        """
        if self.reexecutor:
            # Try to extract number
            num = self.reexecutor.extract_number(solution)
            if num is not None:
                return str(num)

        # Fallback: look for common answer patterns
        answer_patterns = [
            r'(?:final answer|answer|result)(?:\s+is)?[:\s]+(\d+(?:\.\d+)?)',
            r'####\s*(\d+(?:\.\d+)?)',
            r'= (\d+(?:\.\d+)?)\s*$'
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        # Last resort: extract any number from the end
        numbers = re.findall(r'\d+(?:\.\d+)?', solution)
        if numbers:
            return numbers[-1]

        return "unknown"

    def get_trace(self) -> Optional[TraceLogger]:
        """Get the current trace."""
        return self.trace


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Test the agent
    agent = GSM8KAgent()

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gold_answer = "18"  # 16 - 3 - 4 = 9 eggs, 9 * $2 = $18

    print(f"Question: {question}\n")
    result = agent.solve(question, gold_answer)

    print(f"Agent's Answer: {result['answer']}")
    print(f"Correct Answer: {gold_answer}")
    print(f"Success: {result['success']}")
    print(f"\nTrace has {len(result['trace'].steps)} steps")

    # Save trace
    result['trace'].to_json("gsm8k_test_trace.json")
    print("Trace saved to gsm8k_test_trace.json")
