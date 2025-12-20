import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trace_logger import TraceLogger
from llm_client import LLMClient
from math_reexecutor import MathReexecutor


class GSM8KAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "google/gemini-2.5-flash",
    ):
        self.llm = llm_client
        self.model = model
        self.reexecutor = MathReexecutor()
        self.trace: Optional[TraceLogger] = None

    def solve(
        self,
        question: str,
        gold_answer: str,
    ) -> TraceLogger:

        self.trace = TraceLogger(problem_statement=question, gold_answer=gold_answer)

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

        solution = self.llm.generate_structured(
            breakdown_prompt,
            schema_name="gsm8k_solution",
            system_message="You are a helpful math tutor. Solve problems step by step with clear, evaluatable mathematical expressions.",
            model_name=self.model
        )

        # Log the LLM response (structured solution) so CausalFlow can intervene on it
        solution_json = json.dumps({
            "reasoning": solution.reasoning,
            "steps": [
                {
                    "description": step.description,
                    "operation": step.operation,
                    "expression": step.expression
                }
                for step in solution.steps
            ],
            "final_answer": solution.final_answer
        }, indent=2)

        llm_response_step = self.trace.log_llm_response(
            llm_response=solution_json,
            dependencies=[step_0]
        )

        step_1 = self.trace.log_reasoning(
            f"Overall approach: {solution.reasoning}",
            dependencies=[llm_response_step]
        )

        current_deps = [step_1]

        # Process each calculation step
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

            # Evaluate the expression deterministically
            result = self.reexecutor.evaluate_expression(expr)
            tool_call_success = result is not None

            step_tool_response = self.trace.log_tool_response(
                tool_name="calculator",
                dependencies=[step_tool_call],
                tool_call_result=tool_call_success,
                tool_output=str(result) if result is not None else "Error: Could not evaluate expression"
            )

            current_deps = [step_tool_response]

        final_answer = solution.final_answer

        self.trace.log_final_answer(
            final_answer,
            dependencies=current_deps
        )

        # Grade the answer using MathReexecutor for numeric comparison
        success = self.reexecutor.compare_answers(final_answer, gold_answer)

        self.trace.record_outcome(
            final_answer=final_answer,
            gold_answer=gold_answer
        )
        self.trace.success = success

        return self.trace
