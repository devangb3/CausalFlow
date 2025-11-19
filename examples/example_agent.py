"""
Example Agent: Demonstrates how to create an agent that uses TraceLogger.

This example shows a simple math reasoning agent that solves word problems
and logs its execution trace for CausalFlow analysis.
"""

import os
import sys
from typing import Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_logger import TraceLogger
from llm_client import LLMClient


class MathReasoningAgent:
    """
    A simple agent that solves math word problems using chain-of-thought reasoning.

    This agent demonstrates:
    - Using TraceLogger to capture execution steps
    - Tool calls (calculator)
    - Reasoning steps
    - Final answer generation
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the math reasoning agent.

        Args:
            llm_client: LLM client for reasoning
        """
        self.llm_client = llm_client
        self.trace_logger = TraceLogger()

    def solve(self, problem: str, gold_answer: Optional[str] = None) -> str:
        """
        Solve a math word problem.

        Args:
            problem: The math problem to solve
            gold_answer: Optional gold answer for evaluation

        Returns:
            The agent's final answer
        """
        # Initialize a new trace with the problem statement
        self.trace_logger = TraceLogger(problem_statement=problem)

        print(f"\nSolving: {problem}")
        print("-" * 60)

        # Step 1: Generate initial reasoning
        reasoning_prompt = f"""Solve this math problem step by step:

Problem: {problem}

Think through the problem and identify what calculations are needed.
"""

        reasoning = self.llm_client.generate(
            reasoning_prompt,
            system_message="You are a math tutor. Think step by step."
        )

        step_1 = self.trace_logger.log_reasoning(reasoning, dependencies=[])
        print(f"Step {step_1} [Reasoning]: {reasoning}")

        # Step 2: Identify calculation needed
        calc_prompt = f"""Based on this reasoning:
{reasoning}

What calculation do we need to perform? Provide it in the format:
OPERATION: <operation>
EXPRESSION: <mathematical expression>
"""

        calc_plan = self.llm_client.generate(calc_prompt)
        step_2 = self.trace_logger.log_reasoning(calc_plan, dependencies=[step_1])
        print(f"Step {step_2} [Planning]: Identified calculation")

        # Step 3: Call calculator tool
        # Extract expression from calc_plan
        expression = self._extract_expression(calc_plan)

        step_3 = self.trace_logger.log_tool_call(
            tool_name="calculator",
            tool_args={"expression": expression},
            dependencies=[step_2]
        )
        print(f"Step {step_3} [Tool Call]: calculator({expression})")

        # Step 4: Execute calculation
        result = self._calculate(expression)

        step_4 = self.trace_logger.log_tool_response(
            tool_output=result,
            dependencies=[step_3]
        )
        print(f"Step {step_4} [Tool Response]: {result}")

        # Step 5: Formulate final answer
        answer_prompt = f"""Problem: {problem}

Calculation result: {result}

Provide the final answer concisely.
"""

        final_reasoning = self.llm_client.generate(answer_prompt)
        step_5 = self.trace_logger.log_reasoning(
            final_reasoning,
            dependencies=[step_4]
        )

        # Extract the answer
        final_answer = self._extract_answer(final_reasoning)

        # Step 6: Log final answer
        step_6 = self.trace_logger.log_final_answer(
            final_answer,
            dependencies=[step_5]
        )
        print(f"Step {step_6} [Final Answer]: {final_answer}")

        # Record outcome if gold answer provided
        if gold_answer:
            self.trace_logger.record_outcome(final_answer, gold_answer)
            print(f"\nOutcome: {'SUCCESS' if self.trace_logger.success else 'FAILURE'}")

        return final_answer

    def _extract_expression(self, text: str) -> str:
        """Extract mathematical expression from text."""
        lines = text.split('\n')
        for line in lines:
            if 'EXPRESSION:' in line.upper():
                return line.split(':', 1)[1].strip()

        # Fallback: return entire text
        return text.strip()

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from text."""
        # Simple extraction: take last line or look for numbers
        import re

        # Look for "answer is X" pattern
        match = re.search(r'answer is:?\s*([0-9.,]+)', text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Look for last number in text
        numbers = re.findall(r'[0-9.,]+', text)
        if numbers:
            return numbers[-1]

        # Fallback
        return text.strip()

    def _calculate(self, expression: str) -> float:
        """
        Simple calculator tool.

        Args:
            expression: Mathematical expression

        Returns:
            Result of calculation
        """
        try:
            # WARNING: In production, use a safer math parser
            # This is just for demonstration
            result = eval(expression)
            return result
        except Exception as e:
            return f"Error: {e}"

    def get_trace(self) -> TraceLogger:
        """Get the execution trace."""
        return self.trace_logger

    def save_trace(self, filepath: str):
        """Save the trace to a file."""
        self.trace_logger.to_json(filepath)
        print(f"\nTrace saved to: {filepath}")
