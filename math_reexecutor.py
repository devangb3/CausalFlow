import re
from typing import Optional, Union, Tuple
import ast
import operator
class MathReexecutor:
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

    def extract_number(self, text: str) -> Optional[float]:
        if not text:
            return None

        text = str(text).strip()
        gsm8k_pattern = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
        match = re.search(gsm8k_pattern, text)
        if match:
            number_str = match.group(1).replace(',', '')
            try:
                return float(number_str)
            except ValueError:
                pass

        # Try to extract number from end of string (common in answers)
        # Match: optional dollar sign, optional minus, digits with optional commas and decimal
        patterns = [
            r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?',  # General number
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)',  # Dollar amounts
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the last number found (usually the answer)
                number_str = matches[-1] if not isinstance(matches[-1], tuple) else matches[-1][0]
                number_str = number_str.replace(',', '').replace('$', '').strip()
                try:
                    return float(number_str)
                except ValueError:
                    continue

        # Try converting the whole string
        cleaned = text.replace(',', '').replace('$', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            pass

        return None

    def evaluate_expression(self, expression: str) -> Optional[float]:
        try:
            expression = expression.strip()
            
            if '=' in expression:
                parts = expression.split('=', 1)  # Split on '=' and get the right-hand side
                rhs = parts[1].strip()
                
                rhs_number = self.extract_number(rhs)
                if rhs_number is not None:
                    return float(rhs_number)
                
                try:
                    tree = ast.parse(rhs, mode='eval')
                    result = self._eval_node(tree.body)
                    return float(result)
                except Exception:
                    pass
                
                lhs = parts[0].strip()
                try:
                    tree = ast.parse(lhs, mode='eval')
                    result = self._eval_node(tree.body)
                    return float(result)
                except Exception:
                    pass
                
                return None
            
            tree = ast.parse(expression, mode='eval')

            result = self._eval_node(tree.body)
            return float(result)
        except Exception:
            return None

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            if type(node.op) not in self.operators:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand>
            if type(node.op) not in self.operators:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            operand = self._eval_node(node.operand)
            return self.operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def compare_answers(
        self,
        answer1: Union[str, float],
        answer2: Union[str, float]
    ) -> bool:
        # Extract numbers from both answers
        num1 = self.extract_number(str(answer1)) if isinstance(answer1, str) else answer1
        num2 = self.extract_number(str(answer2)) if isinstance(answer2, str) else answer2

        if num1 is None or num2 is None:
            # Fallback to string comparison
            return str(answer1).strip().lower() == str(answer2).strip().lower()

        # Compare with tolerance
        return abs(num1 - num2) <= self.tolerance

    def verify_calculation(
        self,
        expression: str,
        expected_result: Union[str, float]
    ) -> Tuple[bool, Optional[float]]:
        """
        Verify if a calculation expression produces the expected result.

        Args:
            expression: Mathematical expression to evaluate
            expected_result: Expected result (can be string or number)

        Returns:
            Tuple of (is_correct, computed_value)
        """
        computed = self.evaluate_expression(expression)

        if computed is None:
            return False, None

        expected_num = (
            self.extract_number(str(expected_result))
            if isinstance(expected_result, str)
            else expected_result
        )

        if expected_num is None:
            return False, computed

        is_correct = abs(computed - expected_num) <= self.tolerance
        return is_correct, computed


if __name__ == "__main__":
    executor = MathReexecutor()

    print("Testing number extraction:")
    test_cases = [
        "The answer is 42",
        "#### 1234",
        "$1,234.56",
        "Total: 3.5 kg",
        "-42.5",
        "1.5e10",
        "42 dollars and 50 cents"
    ]

    for text in test_cases:
        num = executor.extract_number(text)
        print(f"  '{text}' -> {num}")

    print("\nTesting expression evaluation:")
    expressions = [
        "120 * 0.35",
        "100 + 50 - 25",
        "10 / 2",
        "2 ** 3"
    ]

    for expr in expressions:
        result = executor.evaluate_expression(expr)
        print(f"  '{expr}' = {result}")

    print("\nTesting answer comparison:")
    comparisons = [
        ("42", "42.0"),
        ("#### 1234", "1234"),
        ("$100.50", "100.5"),
        ("The answer is 42", "42")
    ]

    for ans1, ans2 in comparisons:
        match = executor.compare_answers(ans1, ans2)
        print(f"  '{ans1}' == '{ans2}': {match}")

    print("\nTesting calculation verification:")
    verifications = [
        ("120 * 0.35", "42"),
        ("100 - 25", "75"),
        ("10 / 2", "5.0")
    ]

    for expr, expected in verifications:
        is_correct, computed = executor.verify_calculation(expr, expected)
        print(f"  '{expr}' = {expected}: {is_correct} (computed: {computed})")
