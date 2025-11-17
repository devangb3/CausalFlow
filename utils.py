"""
Utility functions used across the codebase.
Centralizes common operations to reduce code duplication.
"""

from typing import Optional
from trace_logger import Step, StepType


def summarize_step(step: Step) -> str:
    """
    Convert a Step object into a human-readable summary string.

    Args:
        step: The Step object to summarize

    Returns:
        A concise string representation of the step
    """
    if step.step_type == StepType.REASONING:
        return f"[Reasoning] {step.text}"

    elif step.step_type == StepType.TOOL_CALL:
        args_str = str(step.tool_args) if step.tool_args else ""
        return f"[Tool Call] {step.tool_name}({args_str})"

    elif step.step_type == StepType.TOOL_RESPONSE:
        output_str = str(step.tool_output)[:100]  # Truncate long outputs
        return f"[Tool Response] {output_str}"

    elif step.step_type == StepType.MEMORY_ACCESS:
        return f"[Memory Access] Key: {step.memory_key}, Value: {step.memory_value}"

    elif step.step_type == StepType.ENVIRONMENT_ACTION:
        return f"[Environment Action] {step.action}"

    elif step.step_type == StepType.ENVIRONMENT_OBSERVATION:
        return f"[Environment Observation] {step.observation}"

    elif step.step_type == StepType.FINAL_ANSWER:
        return f"[Final Answer] {step.text}"

    else:
        return f"[Unknown Step Type] {step.step_type}"


def extract_step_text(step: Step) -> str:
    """
    Extract the primary text content from a step based on its type.

    Args:
        step: The Step object to extract text from

    Returns:
        The text content of the step
    """
    if step.step_type == StepType.REASONING:
        return step.text or ""

    elif step.step_type == StepType.TOOL_CALL:
        # For tool calls, format as: tool_name(args)
        args_str = str(step.tool_args) if step.tool_args else ""
        return f"{step.tool_name}({args_str})"

    elif step.step_type == StepType.TOOL_RESPONSE:
        return str(step.tool_output) if step.tool_output else ""

    elif step.step_type == StepType.MEMORY_ACCESS:
        return f"Memory[{step.memory_key}] = {step.memory_value}"

    elif step.step_type == StepType.ENVIRONMENT_ACTION:
        return step.action or ""

    elif step.step_type == StepType.ENVIRONMENT_OBSERVATION:
        return step.observation or ""

    elif step.step_type == StepType.FINAL_ANSWER:
        return step.text or ""

    return ""


def format_step_context(step: Step, context: str) -> str:
    """
    Format a step with its context for prompts.

    Args:
        step: The Step object
        context: Context information (e.g., from dependencies)

    Returns:
        Formatted string combining step and context
    """
    step_summary = summarize_step(step)

    if context:
        return f"""Step ID: {step.step_id}
Type: {step.step_type.value}
Content: {step_summary}

Context from dependencies:
{context}"""
    else:
        return f"""Step ID: {step.step_id}
Type: {step.step_type.value}
Content: {step_summary}"""


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate a simple token-based similarity score between two texts.
    Used for minimality scoring.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0.0 and 1.0 (1.0 = identical)
    """
    # Simple word-based tokenization
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0  # Both empty = identical

    if not tokens1 or not tokens2:
        return 0.0  # One empty, one not = completely different

    # Jaccard similarity
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def calculate_minimality_score(original: str, modified: str) -> float:
    """
    Calculate minimality score based on how much text was changed.
    Higher score = more minimal (fewer changes).

    Args:
        original: Original text
        modified: Modified text

    Returns:
        Minimality score between 0.0 and 1.0 (1.0 = no changes)
    """
    if original == modified:
        return 1.0

    # Simple token-based approach
    original_tokens = original.split()
    modified_tokens = modified.split()

    if not original_tokens:
        return 0.0 if modified_tokens else 1.0

    # Calculate the proportion of unchanged tokens
    max_len = max(len(original_tokens), len(modified_tokens))

    # Count matching tokens (simple approach)
    matches = sum(1 for i in range(min(len(original_tokens), len(modified_tokens)))
                  if original_tokens[i] == modified_tokens[i])

    # Additional penalty for length difference
    len_diff_penalty = abs(len(original_tokens) - len(modified_tokens)) / max_len

    minimality = (matches / max_len) * (1 - len_diff_penalty * 0.5)

    return max(0.0, min(1.0, minimality))


def parse_agreement(text: str) -> bool:
    """
    Parse agreement from critique response text.

    Args:
        text: Response text containing agreement indicator

    Returns:
        True if AGREE, False if DISAGREE or PARTIAL
    """
    text_upper = text.upper()

    if "AGREE" in text_upper and "DISAGREE" not in text_upper:
        return True

    return False


def safe_get(dict_obj: dict, key: str, default=None):
    """
    Safely get a value from a dictionary with a default.

    Args:
        dict_obj: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value from dictionary or default
    """
    return dict_obj.get(key, default)
