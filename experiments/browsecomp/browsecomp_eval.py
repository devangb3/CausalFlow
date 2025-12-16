import base64
import hashlib
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas
from llm_client import LLMClient

from . import common
from .browsecomp_types import Eval, EvalResult, SamplerBase, SingleEvalResult


LOCAL_CSV_PATH = Path(__file__).parent.parent.parent / "browse_comp_test_set.csv"
REMOTE_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
You are an answer checker. Use ONLY the provided [response] and [correct_answer]. Do NOT use or infer the original question.

Response: {response}

Correct Answer: {correct_answer}

Your judgement must follow this format:

extracted_final_answer: Extract the final exact answer from Response. If none is present, output 'None'.

reasoning: Compare extracted_final_answer against Correct Answer only. Consider them equivalent when:
- They match after normalization (case-insensitive, trimmed whitespace/punctuation).
- For person names, matching first and last names is sufficient even if middle names/initials are missing or abbreviated, unless first or last names differ.
- Minor formatting or ordering differences that do not change meaning are acceptable.

correct: 'yes' if extracted_final_answer is equivalent to [correct_answer] (including the lenient name rule and small numeric rounding differences); otherwise 'no'.

confidence: Extract the confidence between 0 and 100 from [response]; use 100 if none is provided.
""".strip()

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def load_browsecomp_examples(
    num_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:

    df = pandas.read_csv(LOCAL_CSV_PATH) if LOCAL_CSV_PATH.exists() else pandas.read_csv(REMOTE_CSV_URL)
    examples = [row.to_dict() for _, row in df.iterrows()]
    
    if num_examples:
        rng = random.Random(42)
        examples = rng.sample(examples, min(num_examples, len(examples)))
    
    return examples


def grade_response(correct_answer: str, response: str, llm: LLMClient) -> bool:
    grader_prompt = GRADER_TEMPLATE.format(
        correct_answer=correct_answer,
        response=response,
    )
    
    grading_response = llm.generate(grader_prompt, temperature=0.0)
    
    match = re.search(r"correct: (yes|no)", grading_response, re.IGNORECASE)
    if not match:
        return False
    
    return match.group(1).lower() == "yes"


class BrowseCompEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        num_examples: Optional[int] = None,
    ):
        self.examples = load_browsecomp_examples(
            num_examples=num_examples,
        )
        self.grader_model = grader_model

    def grade_sample(self, correct_answer: str, response: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            correct_answer=correct_answer,
            response=response,
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        grading_response = sampler_response.response_text

        # Fail-fast: raise if judge output cannot be parsed
        match = re.search(r"correct: (yes|no)", grading_response, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Failed to parse judge output. Expected 'correct: yes' or 'correct: no' "
                f"but got:\n{grading_response}"
            )
        
        return match.group(0).lower()

    def __call__(self, sampler: SamplerBase) -> EvalResult:

        def fn(row: dict) -> SingleEvalResult:
            problem = decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = decrypt(row.get("answer", ""), row.get("canary", ""))
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(Question=problem), role="user")
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            grade_result = self.grade_sample(problem, answer, response_text)

            # Metrics based on grading response
            is_correct = grade_result == "correct: yes"
            is_incorrect = grade_result == "correct: no"
            
            score = float(is_correct)

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=answer,
                extracted_answer=response_text,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                "is_correct": is_correct,
                "is_incorrect": is_incorrect,
            })

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples, desc="Evaluating BrowseComp")

        # Aggregate metrics
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
            "is_incorrect": sum(result.metrics["is_incorrect"] for result in results) / len(results),
        }
        print("AGGREGATE METRICS") 
        print(aggregate_metrics) 
        print("##################")

        output_d = {
            "accuracy": aggregate_metrics["is_correct"],
        }
        
        print(f"Accuracy: {output_d['accuracy']:.3f}")
        
        return common.aggregate_results(results)
