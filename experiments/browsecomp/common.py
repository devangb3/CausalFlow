from typing import Any, Callable, Dict, List, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import jinja2

from .browsecomp_types import EvalResult, SingleEvalResult


# Jinja2 environment for HTML rendering
jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)


# HTML template for rendering evaluation results
HTML_JINJA = """
<div class="eval-result">
    <h3>Conversation</h3>
    {% for msg in prompt_messages %}
    <div class="message {{ msg.role }}">
        <strong>{{ msg.role }}:</strong>
        <pre>{{ msg.content }}</pre>
    </div>
    {% endfor %}
    
    <div class="message assistant">
        <strong>{{ next_message.role }}:</strong>
        <pre>{{ next_message.content }}</pre>
    </div>
    
    <h3>Result</h3>
    <p><strong>Score:</strong> {{ score }}</p>
    <p><strong>Correct Answer:</strong> {{ correct_answer }}</p>
    <p><strong>Extracted Answer:</strong> {{ extracted_answer }}</p>
</div>
"""


T = TypeVar('T')
R = TypeVar('R')


def map_with_progress(
    fn: Callable[[T], R],
    items: List[T],
    max_workers: int = 1,
    desc: str = "Processing"
) -> List[R]:
    """Map a function over items with a progress bar.
    
    Args:
        fn: Function to apply to each item.
        items: List of items to process.
        max_workers: Number of parallel workers (default 1 for determinism).
        desc: Description for progress bar.
        
    Returns:
        List of results in the same order as inputs.
    """
    if max_workers == 1:
        # Sequential processing for determinism
        results = []
        for item in tqdm(items, desc=desc):
            results.append(fn(item))
        return results
    
    # Parallel processing (order preserved)
    results: List[R] = [None] * len(items)  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(fn, item): idx 
            for idx, item in enumerate(items)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(items), desc=desc):
            idx = future_to_idx[future]
            results[idx] = future.result()
    
    return results


def aggregate_results(results: List[SingleEvalResult]) -> EvalResult:

    if not results:
        return EvalResult(
            score=0.0,
            metrics={},
            htmls=[],
            convos=[]
        )
    
    # Compute average score
    total_score = sum(r.score for r in results)
    avg_score = total_score / len(results)
    
    # Aggregate metrics
    aggregated_metrics: Dict[str, Any] = {}
    metric_keys = set()
    for r in results:
        metric_keys.update(r.metrics.keys())
    
    for key in metric_keys:
        values = [r.metrics.get(key, 0) for r in results]
        if all(isinstance(v, (int, float)) for v in values):
            aggregated_metrics[key] = sum(values) / len(values)
        else:
            aggregated_metrics[key] = values
    
    return EvalResult(
        score=avg_score,
        metrics=aggregated_metrics,
        htmls=[r.html for r in results],
        convos=[r.convo for r in results]
    )
