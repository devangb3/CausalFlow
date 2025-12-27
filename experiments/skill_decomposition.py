import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from mongodb_storage import MongoDBStorage


EXPERIMENT_FOLDERS = {
    "GSM8K": Path("experiments/gsm8k"),
    "MBPP": Path("experiments/mbpp"),
    "Humaneval": Path("experiments/humaneval"),
    "BrowseComp": Path("experiments/browsecomp"),
    "SealQA": Path("experiments/browsecomp"),
    "MedBrowseComp": Path("experiments/browsecomp"),
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_steps_from_run(run: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    steps: List[Dict[str, Any]] = []
    stats = run.get("stats", {})

    for trace_doc in run.get("passing_traces", []):
        trace = trace_doc.get("trace", {})
        for step in trace.get("steps", []):
            steps.append({
                "problem_id": trace_doc.get("problem_id"),
                "problem_statement": trace_doc.get("problem_statement"),
                "gold_answer": trace_doc.get("gold_answer"),
                "final_answer": trace_doc.get("final_answer"),
                "trace_success": True,
                "step_id": _coerce_int(step.get("step_id")),
                "step_type": step.get("step_type"),
                "tool_name": step.get("tool_name"),
                "tool_args": step.get("tool_args"),
                "text": step.get("text"),
                "action": step.get("action"),
                "observation": step.get("observation"),
                "tool_output": step.get("tool_output"),
                "is_causal": False,
                "repaired_successfully": False,
            })

    for trace_doc in run.get("failing_traces", []):
        trace = trace_doc.get("trace", {})
        analysis = trace_doc.get("analysis", {}) or {}
        causal_steps = analysis.get("causal_attribution", {}).get("causal_steps", [])
        causal_step_ids = {step_id for step_id in (_coerce_int(s) for s in causal_steps) if step_id is not None}

        repaired = analysis.get("counterfactual_repair", {}).get("successful_repairs", {})
        repaired_step_ids = {step_id for step_id in (_coerce_int(s) for s in repaired.keys()) if step_id is not None}

        for step in trace.get("steps", []):
            step_id = _coerce_int(step.get("step_id"))
            steps.append({
                "problem_id": trace_doc.get("problem_id"),
                "problem_statement": trace_doc.get("problem_statement"),
                "gold_answer": trace_doc.get("gold_answer"),
                "final_answer": trace_doc.get("final_answer"),
                "trace_success": False,
                "step_id": step_id,
                "step_type": step.get("step_type"),
                "tool_name": step.get("tool_name"),
                "tool_args": step.get("tool_args"),
                "text": step.get("text"),
                "action": step.get("action"),
                "observation": step.get("observation"),
                "tool_output": step.get("tool_output"),
                "is_causal": step_id in causal_step_ids if step_id is not None else False,
                "repaired_successfully": step_id in repaired_step_ids if step_id is not None else False,
            })

    return steps, {
        "total_traces": stats.get("total", 0),
        "passing_traces": stats.get("passing", 0),
        "failing_traces": stats.get("failing", 0),
    }


def _shorten(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _summarize_step_for_prompt(step: Dict[str, Any]) -> str:
    parts: List[str] = []
    step_type = step.get("step_type")
    if step_type:
        parts.append(f"type={step_type}")
    tool_name = step.get("tool_name")
    if tool_name:
        parts.append(f"tool={tool_name}")

    content = step.get("text") or step.get("action") or step.get("observation")
    if content:
        parts.append(f"text={_shorten(str(content), 160)}")
    elif step.get("tool_output"):
        parts.append(f"output={_shorten(str(step.get('tool_output')), 160)}")

    return ", ".join(parts) if parts else "empty_step"


def _build_trace_index(steps: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    trace_index: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for step in steps:
        trace_index[step.get("problem_id")].append(step)
    for problem_id, trace_steps in trace_index.items():
        trace_steps.sort(key=lambda s: s.get("step_id") if s.get("step_id") is not None else -1)
    return trace_index


def _build_context_lines(
    trace_steps: List[Dict[str, Any]],
    step_id: Optional[int],
) -> List[str]:
    if step_id is None:
        return []
    prior_steps = [step for step in trace_steps if step.get("step_id") is not None and step.get("step_id") < step_id]
    return [f"Step {step.get('step_id')}: {_summarize_step_for_prompt(step)}" for step in prior_steps]


def _make_step_signature(step: Dict[str, Any]) -> str:
    signature_fields = {
        "step_type": step.get("step_type"),
        "tool_name": step.get("tool_name"),
        "tool_args": step.get("tool_args"),
        "text": step.get("text"),
        "action": step.get("action"),
        "observation": step.get("observation"),
        "tool_output": step.get("tool_output"),
    }
    raw = json.dumps(signature_fields, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_llm_cache(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _save_llm_cache(path: Optional[str], cache: Dict[str, Dict[str, Any]]) -> None:
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _label_skills_with_llm(
    steps: List[Dict[str, Any]],
    trace_index: Dict[Any, List[Dict[str, Any]]],
    model: str,
    temperature: float,
    cache_path: str,
) -> None:
    cache = _load_llm_cache(cache_path)
    llm = LLMClient(model=model, temperature=temperature)

    system_message = (
        "You are a meta-critic analyzing agent execution failures. Your task is to identify the underlying skill "
        "or capability that failed in a causal step, abstracting away from the specific problem context. "
        "Focus on the generalizable failure pattern: what fundamental capability was missing or incorrectly applied? "
        "Use domain-agnostic skill labels (e.g., 'tool argument synthesis', 'evidence selection', 'logical reasoning') "
        "that capture transferable knowledge, not problem-specific details."
    )

    for step in steps:
        signature = _make_step_signature(step)
        cached = cache.get(signature)
        if cached:
            step["skill_label"] = cached.get("skill_label")
            step["skill_description"] = cached.get("skill_description")
            step["skill_confidence"] = cached.get("confidence")
            step["skill_rationale"] = cached.get("rationale")
            continue

        trace_steps = trace_index.get(step.get("problem_id"), [])
        context_lines = _build_context_lines(trace_steps, step.get("step_id"))
        context_text = "\n".join(context_lines) if context_lines else "No prior context."

        prompt = f"""You are labeling a causal step from an agent trace with the underlying skill that failed.

Problem statement:
{step.get("problem_statement") or "Not available"}

Step details:
step_id: {step.get("step_id")}
step_type: {step.get("step_type")}
tool_name: {step.get("tool_name") or "None"}
tool_args: {json.dumps(step.get("tool_args"), default=str)}
step_content: {_shorten(str(step.get("text") or step.get("action") or step.get("observation") or step.get("tool_output") or ""), 500)}

Recent context:
{context_text}

Provide a short skill label (2-4 words), a 1-2 sentence description of the skill/failure pattern, a confidence score, and a brief rationale grounded in the step content.
Do NOT reference the final answer or dataset-specific entities. Use general capability terms (e.g., "tool argument synthesis", "evidence selection", "math expression parsing")."""

        result = llm.generate_structured(
            prompt,
            schema_name="skill_attribution",
            system_message=system_message,
            temperature=temperature,
            model_name=model,
        )

        step["skill_label"] = result.skill_label
        step["skill_description"] = result.skill_description
        step["skill_confidence"] = result.confidence
        step["skill_rationale"] = result.rationale

        cache[signature] = {
            "skill_label": result.skill_label,
            "skill_description": result.skill_description,
            "confidence": result.confidence,
            "rationale": result.rationale,
        }

    _save_llm_cache(cache_path, cache)


def _build_step_tokens(step: Dict[str, Any], max_text_tokens: int) -> List[str]:
    tokens: List[str] = []
    step_type = step.get("step_type")
    if step_type:
        tokens.append(f"type:{step_type}")
    tool_name = step.get("tool_name")
    if tool_name:
        tokens.append(f"tool:{tool_name}")

    tool_args = step.get("tool_args")
    if isinstance(tool_args, dict):
        for key, value in tool_args.items():
            tokens.append(f"arg:{key}")
            if isinstance(value, str) and value.strip():
                tokens.extend(_tokenize(value)[:8])
            elif isinstance(value, (int, float)):
                tokens.append(str(value))

    text_parts: List[str] = []
    for field in ("text", "action", "observation"):
        value = step.get(field)
        if isinstance(value, str) and value.strip():
            text_parts.append(value)
    if step.get("tool_output") and step_type == "tool_response":
        text_parts.append(str(step.get("tool_output")))

    if text_parts:
        merged = " ".join(text_parts)
        tokens.extend(_tokenize(merged)[:max_text_tokens])

    return tokens


def _build_skill_tokens(step: Dict[str, Any], max_text_tokens: int) -> List[str]:
    tokens: List[str] = []
    label = step.get("skill_label")
    if label:
        tokens.extend(_tokenize(str(label)))
    description = step.get("skill_description")
    if description:
        tokens.extend(_tokenize(str(description))[:max_text_tokens])

    step_type = step.get("step_type")
    if step_type:
        tokens.append(f"type:{step_type}")
    tool_name = step.get("tool_name")
    if tool_name:
        tokens.append(f"tool:{tool_name}")

    if not tokens:
        tokens.extend(_build_step_tokens(step, max_text_tokens=max_text_tokens))
    return tokens


def _build_vocab(token_lists: List[List[str]], max_features: int, min_df: int) -> Dict[str, int]:
    df_counts = Counter()
    for tokens in token_lists:
        df_counts.update(set(tokens))

    filtered = [t for t, c in df_counts.items() if c >= min_df]
    filtered.sort(key=lambda t: (-df_counts[t], t))
    filtered = filtered[:max_features]
    return {token: idx for idx, token in enumerate(filtered)}


def _vectorize(token_lists: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    num_steps = len(token_lists)
    num_features = len(vocab)
    matrix = np.zeros((num_steps, num_features), dtype=float)
    for i, tokens in enumerate(token_lists):
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                matrix[i, idx] += 1.0
    return matrix


def _tfidf_transform(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    num_docs = matrix.shape[0]
    df = np.count_nonzero(matrix > 0, axis=0)
    idf = np.log((1 + num_docs) / (1 + df)) + 1.0
    tfidf = matrix * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return tfidf / norms


def _init_centroids(data: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_samples = data.shape[0]
    centroids = np.zeros((k, data.shape[1]), dtype=float)
    first_idx = rng.integers(num_samples)
    centroids[0] = data[first_idx]

    distances = np.full(num_samples, np.inf)
    for i in range(1, k):
        diff = data[:, None, :] - centroids[:i][None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        distances = np.minimum(distances, np.min(dist_sq, axis=1))
        if distances.sum() == 0:
            next_idx = rng.integers(num_samples)
        else:
            probs = distances / distances.sum()
            next_idx = rng.choice(num_samples, p=probs)
        centroids[i] = data[next_idx]
    return centroids


def _kmeans(data: np.ndarray, k: int, seed: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 1 or data.shape[0] == 0:
        labels = np.zeros(data.shape[0], dtype=int)
        centroids = np.mean(data, axis=0, keepdims=True) if data.shape[0] > 0 else np.zeros((1, data.shape[1]))
        return labels, centroids

    centroids = _init_centroids(data, k, seed)
    labels = np.zeros(data.shape[0], dtype=int)

    for _ in range(max_iter):
        diff = data[:, None, :] - centroids[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        new_labels = np.argmin(dist_sq, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for cluster_idx in range(k):
            cluster_points = data[labels == cluster_idx]
            if cluster_points.size == 0:
                centroids[cluster_idx] = data[np.random.randint(0, data.shape[0])]
            else:
                centroids[cluster_idx] = np.mean(cluster_points, axis=0)

    return labels, centroids


def _summarize_clusters(
    steps: List[Dict[str, Any]],
    labels: np.ndarray,
    centroids: np.ndarray,
    vocab: Dict[str, int],
    top_terms: int,
    examples_per_cluster: int,
) -> List[Dict[str, Any]]:
    inv_vocab = {idx: token for token, idx in vocab.items()}
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)

    summaries: List[Dict[str, Any]] = []
    for cluster_id in sorted(clusters.keys()):
        indices = clusters[cluster_id]
        cluster_steps = [steps[i] for i in indices]
        step_type_counts = Counter(step.get("step_type") for step in cluster_steps if step.get("step_type"))
        tool_counts = Counter(step.get("tool_name") for step in cluster_steps if step.get("tool_name"))
        skill_label_counts = Counter(step.get("skill_label") for step in cluster_steps if step.get("skill_label"))

        causal_count = sum(1 for step in cluster_steps if step.get("is_causal"))
        repair_count = sum(1 for step in cluster_steps if step.get("repaired_successfully"))
        total_count = len(cluster_steps)

        centroid = centroids[cluster_id] if cluster_id < centroids.shape[0] else np.zeros(len(vocab))
        top_indices = np.argsort(-centroid)[:top_terms]
        top_tokens = [inv_vocab[idx] for idx in top_indices if centroid[idx] > 0]

        examples: List[Dict[str, Any]] = []
        for step in cluster_steps[:examples_per_cluster]:
            text = step.get("text") or step.get("action") or step.get("observation") or ""
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            examples.append({
                "problem_id": step.get("problem_id"),
                "step_id": step.get("step_id"),
                "step_type": step.get("step_type"),
                "tool_name": step.get("tool_name"),
                "skill_label": step.get("skill_label"),
                "snippet": snippet,
                "is_causal": step.get("is_causal", False),
            })

        summaries.append({
            "skill_id": cluster_id,
            "size": total_count,
            "causal_steps": causal_count,
            "causal_rate": round(causal_count / total_count, 4) if total_count else 0.0,
            "repair_success_count": repair_count,
            "repair_success_rate": round(repair_count / total_count, 4) if total_count else 0.0,
            "dominant_step_types": step_type_counts.most_common(5),
            "dominant_tools": tool_counts.most_common(5),
            "dominant_skill_labels": skill_label_counts.most_common(5),
            "top_terms": top_tokens,
            "examples": examples,
        })

    return summaries


def _write_reports(report: Dict[str, Any], output_prefix: Path) -> Tuple[Path, Path]:
    if output_prefix.suffix:
        output_prefix = output_prefix.with_suffix("")

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines: List[str] = []
    metadata = report.get("metadata", {})
    md_lines.append("# Causal Skill Decomposition Report")
    md_lines.append("")
    md_lines.append(f"- Experiment: {metadata.get('experiment_name', 'unknown')}")
    md_lines.append(f"- Run ID: {metadata.get('run_id', 'unknown')}")
    md_lines.append(f"- Total traces: {metadata.get('total_traces', 0)}")
    md_lines.append(f"- Passing traces: {metadata.get('passing_traces', 0)}")
    md_lines.append(f"- Failing traces: {metadata.get('failing_traces', 0)}")
    md_lines.append(f"- Total steps analyzed: {metadata.get('total_steps', 0)}")
    md_lines.append(f"- Causal steps labeled: {metadata.get('causal_steps', 0)}")
    md_lines.append(f"- Clusters (skills): {metadata.get('num_clusters', 0)}")
    md_lines.append(f"- Vocabulary size: {metadata.get('vocab_size', 0)}")
    md_lines.append(f"- Skill labeling model: {metadata.get('llm_model')}")
    md_lines.append("")

    for cluster in report.get("skill_clusters", []):
        md_lines.append(f"## Skill {cluster.get('skill_id')}")
        md_lines.append(f"- Size: {cluster.get('size')}")
        md_lines.append(f"- Causal rate: {cluster.get('causal_rate')}")
        md_lines.append(f"- Repair success rate: {cluster.get('repair_success_rate')}")
        md_lines.append(f"- Dominant step types: {cluster.get('dominant_step_types')}")
        md_lines.append(f"- Dominant tools: {cluster.get('dominant_tools')}")
        md_lines.append(f"- Dominant skill labels: {cluster.get('dominant_skill_labels')}")
        md_lines.append(f"- Top terms: {cluster.get('top_terms')}")
        md_lines.append("- Examples:")
        for example in cluster.get("examples", []):
            md_lines.append(
                f"  - [{example.get('problem_id')}] step {example.get('step_id')} "
                f"{example.get('step_type')} {example.get('tool_name') or ''} "
                f"[{example.get('skill_label') or 'unlabeled'}]: {example.get('snippet')}"
            )
        md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return json_path, md_path


def _default_output_prefix(experiment_name: Optional[str]) -> Path:
    if experiment_name and experiment_name in EXPERIMENT_FOLDERS:
        return EXPERIMENT_FOLDERS[experiment_name] / "skill_decomposition_report"
    return Path("skill_decomposition_report")


def run_skill_decomposition(
    experiment_name: Optional[str],
    run_id: Optional[str],
    clusters: Optional[int],
    max_features: int,
    max_text_tokens: int,
    llm_model: str,
    llm_temperature: float,
    seed: int,
    max_clusters: int,
    examples_per_cluster: int,
    top_terms: int,
) -> Dict[str, Any]:
    storage = MongoDBStorage()

    try:
        run = storage.get_run(run_id) if run_id else None
        if run is None:
                raise ValueError("No matching run found in MongoDB.")

        steps, trace_stats = _extract_steps_from_run(run)
        if not steps:
            raise ValueError("No steps found in the run.")

        causal_steps = [step for step in steps if step.get("is_causal")]
        if not causal_steps:
            raise ValueError("No causal steps found to label. Run CausalFlow analysis first.")

        trace_index = _build_trace_index(steps)
        _label_skills_with_llm(
            steps=causal_steps,
            trace_index=trace_index,
            model=llm_model,
            temperature=llm_temperature,
            cache_path="skill_decomposition_cache.json",
        )

        token_lists = [_build_skill_tokens(step, max_text_tokens=max_text_tokens) for step in causal_steps]
        min_df = 2 if len(token_lists) > 200 else 1
        vocab = _build_vocab(token_lists, max_features=max_features, min_df=min_df)
        if not vocab:
            raise ValueError("Vocabulary is empty. Check input data or lower min_df.")

        matrix = _vectorize(token_lists, vocab)
        matrix = _tfidf_transform(matrix)

        num_steps = matrix.shape[0]
        if clusters is None:
            clusters = max(2, int(math.sqrt(num_steps)))
        clusters = max(1, min(clusters, max_clusters, num_steps))

        labels, centroids = _kmeans(matrix, clusters, seed=seed)
        summaries = _summarize_clusters(
            causal_steps,
            labels,
            centroids,
            vocab,
            top_terms=top_terms,
            examples_per_cluster=examples_per_cluster,
        )

        experiment = run.get("experiment_name", experiment_name or "unknown")
        report = {
            "metadata": {
                "experiment_name": experiment,
                "run_id": run.get("run_id"),
                "timestamp": run.get("timestamp"),
                "model_used": run.get("model_used"),
                "total_steps": len(steps),
                "causal_steps": len(causal_steps),
                "num_clusters": clusters,
                "vocab_size": len(vocab),
                "llm_model": llm_model,
                "llm_temperature": llm_temperature,
                **trace_stats,
            },
            "skill_clusters": summaries,
        }

        output_path = _default_output_prefix(experiment)
        json_path, md_path = _write_reports(report, output_path)
        print(f"Wrote skill decomposition report to {json_path} and {md_path}")
        return report
    finally:
        storage.close()

if __name__ == "__main__":
    run_skill_decomposition(
        experiment_name="MBPP",
        run_id="run_MBPP_2025-12-06T08:50:36.324654",
        clusters=None, #Number of skill clusters.
        max_features=800, #Maximum vocabulary size.
        max_text_tokens=120, #Max tokens per step text.
        llm_model="google/gemini-3-flash-preview",
        llm_temperature=0.2,
        seed=7,
        max_clusters=20,
        examples_per_cluster=3,
        top_terms=10,
    )
