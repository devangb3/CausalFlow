import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        #repaired = analysis.get("final_repairs", {})
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
    max_context_steps: int,
    cache_path: Optional[str],
) -> None:
    cache = _load_llm_cache(cache_path)
    llm = LLMClient(model=model, temperature=0.2)

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
            temperature=0.2,
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


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _build_label_entries(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_counts: Counter[str] = Counter()
    label_descriptions: Dict[str, str] = {}
    label_canonical: Dict[str, str] = {}

    for step in steps:
        label = step.get("skill_label")
        if not label:
            continue
        norm = _normalize_label(str(label))
        label_counts[norm] += 1
        if norm not in label_canonical:
            label_canonical[norm] = str(label).strip()
        description = step.get("skill_description")
        if description and norm not in label_descriptions:
            label_descriptions[norm] = str(description).strip()

    entries = []
    for norm_label, count in label_counts.items():
        entries.append({
            "label": label_canonical.get(norm_label, norm_label),
            "description": label_descriptions.get(norm_label, ""),
            "count": count,
            "normalized": norm_label,
        })

    return sorted(entries, key=lambda e: (-e["count"], e["label"]))


def _group_skill_labels_with_llm(
    label_entries: List[Dict[str, Any]],
    target_groups: int,
    model: str,
    cache_path: Optional[str],
) -> List[Dict[str, Any]]:
    cache = _load_llm_cache(cache_path)
    payload = {
        "labels": label_entries,
        "target_groups": target_groups,
        "model": model,
    }
    cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    llm = LLMClient(model=model, temperature=0.2)
    label_lines = []
    for entry in label_entries:
        desc = entry["description"] or "No description."
        label_lines.append(f"- {entry['label']} (count={entry['count']}): {desc}")

    group_target = f"Create around {target_groups} groups."
    prompt = f"""You are grouping failure skill labels into higher-level semantic categories.

Skill labels:
{chr(10).join(label_lines)}

Requirements:
- Each label must appear in exactly one group.
- Group labels by semantic similarity in underlying capability.
- Provide a short group name, a concise description, and a brief rationale.
- {group_target}
"""

    result = llm.generate_structured(
        prompt,
        schema_name="skill_grouping",
        system_message="You are a meta-critic who clusters skill labels into coherent failure skill groups.",
        temperature=0.2,
        model_name=model,
    )

    groups = [group.model_dump() for group in result.groups]
    name_counts: Dict[str, int] = {}
    for group in groups:
        name = str(group.get("group_name") or "Unnamed").strip()
        count = name_counts.get(name, 0) + 1
        name_counts[name] = count
        if count > 1:
            group["group_name"] = f"{name} ({count})"
        else:
            group["group_name"] = name
    cache[cache_key] = groups
    _save_llm_cache(cache_path, cache)
    return groups


def _assign_groups_to_labels(
    label_entries: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], List[str]]:
    label_lookup = {entry["normalized"]: entry["label"] for entry in label_entries}
    assigned: Dict[str, str] = {}
    unassigned = set(label_lookup.keys())

    for group in groups:
        for label in group.get("member_labels", []):
            norm = _normalize_label(str(label))
            canonical = label_lookup.get(norm)
            if canonical and canonical not in assigned:
                assigned[canonical] = group["group_name"]
                unassigned.discard(norm)

    leftovers = [label_lookup[norm] for norm in sorted(unassigned)]
    return assigned, leftovers


def _summarize_groups(
    steps: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    label_entries: List[Dict[str, Any]],
    examples_per_group: int,
) -> List[Dict[str, Any]]:
    label_to_group, leftovers = _assign_groups_to_labels(label_entries, groups)
    label_desc_map = {entry["label"]: entry.get("description", "") for entry in label_entries}
    group_map: Dict[str, Dict[str, Any]] = {}

    for group in groups:
        group_name = group.get("group_name")
        if not group_name:
            continue
        group_map[group_name] = {
            "group_name": group_name,
            "group_description": group.get("group_description", ""),
            "rationale": group.get("rationale", ""),
            "member_labels": [],
            "size": 0,
            "repair_success_count": 0,
            "repair_success_rate": 0.0,
            "dominant_step_types": [],
            "dominant_tools": [],
            "examples": [],
        }

    if leftovers:
        group_map["Unassigned"] = {
            "group_name": "Unassigned",
            "group_description": "Labels not mapped by the skill labeling step. Might be outlier failures",
            "rationale": "Auto-added fallback group.",
            "member_labels": [],
            "size": 0,
            "repair_success_count": 0,
            "repair_success_rate": 0.0,
            "dominant_step_types": [],
            "dominant_tools": [],
            "examples": [],
        }

    group_steps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for step in steps:
        label = step.get("skill_label")
        group_name = label_to_group.get(label, "Unassigned")
        group_steps[group_name].append(step)

    for group_name, group_step_list in group_steps.items():
        group_entry = group_map.get(group_name)
        if not group_entry:
            continue
        step_type_counts = Counter(step.get("step_type") for step in group_step_list if step.get("step_type"))
        tool_counts = Counter(step.get("tool_name") for step in group_step_list if step.get("tool_name"))
        label_counts = Counter(step.get("skill_label") for step in group_step_list if step.get("skill_label"))
        repair_count = sum(1 for step in group_step_list if step.get("repaired_successfully"))

        group_entry["size"] = len(group_step_list)
        group_entry["repair_success_count"] = repair_count
        group_entry["repair_success_rate"] = round(repair_count / len(group_step_list), 4) if group_step_list else 0.0
        group_entry["dominant_step_types"] = step_type_counts.most_common(5)
        group_entry["dominant_tools"] = tool_counts.most_common(5)
        member_labels: List[Dict[str, Any]] = []
        for label, count in label_counts.most_common():
            member_labels.append({
                "label": label,
                "count": count,
                "description": label_desc_map.get(label, ""),
            })
        group_entry["member_labels"] = member_labels

        examples: List[Dict[str, Any]] = []
        for step in group_step_list[:examples_per_group]:
            text = step.get("text") or step.get("action") or step.get("observation") or ""
            snippet = text.strip().replace("\n", " ")
            
            examples.append({
                "problem_id": step.get("problem_id"),
                "step_id": step.get("step_id"),
                "step_type": step.get("step_type"),
                "tool_name": step.get("tool_name"),
                "skill_label": step.get("skill_label"),
                "snippet": snippet,
                "is_causal": step.get("is_causal", False),
            })
        group_entry["examples"] = examples

    summaries = list(group_map.values())
    summaries.sort(key=lambda g: (-g["size"], g["group_name"]))
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
    md_lines.append(f"- Skill groups: {metadata.get('num_groups', 0)}")
    if metadata.get("skill_label_model"):
        md_lines.append(f"- Skill labeling model: {metadata.get('skill_label_model')}")
    if metadata.get("skill_grouping_model"):
        md_lines.append(f"- Skill grouping model: {metadata.get('skill_grouping_model')}")
    md_lines.append("")

    for group in report.get("skill_groups", []):
        md_lines.append(f"## {group.get('group_name')}")
        if group.get("group_description"):
            md_lines.append(f"- Description: {group.get('group_description')}")
        md_lines.append(f"- Size: {group.get('size')}")
        md_lines.append(f"- Repair success rate: {group.get('repair_success_rate')}")
        md_lines.append(f"- Dominant step types: {group.get('dominant_step_types')}")
        md_lines.append(f"- Dominant tools: {group.get('dominant_tools')}")
        md_lines.append("- Member skill labels:")
        for member in group.get("member_labels", []):
            label = member.get("label", "")
            count = member.get("count", 0)
            description = member.get("description") or "No description."
            md_lines.append(f"  - {label} (count={count}): {description}")
        md_lines.append("- Examples:")
        for example in group.get("examples", []):
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
    llm_model: str,
    llm_cache_file: Optional[str],
    grouping_model: str,
    grouping_cache_file: Optional[str],
    max_context_steps: int,
    examples_per_group: int,
) -> Dict[str, Any]:
    storage = MongoDBStorage()

    try:
        run = storage.get_run(run_id) if run_id else None
        if run is None:
            if not experiment_name:
                raise ValueError("Provide --experiment or --run-id.")
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
            max_context_steps=max_context_steps,
            cache_path=llm_cache_file,
        )

        label_entries = _build_label_entries(causal_steps)
        if not label_entries:
            raise ValueError("No skill labels were produced. Check causal steps and LLM labeling.")
        
        target_group_count = max(2, int(math.sqrt(len(label_entries))))

        groups = _group_skill_labels_with_llm(
            label_entries=label_entries,
            target_groups=target_group_count,
            model=grouping_model,
            cache_path=grouping_cache_file,
        )

        summaries = _summarize_groups(
            causal_steps,
            groups,
            label_entries=label_entries,
            examples_per_group=examples_per_group,
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
                "num_groups": len(summaries),
                "skill_label_model": llm_model,
                "skill_label_cache_file": llm_cache_file,
                "skill_grouping_model": grouping_model,
                "skill_grouping_cache_file": grouping_cache_file,
                "max_context_steps": max_context_steps,
                **trace_stats,
            },
            "skill_groups": summaries,
        }

        output_path = _default_output_prefix(experiment)
        json_path, md_path = _write_reports(report, output_path)
        print(f"Wrote skill decomposition report to {json_path} and {md_path}")
        return report
    finally:
        storage.close()

if __name__ == "__main__":
    run_skill_decomposition(
        experiment_name="GSM8K",
        run_id="run_GSM8K_2025-12-20T07:38:08.801930",
        llm_model="google/gemini-3-flash-preview",
        llm_cache_file=".cache/skill_decomposition/skill_label_cache.json",
        grouping_model="google/gemini-3-flash-preview",
        grouping_cache_file=".cache/skill_decomposition/skill_grouping_cache.json",
        max_context_steps=5,
        examples_per_group=3,
    )
