"""Benchmark metrics and failure classification. Definitions are documented in README.md."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

HEADLINE_METRICS = ("task_pass_rate", "grounding_precision", "tool_schema_valid_rate",
                    "abstention_accuracy", "avg_steps", "p95_latency_ms")

# Precedence order: the first matching type is assigned to a failed case.
FAILURE_TYPES = ("missed_abstention", "wrong_tool", "schema_invalid", "tool_error",
                 "retrieval_miss", "false_abstention", "distractor_answer", "wrong_answer")


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    gold_answer: str | None
    predicted: str | None
    passed: bool
    gold_doc_ids: list[str]
    citations: list[str]
    evidence_ids: list[str]
    retrieved_doc_ids: list[str]
    tool_calls: list[dict]
    expected_tool: str | None
    steps: int
    latency_ms: float
    forced_abstention: bool = False

    @property
    def abstained(self) -> bool:
        return self.predicted is None

    @property
    def model_calls(self) -> list[dict]:
        """Tool invocations issued by the model (excludes guardrail re-retrievals)."""
        return [c for c in self.tool_calls if not c["system"]]


def classify_failure(r: CaseResult) -> str | None:
    if r.passed:
        return None
    if r.gold_answer is None:
        return "missed_abstention"
    calls = r.model_calls
    if r.expected_tool and all(c["tool"] != r.expected_tool for c in calls):
        return "wrong_tool"
    if any(not c["valid"] for c in calls):
        return "schema_invalid"
    if any(c["error_kind"] in ("transient", "execution") for c in calls):
        return "tool_error"
    if r.gold_doc_ids and not set(r.gold_doc_ids) <= set(r.retrieved_doc_ids):
        return "retrieval_miss"
    if r.abstained:
        return "false_abstention"
    if r.gold_doc_ids and not set(r.evidence_ids) <= set(r.gold_doc_ids):
        return "distractor_answer"
    return "wrong_answer"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def compute_metrics(results: Sequence[CaseResult]) -> dict[str, float]:
    answered = [r for r in results if not r.abstained]
    n_cited = sum(len(r.citations) for r in answered)
    n_cited_gold = sum(len(set(r.citations) & set(r.gold_doc_ids)) for r in answered)
    calls = [c for r in results for c in r.model_calls]
    answerable = [r for r in results if r.gold_answer is not None]
    unanswerable = [r for r in results if r.gold_answer is None]
    metrics = {
        "task_pass_rate": _mean([r.passed for r in results]),
        "grounding_precision": n_cited_gold / n_cited if n_cited else float("nan"),
        "tool_schema_valid_rate": _mean([c["valid"] for c in calls]),
        "abstention_accuracy": _mean([r.abstained == (r.gold_answer is None) for r in results]),
        "abstain_recall": _mean([r.abstained for r in unanswerable]),
        "false_abstention_rate": _mean([r.abstained for r in answerable]),
        "avg_steps": _mean([r.steps for r in results]),
        "p95_latency_ms": float(np.percentile([r.latency_ms for r in results], 95)),
        "answered_rate": len(answered) / len(results),
        "avg_citations": n_cited / len(answered) if answered else 0.0,
    }
    for category in sorted({r.category for r in results}):
        metrics[f"pass_rate_{category}"] = _mean([r.passed for r in results if r.category == category])
    return metrics


def failure_breakdown(results: Sequence[CaseResult]) -> dict:
    labels = {r.case_id: classify_failure(r) for r in results}
    by_category: dict[str, Counter] = {}
    for r in results:
        if labels[r.case_id]:
            by_category.setdefault(r.category, Counter())[labels[r.case_id]] += 1
    counts = Counter(label for label in labels.values() if label)
    return {
        "n_cases": len(results),
        "n_failed": sum(counts.values()),
        "by_type": {t: counts.get(t, 0) for t in FAILURE_TYPES},
        "by_category": {c: dict(sorted(v.items())) for c, v in sorted(by_category.items())},
    }


def failures_jsonl(results: Sequence[CaseResult]) -> str:
    lines = []
    for r in results:
        label = classify_failure(r)
        if label:
            row = asdict(r)
            row["failure_type"] = label
            row["abstained"] = r.abstained
            lines.append(json.dumps(row, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")
