"""Aggregate seed-level MLflow runs into mean and std per config and per factor."""

from __future__ import annotations

import pandas as pd

from .metrics import HEADLINE_METRICS

FACTORS = ("retriever", "guardrail", "top_k")
# Columns as they appear in the markdown tables, with the number of decimals shown.
DECIMALS = {"task_pass_rate": 3, "grounding_precision": 3, "tool_schema_valid_rate": 3,
            "abstention_accuracy": 3, "avg_steps": 2, "p95_latency_ms": 0}


def seed_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Turn `mlflow.search_runs` output for seed runs into one row per (config, seed)."""
    df = pd.DataFrame({
        "retriever": runs["params.retriever"],
        "guardrail": runs["params.guardrail"],
        "top_k": runs["params.top_k"].astype(int),
        "seed": runs["params.seed"].astype(int),
    })
    for m in HEADLINE_METRICS:
        df[m] = runs[f"metrics.{m}"].astype(float)
    return df


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and sample std (ddof=1) over all seed rows in each group."""
    grouped = df.groupby(by, sort=True)[list(HEADLINE_METRICS)]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    out["n_runs"] = grouped.size()
    return out.reset_index()


def _cell(row, metric: str) -> str:
    d = DECIMALS[metric]
    return f"{row[metric + '_mean']:.{d}f} ± {row[metric + '_std']:.{d}f}"


def markdown_table(agg: pd.DataFrame, keys: list[str]) -> str:
    cols = keys + list(HEADLINE_METRICS) + ["n_runs"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in agg.to_dict(orient="records"):  # iterrows would upcast top_k to float
        cells = [str(row[k]) for k in keys] + [_cell(row, m) for m in HEADLINE_METRICS]
        lines.append("| " + " | ".join(cells + [str(int(row["n_runs"]))]) + " |")
    return "\n".join(lines)
