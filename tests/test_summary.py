import pandas as pd
import pytest

from agent_eval_mlflow.metrics import HEADLINE_METRICS
from agent_eval_mlflow.summary import aggregate, markdown_table, seed_frame


def _runs(rows):
    data = {"params.retriever": [], "params.guardrail": [], "params.top_k": [], "params.seed": []}
    data.update({f"metrics.{m}": [] for m in HEADLINE_METRICS})
    for retriever, seed, pass_rate in rows:
        data["params.retriever"].append(retriever)
        data["params.guardrail"].append("baseline")
        data["params.top_k"].append("2")
        data["params.seed"].append(str(seed))
        for m in HEADLINE_METRICS:
            data[f"metrics.{m}"].append(pass_rate if m == "task_pass_rate" else 1.0)
    return pd.DataFrame(data)


def test_aggregate_uses_sample_std_across_seeds():
    df = seed_frame(_runs([("bm25", 11, 0.5), ("bm25", 22, 0.7), ("bm25", 33, 0.9)]))
    row = aggregate(df, ["retriever"]).iloc[0]
    assert row["task_pass_rate_mean"] == pytest.approx(0.7)
    assert row["task_pass_rate_std"] == pytest.approx(0.2)  # sample std of 0.5, 0.7, 0.9
    assert row["n_runs"] == 3
    assert "| bm25 | 0.700 ± 0.200 |" in markdown_table(aggregate(df, ["retriever"]), ["retriever"])
