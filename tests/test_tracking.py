import json
from pathlib import Path

import mlflow
import pytest

from agent_eval_mlflow.runner import Benchmark, RunConfig
from agent_eval_mlflow.tracking import log_config_sweep

SMALL = ["mh-01", "ab-01", "sv-01", "ts-01"]


@pytest.fixture
def tracking_uri(tmp_path):
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_parent_and_nested_seed_runs_are_logged_with_artifacts(tracking_uri, tmp_path):
    bench = Benchmark(case_ids=SMALL)
    config = RunConfig("bm25", "schema_retry", 4)
    parent_id = log_config_sweep(config, seeds=(11, 22), bench=bench,
                                 tracking_uri=tracking_uri, experiment="t")

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=["t"], output_format="list")
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent_id]
    assert len(runs) == 3
    assert sorted(int(r.data.params["seed"]) for r in children) == [11, 22]
    for r in children:
        assert r.data.params["retriever"] == "bm25"
        assert r.data.params["top_k"] == "4"
        assert r.data.tags["config"] == "bm25__schema_retry__k4"
        assert 0.0 <= r.data.metrics["task_pass_rate"] <= 1.0
        assert "p95_latency_ms" in r.data.metrics
        local = Path(mlflow.artifacts.download_artifacts(run_id=r.info.run_id,
                                                         dst_path=str(tmp_path / r.info.run_id)))
        breakdown = json.loads((local / "failure_breakdown.json").read_text())
        assert breakdown["n_cases"] == len(SMALL)
        failures = (local / "failures.jsonl").read_text().splitlines()
        assert len(failures) == breakdown["n_failed"]
    parent = next(r for r in runs if r.info.run_id == parent_id)
    assert "task_pass_rate_mean" in parent.data.metrics
    assert "task_pass_rate_std" in parent.data.metrics
