"""MLflow logging: one parent run per config, one nested child run per seed."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

import mlflow
import numpy as np

from .agent import GUARDRAILS, HeuristicModel, LatencyModel
from .metrics import HEADLINE_METRICS, failure_breakdown, failures_jsonl
from .runner import SEEDS, Benchmark, RunConfig, run_config

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "agent-eval-ablation"


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    # MLflow rejects NaN in some backends; drop undefined metrics instead of logging them.
    return {k: v for k, v in metrics.items() if not math.isnan(v)}


def log_config_sweep(config: RunConfig, seeds: Sequence[int] = SEEDS, bench: Benchmark | None = None,
                     tracking_uri: str = DEFAULT_TRACKING_URI,
                     experiment: str = DEFAULT_EXPERIMENT) -> str:
    """Run `config` once per seed and log it to MLflow. Returns the parent run id."""
    bench = bench or Benchmark()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    guardrail = GUARDRAILS[config.guardrail]
    shared_params = {
        **config.params(),
        "max_retries": guardrail.max_retries,
        "n_cases": len(bench.cases),
        **{f"model.{k}": v for k, v in HeuristicModel(0).params().items()},
        **{f"latency.{k}": v for k, v in LatencyModel().params().items()},
    }
    tags = {"config": config.name, "data_fingerprint": bench.fingerprint,
            "model": HeuristicModel(0).name}
    per_seed: list[dict[str, float]] = []
    with mlflow.start_run(run_name=config.name, tags={**tags, "level": "config"}) as parent:
        mlflow.log_params({**shared_params, "seeds": ",".join(map(str, seeds))})
        for seed in seeds:
            run = run_config(config, seed, bench)
            per_seed.append(run.metrics)
            with mlflow.start_run(run_name=f"{config.name}__s{seed}", nested=True,
                                  tags={**tags, "level": "seed"}):
                mlflow.log_params({**shared_params, "seed": seed})
                mlflow.log_metrics(_finite(run.metrics))
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    (tmp / "failures.jsonl").write_text(failures_jsonl(run.results))
                    (tmp / "failure_breakdown.json").write_text(
                        json.dumps(failure_breakdown(run.results), indent=2, sort_keys=True))
                    mlflow.log_artifacts(str(tmp))
        agg = {}
        for name in HEADLINE_METRICS:
            values = np.array([m[name] for m in per_seed], dtype=float)
            agg[f"{name}_mean"] = float(np.nanmean(values))
            agg[f"{name}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        mlflow.log_metrics(_finite(agg))
        return parent.info.run_id
