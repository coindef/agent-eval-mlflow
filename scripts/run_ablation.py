"""Run the full ablation grid (27 configs x 5 seeds) and log every run to MLflow."""

import argparse
import time

from agent_eval_mlflow.runner import SEEDS, Benchmark, ablation_grid
from agent_eval_mlflow.tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, log_config_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    bench = Benchmark()
    configs = ablation_grid()
    start = time.time()
    for i, config in enumerate(configs, 1):
        run_id = log_config_sweep(config, args.seeds, bench, args.tracking_uri, args.experiment)
        print(f"[{i}/{len(configs)}] {config.name} parent_run={run_id}")
    print(f"logged {len(configs)} parent runs x {len(args.seeds)} seeds "
          f"to {args.tracking_uri} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
