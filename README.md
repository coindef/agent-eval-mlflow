# agent-eval-mlflow

An offline benchmark for a tool-using RAG agent, with every run tracked in MLflow. It runs a 27-config ablation (retriever x guardrail x top_k) with 5 seeds each, 135 seed runs in total, on CPU in about 10 seconds. No API keys or network access are needed.

## What is in the repo

- `data/corpus.jsonl` has 53 short documents about a fictional robotics company (teams, products, incidents, policies). 9 of them have `"status": "superseded"` and carry outdated facts that conflict with a current doc.
- `data/records.jsonl` has 19 rows for the `lookup_record` tool, including duplicate keys with inactive status.
- `data/eval_cases.jsonl` has 60 hand-written cases: 16 multi_hop (the answer needs 2 docs), 16 tool_selection, 13 schema_validation and 15 abstention (the corpus does not contain the answer).
- `src/agent_eval_mlflow/` has the code:
  - `retrieval.py`: BM25 (k1=1.5, b=0.75), a char 3-5 gram TF-IDF cosine retriever (`char_ngram_dense`), and Reciprocal Rank Fusion of the two (`hybrid_rrf`, k=60). All written with numpy, no search library.
  - `tools.py`: a tool sandbox with `search_docs`, `lookup_record`, `calculator` and `unit_convert`, a JSON-schema subset validator, a deterministic argument repair function, and seeded perturbations (8% transient timeouts, 30% of lookups return numbers as formatted strings, 30% of lookups add a stale duplicate row).
  - `agent.py`: the agent loop, the `Model` and `Judge` protocols, the default `HeuristicModel`, the three guardrails and a simulated latency model.
  - `runner.py`, `metrics.py`: run one config for one seed and compute metrics and failure labels.
  - `tracking.py`: MLflow logging. `summary.py`: aggregation used by the report script.
- `scripts/run_ablation.py` runs the grid. `scripts/summarize_mlflow.py` reads it back with `mlflow.search_runs` and writes `results/ablation_summary.md` and `results/ablation_summary.json`.

The default model is a rule-based extractive policy, not an LLM. It is seeded and makes mistakes on purpose: 15% of tool calls get one schema error (a dropped required field, a number as a string, or an enum alias such as "kilograms"), it copies messy numeric strings into tool arguments half the time, and for each extra candidate doc in context it has a 10% chance of answering from that doc instead of the top one. Its baseline citation rule cites the evidence doc plus every retrieved doc that shares 2 or more query terms. Any object with a `next_action(state)` method can replace it, for example an LLM wrapper.

### Guardrails

- `baseline` sends each tool call once and returns the model's first answer.
- `schema_retry` validates arguments before each call. It first tries the deterministic repair, then asks the model again, up to 2 retries. It also retries transient timeouts up to 2 times.
- `schema_retry_grounding_gate` adds a check on the final answer. Each piece of evidence must come from a non-superseded doc, the answer value must appear in the cited sentence, and the question's subject must appear in the doc. Record rows must match the key and be active. If a check fails, the agent re-retrieves with a rewritten query (top_k doubled, max 20) and tries again, for up to 3 rounds, then abstains. A passing answer cites only the docs its evidence came from.

### MLflow layout

Each config is a parent run named like `hybrid_rrf__schema_retry__k4`. Each seed is a nested child run (`mlflow.start_run(nested=True)`) with:

- params: `retriever`, `guardrail`, `top_k`, `seed`, `max_retries`, `n_cases`, the model's noise rates and the latency model constants
- tags: `config`, `level=seed`, `model`, and `data_fingerprint` (a SHA-256 prefix of the three data files)
- metrics: the headline metrics below plus `abstain_recall`, `false_abstention_rate`, `answered_rate`, `avg_citations` and pass rate per category
- artifacts: `failures.jsonl` (one row per failed case with the tool calls, retrieved docs and a failure label) and `failure_breakdown.json` (failure counts by label and by category)

The parent run logs `<metric>_mean` and `<metric>_std` across its seeds. The tracking URI defaults to `sqlite:///mlflow.db`.

### Metrics

- `task_pass_rate`: share of cases passed. An answerable case passes if the agent answers, used the expected tool, and the judge accepts the answer (normalized exact match; numbers match within 0.01 absolute or 0.01% relative). An abstention case passes if the agent abstains.
- `grounding_precision`: over answered cases, the share of cited doc ids that are in the case's gold doc ids.
- `tool_schema_valid_rate`: share of model-issued tool calls whose final arguments pass schema validation.
- `abstention_accuracy`: share of all cases where "the agent abstained" equals "the case is unanswerable". It penalizes both missed and false abstentions.
- `avg_steps`: mean number of steps (model calls, tool calls, repairs and gate checks) per case.
- `p95_latency_ms`: 95th percentile of per-case latency from the simulated latency model. Each step has a base cost (180 ms per model call plus 20 ms per doc or row in context, retrieval cost by retriever plus 4 ms per requested result, 400 ms per timeout) with seeded lognormal jitter. No wall-clock time is measured, so this number is reproducible but is not a real latency.

## Reproduce

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/run_ablation.py
.venv/bin/python scripts/summarize_mlflow.py
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`run_ablation.py` appends to the experiment, and `summarize_mlflow.py` exits with an error if it finds the same (config, seed) twice. Delete `mlflow.db` and `mlruns/` before a second full run, or pass `--tracking-uri` and `--experiment`. Both scripts accept `--seeds`, `--tracking-uri` and `--experiment`; the summarizer also takes `--out-dir`.

## Results

Everything below is copied from `results/ablation_summary.md`, which `summarize_mlflow.py` generated from the 135 seed runs in MLflow. Cells are mean ± sample std over seeds 11, 22, 33, 44 and 55.

Generated by `scripts/summarize_mlflow.py` from 135 MLflow seed runs (experiment `agent-eval-ablation`, seeds 11, 22, 33, 44, 55). Cells are mean ± sample std across seeds. p95_latency_ms is simulated.

### All configs

| retriever | guardrail | top_k | task_pass_rate | grounding_precision | tool_schema_valid_rate | abstention_accuracy | avg_steps | p95_latency_ms | n_runs |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | baseline | 2 | 0.503 ± 0.062 | 0.407 ± 0.022 | 0.780 ± 0.014 | 0.557 ± 0.092 | 3.45 ± 0.04 | 778 ± 31 | 5 |
| bm25 | baseline | 4 | 0.443 ± 0.069 | 0.224 ± 0.015 | 0.780 ± 0.012 | 0.523 ± 0.097 | 3.51 ± 0.06 | 882 ± 41 | 5 |
| bm25 | baseline | 8 | 0.417 ± 0.049 | 0.145 ± 0.007 | 0.782 ± 0.017 | 0.520 ± 0.094 | 3.51 ± 0.06 | 1136 ± 56 | 5 |
| char_ngram_dense | baseline | 2 | 0.503 ± 0.073 | 0.409 ± 0.033 | 0.777 ± 0.015 | 0.550 ± 0.099 | 3.51 ± 0.06 | 797 ± 42 | 5 |
| char_ngram_dense | baseline | 4 | 0.443 ± 0.086 | 0.230 ± 0.021 | 0.780 ± 0.012 | 0.510 ± 0.095 | 3.51 ± 0.06 | 915 ± 52 | 5 |
| char_ngram_dense | baseline | 8 | 0.420 ± 0.059 | 0.153 ± 0.006 | 0.785 ± 0.014 | 0.513 ± 0.097 | 3.51 ± 0.06 | 1197 ± 78 | 5 |
| hybrid_rrf | baseline | 2 | 0.513 ± 0.068 | 0.410 ± 0.027 | 0.777 ± 0.015 | 0.567 ± 0.099 | 3.51 ± 0.06 | 810 ± 37 | 5 |
| hybrid_rrf | baseline | 4 | 0.433 ± 0.081 | 0.217 ± 0.014 | 0.780 ± 0.012 | 0.513 ± 0.101 | 3.51 ± 0.06 | 930 ± 51 | 5 |
| hybrid_rrf | baseline | 8 | 0.407 ± 0.049 | 0.142 ± 0.004 | 0.785 ± 0.014 | 0.510 ± 0.098 | 3.51 ± 0.06 | 1217 ± 80 | 5 |
| bm25 | schema_retry | 2 | 0.703 ± 0.027 | 0.431 ± 0.013 | 1.000 ± 0.000 | 0.780 ± 0.007 | 3.97 ± 0.10 | 857 ± 54 | 5 |
| bm25 | schema_retry | 4 | 0.640 ± 0.019 | 0.248 ± 0.009 | 1.000 ± 0.000 | 0.747 ± 0.007 | 4.04 ± 0.09 | 999 ± 53 | 5 |
| bm25 | schema_retry | 8 | 0.600 ± 0.026 | 0.160 ± 0.008 | 1.000 ± 0.000 | 0.747 ± 0.007 | 4.05 ± 0.10 | 1301 ± 67 | 5 |
| char_ngram_dense | schema_retry | 2 | 0.713 ± 0.027 | 0.437 ± 0.009 | 1.000 ± 0.000 | 0.780 ± 0.007 | 4.05 ± 0.08 | 907 ± 34 | 5 |
| char_ngram_dense | schema_retry | 4 | 0.647 ± 0.027 | 0.254 ± 0.005 | 1.000 ± 0.000 | 0.733 ± 0.000 | 4.04 ± 0.08 | 1054 ± 25 | 5 |
| char_ngram_dense | schema_retry | 8 | 0.597 ± 0.032 | 0.164 ± 0.005 | 1.000 ± 0.000 | 0.733 ± 0.000 | 4.04 ± 0.08 | 1374 ± 65 | 5 |
| hybrid_rrf | schema_retry | 2 | 0.720 ± 0.027 | 0.433 ± 0.011 | 1.000 ± 0.000 | 0.797 ± 0.007 | 4.05 ± 0.09 | 914 ± 52 | 5 |
| hybrid_rrf | schema_retry | 4 | 0.623 ± 0.025 | 0.238 ± 0.009 | 1.000 ± 0.000 | 0.733 ± 0.000 | 4.05 ± 0.09 | 1056 ± 49 | 5 |
| hybrid_rrf | schema_retry | 8 | 0.573 ± 0.037 | 0.152 ± 0.007 | 1.000 ± 0.000 | 0.733 ± 0.000 | 4.05 ± 0.09 | 1393 ± 52 | 5 |
| bm25 | schema_retry_grounding_gate | 2 | 0.897 ± 0.007 | 0.940 ± 0.002 | 1.000 ± 0.000 | 0.897 ± 0.007 | 5.93 ± 0.09 | 1390 ± 248 | 5 |
| bm25 | schema_retry_grounding_gate | 4 | 0.917 ± 0.012 | 0.944 ± 0.002 | 1.000 ± 0.000 | 0.917 ± 0.012 | 6.66 ± 0.12 | 2170 ± 298 | 5 |
| bm25 | schema_retry_grounding_gate | 8 | 0.893 ± 0.028 | 0.934 ± 0.014 | 1.000 ± 0.000 | 0.897 ± 0.027 | 6.95 ± 0.11 | 3501 ± 735 | 5 |
| char_ngram_dense | schema_retry_grounding_gate | 2 | 0.930 ± 0.007 | 0.947 ± 0.001 | 1.000 ± 0.000 | 0.930 ± 0.007 | 6.13 ± 0.09 | 1421 ± 148 | 5 |
| char_ngram_dense | schema_retry_grounding_gate | 4 | 0.917 ± 0.012 | 0.939 ± 0.013 | 1.000 ± 0.000 | 0.920 ± 0.007 | 6.70 ± 0.17 | 2220 ± 321 | 5 |
| char_ngram_dense | schema_retry_grounding_gate | 8 | 0.903 ± 0.025 | 0.941 ± 0.005 | 1.000 ± 0.000 | 0.903 ± 0.025 | 7.01 ± 0.13 | 3632 ± 785 | 5 |
| hybrid_rrf | schema_retry_grounding_gate | 2 | 0.930 ± 0.007 | 0.947 ± 0.001 | 1.000 ± 0.000 | 0.930 ± 0.007 | 6.15 ± 0.09 | 1633 ± 208 | 5 |
| hybrid_rrf | schema_retry_grounding_gate | 4 | 0.923 ± 0.009 | 0.946 ± 0.002 | 1.000 ± 0.000 | 0.923 ± 0.009 | 6.80 ± 0.17 | 2384 ± 359 | 5 |
| hybrid_rrf | schema_retry_grounding_gate | 8 | 0.893 ± 0.035 | 0.939 ± 0.009 | 1.000 ± 0.000 | 0.893 ± 0.035 | 7.18 ± 0.10 | 4168 ± 528 | 5 |

### By retriever (all runs pooled over the other two factors)

Std here is across all pooled seed runs, so it includes config-to-config spread.

| retriever | task_pass_rate | grounding_precision | tool_schema_valid_rate | abstention_accuracy | avg_steps | p95_latency_ms | n_runs |
|---|---|---|---|---|---|---|---|
| bm25 | 0.668 ± 0.192 | 0.493 ± 0.333 | 0.927 ± 0.105 | 0.731 ± 0.163 | 4.68 ± 1.36 | 1446 ± 875 | 45 |
| char_ngram_dense | 0.675 ± 0.199 | 0.497 ± 0.332 | 0.927 ± 0.105 | 0.730 ± 0.172 | 4.72 ± 1.39 | 1502 ± 903 | 45 |
| hybrid_rrf | 0.669 ± 0.202 | 0.492 ± 0.337 | 0.927 ± 0.105 | 0.733 ± 0.170 | 4.76 ± 1.44 | 1612 ± 1045 | 45 |

### By guardrail (all runs pooled over the other two factors)

Std here is across all pooled seed runs, so it includes config-to-config spread.

| guardrail | task_pass_rate | grounding_precision | tool_schema_valid_rate | abstention_accuracy | avg_steps | p95_latency_ms | n_runs |
|---|---|---|---|---|---|---|---|
| baseline | 0.454 ± 0.073 | 0.260 ± 0.113 | 0.781 ± 0.013 | 0.529 ± 0.090 | 3.51 ± 0.05 | 962 ± 174 | 45 |
| schema_retry | 0.646 ± 0.058 | 0.280 ± 0.116 | 1.000 ± 0.000 | 0.754 ± 0.024 | 4.04 ± 0.08 | 1095 ± 204 | 45 |
| schema_retry_grounding_gate | 0.911 ± 0.022 | 0.942 ± 0.008 | 1.000 ± 0.000 | 0.912 ± 0.022 | 6.61 ± 0.43 | 2502 ± 1062 | 45 |

### By top_k (all runs pooled over the other two factors)

Std here is across all pooled seed runs, so it includes config-to-config spread.

| top_k | task_pass_rate | grounding_precision | tool_schema_valid_rate | abstention_accuracy | avg_steps | p95_latency_ms | n_runs |
|---|---|---|---|---|---|---|---|
| 2 | 0.713 ± 0.175 | 0.596 ± 0.250 | 0.926 ± 0.106 | 0.754 ± 0.160 | 4.53 ± 1.13 | 1056 ± 333 | 45 |
| 4 | 0.665 ± 0.204 | 0.471 ± 0.338 | 0.927 ± 0.105 | 0.724 ± 0.175 | 4.76 ± 1.42 | 1401 ± 642 | 45 |
| 8 | 0.634 ± 0.205 | 0.415 ± 0.375 | 0.928 ± 0.103 | 0.717 ± 0.168 | 4.87 ± 1.57 | 2102 ± 1259 | 45 |

## What the table shows

- The guardrail is the factor that moves the numbers. Pooled over retrievers and top_k, task_pass_rate goes from 0.454 (`baseline`) to 0.646 (`schema_retry`) to 0.911 (`schema_retry_grounding_gate`). The cost is more steps (3.51, 4.04, 6.61) and higher simulated p95 latency (962, 1095, 2502 ms).
- `schema_retry` brings tool_schema_valid_rate from 0.777 to 0.785 under `baseline` to 1.000 in all 90 seed runs that use it.
- Grounding precision stays low without the gate (0.142 to 0.437) and falls as top_k grows, because the baseline citation rule cites every retrieved doc that shares 2 query terms. The gate lifts it to 0.934 to 0.947. Two changes contribute: the gate cites only the docs its evidence came from, and it rejects superseded docs. This sweep does not separate the two effects. The gate reads the corpus's `status` field, so it has metadata a real agent would not always have.
- Without the gate, more context hurts. For `baseline` and `schema_retry`, going from top_k 2 to 8 drops task_pass_rate by 0.083 to 0.147 for every retriever, because the model is more likely to answer from a distractor doc. With the gate, top_k 8 is the worst of the three settings for every retriever (0.893 to 0.903), and its simulated p95 latency is 2.5 to 2.6 times that of top_k 2.
- The retriever choice barely matters on this corpus. Pooled task_pass_rate is 0.668 (bm25), 0.675 (char_ngram_dense) and 0.669 (hybrid_rrf), well inside the seed spread. With the gate at top_k 2, char_ngram_dense and hybrid_rrf both reach 0.930 ± 0.007 against 0.897 ± 0.007 for bm25. hybrid_rrf has the highest simulated latency because it runs both retrievers.
- The best config by task_pass_rate is a tie between `char_ngram_dense` and `hybrid_rrf` with `schema_retry_grounding_gate` at top_k 2 (0.930 ± 0.007). Of the two, char_ngram_dense has lower simulated p95 latency (1421 vs 1633 ms).

## Limitations

- The default model is a seeded heuristic, not an LLM. The results describe how the guardrails interact with its specific error modes, which are fixed constants in `HeuristicModel` (15% malformed calls, 10% distractor pick rate per extra doc). Different rates would give different numbers.
- The corpus is small (53 docs) and fictional, and the 60 cases were written for this repo. This is not an external benchmark.
- Latency is simulated from fixed per-step costs.
- Abstention cases are 15 of 60, so abstention_accuracy and task_pass_rate overlap heavily.

## Tests

`pytest` runs 16 tests: BM25 scores against a hand-computed value, RRF fusion order and scores, char n-gram matching of a misspelled query, schema validation and repair, `schema_retry` repairing a malformed call that `baseline` sends as-is, the grounding gate abstaining on an unanswerable question that `schema_retry` answers, seed determinism, summary aggregation, and an end-to-end test that logs parent and nested seed runs to a temporary SQLite tracking URI and reads the artifacts back.

## License

MIT
