"""Run one (retriever, guardrail, top_k) configuration over every eval case for one seed."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from .agent import GUARDRAILS, Agent, ExactMatchJudge, HeuristicModel, Judge, LatencyModel, Model
from .metrics import CaseResult, compute_metrics
from .retrieval import RETRIEVERS, SearchIndex, load_corpus
from .tools import Perturbations, ToolSandbox, load_records

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEEDS = (11, 22, 33, 44, 55)
TOP_KS = (2, 4, 8)


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    gold_answer: str | None
    gold_doc_ids: tuple[str, ...] = ()
    expected_tool: str | None = None


def load_cases(path: str | Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [EvalCase(**{**r, "gold_doc_ids": tuple(r.get("gold_doc_ids", ()))}) for r in rows]


class Benchmark:
    """Corpus, records and eval cases, plus the search index built once over the corpus."""

    def __init__(self, data_dir: str | Path = DATA_DIR, case_ids: Iterable[str] | None = None):
        data_dir = Path(data_dir)
        files = [data_dir / n for n in ("corpus.jsonl", "records.jsonl", "eval_cases.jsonl")]
        self.docs = load_corpus(files[0])
        self.records = load_records(files[1])
        self.cases = load_cases(files[2])
        if case_ids is not None:
            wanted = set(case_ids)
            self.cases = [c for c in self.cases if c.id in wanted]
        self.index = SearchIndex(self.docs)
        digest = hashlib.sha256()
        for f in files:
            digest.update(f.read_bytes())
        self.fingerprint = digest.hexdigest()[:12]


@dataclass(frozen=True)
class RunConfig:
    retriever: str
    guardrail: str
    top_k: int

    def __post_init__(self):
        if self.retriever not in RETRIEVERS:
            raise ValueError(f"unknown retriever {self.retriever!r}; choose from {RETRIEVERS}")
        if self.guardrail not in GUARDRAILS:
            raise ValueError(f"unknown guardrail {self.guardrail!r}; choose from {tuple(GUARDRAILS)}")

    @property
    def name(self) -> str:
        return f"{self.retriever}__{self.guardrail}__k{self.top_k}"

    def params(self) -> dict[str, str | int]:
        return {"retriever": self.retriever, "guardrail": self.guardrail, "top_k": self.top_k}


def ablation_grid(retrievers: Sequence[str] = RETRIEVERS,
                  guardrails: Sequence[str] = tuple(GUARDRAILS),
                  top_ks: Sequence[int] = TOP_KS) -> list[RunConfig]:
    return [RunConfig(r, g, k) for r, g, k in product(retrievers, guardrails, top_ks)]


@dataclass
class SeedRun:
    config: RunConfig
    seed: int
    results: list[CaseResult]
    metrics: dict[str, float] = field(default_factory=dict)


def run_config(config: RunConfig, seed: int, bench: Benchmark,
               model_factory: Callable[[int], Model] = HeuristicModel,
               judge: Judge | None = None,
               perturbations: Perturbations = Perturbations(),
               latency: LatencyModel = LatencyModel()) -> SeedRun:
    judge = judge or ExactMatchJudge()
    sandbox = ToolSandbox(bench.index, bench.records, config.retriever, seed, perturbations)
    agent = Agent(model_factory(seed), sandbox, GUARDRAILS[config.guardrail], config.top_k, latency)
    results = []
    for case in bench.cases:
        ep = agent.run(case.id, case.question)
        answer = ep.answer
        used = {c["tool"] for c in ep.invocations if c["ok"] and not c["system"]}
        if case.gold_answer is None:
            passed = answer.abstained
        else:
            passed = (not answer.abstained
                      and (case.expected_tool is None or case.expected_tool in used)
                      and judge.is_correct(case.question, case.gold_answer, answer.text))
        results.append(CaseResult(
            case_id=case.id, category=case.category, question=case.question,
            gold_answer=case.gold_answer, predicted=answer.text, passed=bool(passed),
            gold_doc_ids=list(case.gold_doc_ids), citations=list(answer.citations),
            evidence_ids=[e.source_id for e in answer.evidence if e.kind == "doc"],
            retrieved_doc_ids=list(dict.fromkeys(ep.retrieved)), tool_calls=ep.invocations,
            expected_tool=case.expected_tool, steps=len(ep.steps),
            latency_ms=round(ep.latency_ms, 3), forced_abstention=ep.forced_abstention,
        ))
    return SeedRun(config, seed, results, compute_metrics(results))
