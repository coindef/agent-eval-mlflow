from agent_eval_mlflow.agent import GUARDRAILS, Agent, FinalAnswer, LatencyModel, ToolCall
from agent_eval_mlflow.runner import SEEDS, Benchmark, RunConfig, run_config
from agent_eval_mlflow.tools import Perturbations, ToolSandbox


class ScriptedModel:
    """Emits one malformed unit_convert call, then answers with the tool output."""

    name = "scripted"

    def next_action(self, state):
        if not state.history:
            return ToolCall("unit_convert", {"value": "410", "from_unit": "kilograms",
                                             "to_unit": "lb"}, "convert")
        last = state.history[-1]
        return FinalAnswer(str(last.result) if last.ok else None)


def _agent(guardrail):
    bench = Benchmark(case_ids=[])
    sandbox = ToolSandbox(bench.index, bench.records, "bm25", 11, Perturbations(0.0, 0.0, 0.0))
    return Agent(ScriptedModel(), sandbox, GUARDRAILS[guardrail], 4, LatencyModel())


def test_baseline_sends_malformed_call_and_gets_validation_error():
    ep = _agent("baseline").run("x", "Convert 410 kilograms to pounds")
    assert ep.invocations[0]["valid"] is False
    assert ep.invocations[0]["error_kind"] == "validation"
    assert ep.answer.abstained


def test_schema_retry_repairs_call_before_execution():
    ep = _agent("schema_retry").run("x", "Convert 410 kilograms to pounds")
    call = ep.invocations[0]
    assert call["valid"] is True and call["ok"] is True and call["repairs"] == 1
    assert call["args"] == {"value": 410.0, "from_unit": "kg", "to_unit": "lb"}
    # 410 kg = 903.89 lb
    assert "903.8" in ep.answer.text


def test_grounding_gate_abstains_on_unanswerable_question_that_retry_answers():
    bench = Benchmark(case_ids=["ab-01"])  # "Who leads the Quantum Sensing team?" (no such team)
    retry = run_config(RunConfig("bm25", "schema_retry", 2), 11, bench).results[0]
    gated = run_config(RunConfig("bm25", "schema_retry_grounding_gate", 2), 11, bench).results[0]
    assert not retry.abstained and not retry.passed
    assert gated.abstained and gated.passed


def test_same_seed_is_deterministic_and_seeds_change_outcomes():
    bench = Benchmark()
    config = RunConfig("hybrid_rrf", "baseline", 4)
    a = run_config(config, 11, bench).metrics
    b = run_config(config, 11, bench).metrics
    assert a == b
    pass_rates = {run_config(config, s, bench).metrics["task_pass_rate"] for s in SEEDS}
    assert len(pass_rates) > 1


def test_benchmark_has_all_four_categories():
    cats = {c.category for c in Benchmark().cases}
    assert cats == {"multi_hop", "tool_selection", "schema_validation", "abstention"}
