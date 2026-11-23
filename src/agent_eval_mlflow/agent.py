"""Agent loop, Model/Judge protocols, the default heuristic model, and the guardrails.

The default model is a deterministic, seeded, rule-based policy. It is deliberately
imperfect: it can emit malformed tool arguments, copy messy numeric strings verbatim, and
pick an answer span from a distractor document. Any object implementing `Model` (for
example an LLM wrapper) can replace it without changing the loop or the guardrails.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Protocol

import numpy as np

from .retrieval import tokenize
from .tools import (
    ALIASES,
    TOOL_SCHEMAS,
    ToolSandbox,
    coerce_number,
    keyed_rng,
    repair_args,
    validate,
)

# --------------------------------------------------------------------------- data types


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    note: str = ""  # model-side label of the call's purpose, e.g. "hop1|Atlas WMS"


@dataclass
class Evidence:
    """One claim the answer depends on, plus the span that is supposed to support it."""

    hop: int
    kind: str  # "doc" | "row" | "tool"
    source_id: str  # doc id, record row id, or tool name
    value: str  # extracted value
    span: str  # supporting sentence, row key, or tool output
    subject: str  # entity the claim is about
    title: str = ""  # doc title or record key
    status: str = ""  # doc status (current/superseded) or row status (active/...)
    relation: str = ""  # relation phrase, used to rewrite the query on re-retrieval


@dataclass
class FinalAnswer:
    text: str | None  # None means the agent abstains
    citations: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.text is None


@dataclass
class Observation:
    call: ToolCall
    ok: bool
    result: Any = None
    error: str | None = None
    error_kind: str | None = None


@dataclass
class AgentState:
    case_id: str
    question: str
    tool_schemas: Mapping[str, dict]
    top_k: int
    history: list[Observation] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)  # guardrail feedback for the model
    rejected: set[tuple[int, str]] = field(default_factory=set)  # (hop, source id) gate rejects
    attempt: int = 0  # re-ask counter for the pending tool call


Action = ToolCall | FinalAnswer


class Model(Protocol):
    name: str

    def next_action(self, state: AgentState) -> Action: ...


class Judge(Protocol):
    name: str

    def is_correct(self, question: str, gold: str, predicted: str) -> bool: ...


# --------------------------------------------------------------------------- judge


def normalize_answer(text: str) -> str:
    t = text.lower().replace("−", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"[^a-z0-9.\- ]+", " ", t)
    t = re.sub(r"\b(?:a|an|the)\b", " ", t)
    return " ".join(t.split()).strip(" .")


def _split_number(text: str) -> tuple[float | None, str]:
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?: ([a-z][a-z ]*))?", text)
    return (float(m.group(1)), m.group(2) or "") if m else (None, "")


class ExactMatchJudge:
    """Normalised exact match; numbers match within 0.01 absolute or 0.01% relative."""

    name = "exact_match_numeric"

    def is_correct(self, question: str, gold: str, predicted: str) -> bool:
        g, p = normalize_answer(gold), normalize_answer(predicted)
        if g == p:
            return True
        (gn, gu), (pn, pu) = _split_number(g), _split_number(p)
        if gn is None or pn is None or (gu and pu and gu != pu):
            return False
        return abs(gn - pn) <= max(0.01, 1e-4 * abs(gn))


# --------------------------------------------------------------------------- question parsing


@dataclass(frozen=True)
class Relation:
    name: str
    cue: re.Pattern  # matched against the lowercased question
    pattern: re.Pattern  # extraction pattern over document sentences, named group "v"
    phrase: str  # canonical phrase, used to rewrite queries


def _rel(name: str, cue: str, pattern: str, phrase: str) -> Relation:
    return Relation(name, re.compile(cue), re.compile(pattern), phrase)


_PERSON = r"[A-Z][a-z]+ [A-Z][a-z]+"
_TEAM = r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*"
RELATIONS: tuple[Relation, ...] = (
    _rel("lead", r"\bleads\b|\blead of\b|\bhead of\b|\bled by\b",
         rf"\bis led by (?P<v>{_PERSON})", "is led by"),
    _rel("office", r"\bwhich office\b|\boffice where\b|\bbased\b",
         r"\bis based in the (?P<v>[A-Z][a-z]+) office\b", "is based in the office"),
    _rel("oncall_rotation", r"\bon-call rotation\b|\brotation\b",
         r"\bon-call rotation lasts (?P<v>\d+ days)\b", "on-call rotation lasts"),
    _rel("manager", r"\breports? to\b", rf"\breports to (?P<v>{_PERSON})", "reports to"),
    _rel("joined_year", r"\bjoin(?:ed)?\b", r"\bjoined Quillmoor in (?P<v>\d{4})\b",
         "joined Quillmoor in"),
    _rel("address", r"\baddress\b", r"\bis located at (?P<v>[^.]+)\.", "is located at"),
    _rel("opened_year", r"\bopen(?:ed)?\b", r"\bopened in (?P<v>\d{4})\b", "opened in"),
    _rel("site_manager", r"\bsite manager\b", rf"\bsite manager is (?P<v>{_PERSON})",
         "site manager is"),
    _rel("owner_team", r"\bowns\b|\bowned by\b|\bowner of\b",
         rf"\bis owned by the (?P<v>{_TEAM}) team\b", "is owned by the team"),
    _rel("launch_year", r"\blaunch(?:ed)?\b", r"\blaunched in (?P<v>\d{4})\b", "launched in"),
    _rel("max_payload", r"\bpayload\b|\bcarry\b", r"\bmaximum payload of (?P<v>[\d,.]+ kg)\b",
         "maximum payload of"),
    _rel("affected_product", r"\baffected by\b|\bproduct affected\b",
         r"\baffected the (?P<v>[A-Z][A-Za-z0-9]*(?:[ -][A-Z0-9][A-Za-z0-9-]*)*)", "affected the"),
    _rel("root_cause", r"\broot cause\b", r"\broot cause was (?P<v>[^.]+)\.",
         "the root cause was"),
    _rel("resolver_team", r"\bresolved\b", rf"\bwas resolved by the (?P<v>{_TEAM}) team\b",
         "was resolved by the team"),
    _rel("per_diem", r"\bper diem\b", r"\bper diem is (?P<v>[\d,]+ USD)\b", "per diem is"),
    _rel("hotel_limit", r"\bhotel\b", r"\bstays above (?P<v>[\d,]+ USD) per night\b",
         "hotel stays above per night"),
    _rel("expense_deadline", r"\bexpense reports?\b", r"\bfiled within (?P<v>\d+ days)\b",
         "expense reports filed within"),
    _rel("receipt_threshold", r"\breceipts?\b", r"\bany item above (?P<v>[\d,]+ USD)\b",
         "receipts required above"),
    _rel("oncall_stipend", r"\bstipend\b", r"\bstipend of (?P<v>[\d,]+ USD) per week\b",
         "stipend per week"),
    _rel("page_ack", r"\backnowledged?\b", r"\backnowledged within (?P<v>\d+ minutes)\b",
         "acknowledged within"),
    _rel("remote_days", r"\bremotely\b|\bremote\b", r"\bremotely up to (?P<v>\d+ days) per week\b",
         "work remotely up to"),
    _rel("retention", r"\bretained\b|\bretention\b", r"\bretained for (?P<v>\d+ months)\b",
         "retained for"),
    _rel("return_window", r"\breturn\b", r"\breturn unused hardware within (?P<v>\d+ days)\b",
         "return unused hardware within"),
    _rel("refund_time", r"\brefunds?\b", r"\bissued within (?P<v>\d+ business days)\b",
         "refunds issued within"),
)

_UNIT_WORDS = (r"°c|°f|kilograms?|kgs?|pounds?|lbs?|kilomet(?:er|re)s?|km|miles?|met(?:er|re)s?"
               r"|feet|foot|ft|lit(?:er|re)s?|gallons?|gal|celsius|fahrenheit|grams?")
_UNIT_RE = re.compile(rf"(?<![a-z])({_UNIT_WORDS})(?![a-z])")
_CONVERT_RE = re.compile(rf"\bconvert\b|\bhow many (?:{_UNIT_WORDS})\b|\bin (?:{_UNIT_WORDS})\b")
_CALC_RE = re.compile(r"\d\s*(?:[-+*/x×]|plus|minus|times|multiplied by|divided by)\s*\(?\s*\d"
                      r"|\bpercent\b")
_LOOKUP_RE = re.compile(r"\bemployee (?:id|number|record)\b|\bcost cent(?:er|re)\b|\bsku\b"
                        r"|\bin stock\b|\blist price\b")
_NUMBER_RE = re.compile(r"(?<![\w-])[-−]?\d[\d,]*(?:\.\d+)?(?![\w-])")
_DISCOUNT_RE = re.compile(r"after an? (\d+(?:\.\d+)?) percent discount")
_PERCENT_OF_RE = re.compile(r"([-−]?\d[\d,]*(?:\.\d+)?) percent of ([-−]?\d[\d,]*(?:\.\d+)?)")
_WORD_OPS = (("multiplied by", "*"), ("divided by", "/"), ("times", "*"), ("plus", "+"),
             ("minus", "-"))
FIELD_CUES: tuple[tuple[str, str], ...] = (
    ("employee_id", r"\bemployee (?:id|number)\b"),
    ("cost_center", r"\bcost cent(?:er|re)\b"),
    ("sku", r"\bsku\b"),
    ("units_in_stock", r"\bin stock\b"),
    ("list_price_usd", r"\bprice\b"),
    ("weight_kg", r"\bweigh(?:t|s)?\b"),
    ("office", r"\boffice\b"),
)
FIELD_TABLE = {"employee_id": "employees", "cost_center": "employees", "office": "employees",
               "sku": "products", "units_in_stock": "products", "list_price_usd": "products",
               "weight_kg": "products"}
FIELD_UNIT = {"weight_kg": "kg"}
_NOT_NAMES = frozenset(["who", "what", "which", "when", "where", "how", "in", "within", "convert", "look", "is", "the", "quillmoor", "id", "sku", "usd"])  # noqa: E501
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


@dataclass(frozen=True)
class Plan:
    route: str  # tool the question is routed to
    target: Relation | None = None
    bridge: Relation | None = None  # first hop of a two-hop question
    target_cue: str = ""
    subject: str = ""
    field: str | None = None  # record field, when a lookup is needed
    table: str | None = None


def _is_name_word(word: str) -> bool:
    has_digit = any(c.isdigit() for c in word)
    named = word[0].isupper() or (has_digit and any(c.isalpha() for c in word))
    return named and word.lower() not in _NOT_NAMES


def proper_spans(question: str) -> list[tuple[int, str]]:
    """Maximal runs of capitalised words or codes (PR-200, INC-4471), as (start, text)."""
    spans: list[tuple[int, int, list[str]]] = []
    for m in _WORD_RE.finditer(question):
        if not _is_name_word(m.group()):
            continue
        if spans and question[spans[-1][1] : m.start()] == " ":
            start, _, words = spans[-1]
            spans[-1] = (start, m.end(), [*words, m.group()])
        else:
            spans.append((m.start(), m.end(), [m.group()]))
    return [(s, " ".join(words)) for s, _, words in spans]


def _find_cues(q: str) -> list[tuple[int, int, Relation, str]]:
    found = [(m.start(), m.end(), rel, m.group()) for rel in RELATIONS for m in rel.cue.finditer(q)]
    found.sort(key=lambda c: (c[0] - c[1], c[0]))  # longest first, then leftmost
    kept: list[tuple[int, int, Relation, str]] = []
    for c in found:
        if all(c[1] <= k[0] or c[0] >= k[1] for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c[0])


def _route(q: str) -> str:
    if _CONVERT_RE.search(q):
        return "unit_convert"
    if _CALC_RE.search(q):
        return "calculator"
    if _LOOKUP_RE.search(q):
        return "lookup_record"
    return "search_docs"


@cache
def parse_question(question: str) -> Plan:
    q = question.lower()
    route = _route(q)
    spans = proper_spans(question)
    span = max(spans, key=lambda s: (len(s[1].split()), s[0])) if spans else None
    if route != "search_docs":
        needs_record = (route == "unit_convert" and not _NUMBER_RE.search(question)) or (
            route == "calculator" and _DISCOUNT_RE.search(q) is not None) or route == "lookup_record"
        fld = next((f for f, cue in FIELD_CUES if re.search(cue, q)), None) if needs_record else None
        return Plan(route, subject=span[1] if span else "", field=fld,
                    table=FIELD_TABLE.get(fld) if fld else None)
    cues = _find_cues(q)
    if not cues:
        return Plan(route)
    if span is not None:
        subject, subject_start = span[1], span[0]
    else:  # no named entity: use the content words after the last cue
        tail = q[cues[-1][1] :]
        cue_words = {w for c in cues for w in tokenize(c[3])}
        terms = [t for t in tokenize(tail) or tokenize(q) if t not in cue_words]
        subject, subject_start = " ".join(terms), cues[-1][1]
    distinct = list({c[2].name: c for c in cues}.values())
    before = [c for c in cues if c[1] <= subject_start]
    if len(distinct) >= 2 and before:
        bridge = before[-1]
        target = next(c for c in cues if c[2].name != bridge[2].name)
        return Plan(route, target[2], bridge[2], target[3], subject)
    return Plan(route, cues[0][2], None, cues[0][3], subject)


# --------------------------------------------------------------------------- reading


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


@dataclass(frozen=True)
class Candidate:
    doc: Mapping[str, Any]
    value: str
    sentence: str


def extract_candidates(rel: Relation, docs: list[Mapping[str, Any]], query: str,
                       exclude: set[str] = frozenset()) -> list[Candidate]:
    """One candidate per doc (in rank order): the matching sentence with most query overlap."""
    q = set(tokenize(query))
    out = []
    for d in docs:
        if d["id"] in exclude:
            continue
        best: tuple[int, str, str] | None = None
        for sent in split_sentences(d["text"]):
            m = rel.pattern.search(sent)
            if m:
                score = len(q & set(tokenize(sent)))
                if best is None or score > best[0]:
                    best = (score, m.group("v").strip(), sent)
        if best:
            out.append(Candidate(d, best[1], best[2]))
    return out


def naive_citations(docs: list[Mapping[str, Any]], query: str, evidence_id: str,
                    min_shared: int = 2) -> list[str]:
    """Baseline attribution: the evidence doc plus any retrieved doc sharing >= 2 query terms."""
    q = set(tokenize(query))
    extra = [d["id"] for d in docs if d["id"] != evidence_id
             and len(q & set(tokenize(f"{d['title']} {d['text']}"))) >= min_shared]
    return [evidence_id, *extra]


def format_number(x: float) -> str:
    r = round(float(x), 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{r:.2f}".rstrip("0").rstrip(".")


_LONG_FORM = {"kg": "kilograms", "lb": "pounds", "g": "grams", "km": "kilometers", "mi": "miles",
              "m": "meters", "ft": "feet", "l": "liters", "gal": "gallons", "C": "celsius",
              "F": "fahrenheit", "employees": "employee", "products": "product"}


def malform(tool: str, args: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Apply one realistic tool-call mistake."""
    out = dict(args)
    props = TOOL_SCHEMAS[tool]["properties"]
    numeric = [k for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    enums = [k for k in out if "enum" in props.get(k, {}) and out[k] in _LONG_FORM]
    kinds = ["drop_required", "extra_property"]
    kinds += ["number_as_string"] if numeric else []
    kinds += ["enum_alias"] if enums else []
    kinds += ["word_operator"] if tool == "calculator" else []
    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "drop_required":
        required = TOOL_SCHEMAS[tool]["required"]
        out.pop(required[int(rng.integers(len(required)))], None)
    elif kind == "extra_property":
        out["verbose"] = True
    elif kind == "number_as_string":
        name = numeric[int(rng.integers(len(numeric)))]
        out[name] = str(out[name])
    elif kind == "enum_alias":
        name = enums[int(rng.integers(len(enums)))]
        out[name] = _LONG_FORM[out[name]]
    else:
        out["expression"] = out["expression"].replace("*", "x") + " ="
    return out


def _unit_mentions(q: str) -> list[tuple[int, str]]:
    return [(m.start(), ALIASES.get(m.group(1), m.group(1))) for m in _UNIT_RE.finditer(q)]


def _num(span: str, raw: bool) -> str:
    """How the model writes a number it read: verbatim if `raw`, else cleaned."""
    if raw:
        return span
    value = coerce_number(span)
    return span if value is None else format_number(value)


def build_compute_args(plan: Plan, question: str, operand: Any, raw: bool) -> dict | None:
    q = question.lower()
    if plan.route == "unit_convert":
        units = _unit_mentions(q)
        if operand is not None:
            value, from_unit = operand, FIELD_UNIT.get(plan.field or "")
            to_unit = next((u for _, u in units if u != from_unit), None)
        else:
            m = _NUMBER_RE.search(question)
            if m is None:
                return None
            value = m.group()
            after = [u for pos, u in units if pos >= m.end()]
            from_unit = after[0] if after else None
            to_unit = next((u for _, u in units if u != from_unit), None)
        if from_unit is None or to_unit is None:
            return None
        messy = isinstance(value, str) and (raw or coerce_number(value) is None)
        return {"value": value if messy else coerce_number(value), "from_unit": from_unit,
                "to_unit": to_unit}
    if operand is not None:
        m = _DISCOUNT_RE.search(q)
        if m is None:
            return None
        op = operand if raw and isinstance(operand, str) else format_number(coerce_number(operand))
        return {"expression": f"{op} * (1 - {m.group(1)} / 100)"}
    m = _PERCENT_OF_RE.search(q)
    if m:
        return {"expression": f"{_num(m.group(2), raw)} * {_num(m.group(1), raw)} / 100"}
    body = re.sub(r"^\s*what is\s+", "", q).rstrip("?. ")
    for word, sym in _WORD_OPS:
        body = body.replace(f" {word} ", f" {sym} ")
    if not raw:
        body = _NUMBER_RE.sub(lambda mm: _num(mm.group(), False), body)
    return {"expression": body}


# --------------------------------------------------------------------------- the model


def _latest(state: AgentState, tool: str, notes: tuple[str, ...]) -> Observation | None:
    """Most recent successful observation for these notes, else the latest failed one."""
    matching = [o for o in state.history if o.call.tool == tool and o.call.note in notes]
    ok = [o for o in matching if o.ok]
    return (ok or matching or [None])[-1]


@dataclass
class HeuristicModel:
    """Rule-based extractive policy with seeded imperfections.

    p_malformed: probability that a tool call contains one schema mistake.
    p_copy_raw: probability that a messy numeric string (1,250 / 410 kg / Unicode minus) is
        copied verbatim into a tool argument instead of being parsed.
    p_distract: per extra candidate doc in context, probability that the model's answer comes
        from that candidate instead of the top-ranked one (so more context means more risk).
    """

    seed: int
    p_malformed: float = 0.15
    p_copy_raw: float = 0.5
    p_distract: float = 0.10
    name: str = "heuristic-extractive-v1"

    def params(self) -> dict[str, float]:
        return {"p_malformed": self.p_malformed, "p_copy_raw": self.p_copy_raw,
                "p_distract": self.p_distract}

    def next_action(self, state: AgentState) -> Action:
        plan = parse_question(state.question)
        if plan.route == "search_docs":
            return self._answer_from_docs(plan, state)
        if plan.route == "lookup_record":
            return self._answer_from_record(plan, state)
        return self._compute(plan, state)

    def _issue(self, state: AgentState, tool: str, args: dict, note: str) -> ToolCall:
        rng = keyed_rng(self.seed, "args", state.case_id, note, state.attempt)
        if rng.random() < self.p_malformed:
            args = malform(tool, args, rng)
        return ToolCall(tool, args, note)

    def _pick(self, cands: list[Candidate], state: AgentState, hop: int) -> Candidate:
        n_rejected = sum(1 for h, _ in state.rejected if h == hop)
        rng = keyed_rng(self.seed, "pick", state.case_id, hop, n_rejected)
        others = len(cands) - 1
        if others > 0 and rng.random() < 1.0 - (1.0 - self.p_distract) ** others:
            return cands[1 + int(rng.integers(others))]
        return cands[0]

    def _answer_from_docs(self, plan: Plan, state: AgentState) -> Action:
        if plan.target is None:
            return FinalAnswer(None)
        hops = [plan.bridge, plan.target] if plan.bridge else [plan.target]
        subject, evidence, citations = plan.subject, [], []
        for hop, rel in enumerate(hops, start=1):
            obs = _latest(state, "search_docs", (f"hop{hop}|{subject}", f"recheck{hop}|{subject}"))
            if obs is None:
                query = state.question if hop == 1 else f"{subject} {plan.target_cue}"
                return self._issue(state, "search_docs", {"query": query, "top_k": state.top_k},
                                   f"hop{hop}|{subject}")
            if not obs.ok:
                return FinalAnswer(None)
            docs, query = obs.result["docs"], obs.call.args["query"]
            rejected = {sid for h, sid in state.rejected if h == hop}
            cands = extract_candidates(rel, docs, query, rejected)
            if not cands:
                return FinalAnswer(None)
            c = self._pick(cands, state, hop)
            evidence.append(Evidence(hop, "doc", c.doc["id"], c.value, c.sentence, subject,
                                     c.doc["title"], c.doc["status"], rel.phrase))
            citations += naive_citations(docs, query, c.doc["id"])
            subject = c.value
        return FinalAnswer(evidence[-1].value, list(dict.fromkeys(citations)), evidence)

    def _row(self, plan: Plan, state: AgentState) -> ToolCall | dict | None:
        if not plan.subject or not plan.field:
            return None
        note = f"lookup|{plan.subject}"
        obs = _latest(state, "lookup_record", (note,))
        if obs is None:
            args = {"table": plan.table, "key": plan.subject, "field": plan.field}
            return self._issue(state, "lookup_record", args, note)
        if not obs.ok:
            return None
        rows = [r for r in obs.result["rows"]
                if (1, r["row_id"]) not in state.rejected and r.get(plan.field) is not None]
        return rows[0] if rows else None  # naive: first row wins

    @staticmethod
    def _row_evidence(row: dict, plan: Plan) -> Evidence:
        return Evidence(1, "row", row["row_id"], str(row[plan.field]), row["key"], plan.subject,
                        row["key"], row.get("status", ""), plan.field or "")

    def _answer_from_record(self, plan: Plan, state: AgentState) -> Action:
        row = self._row(plan, state)
        if not isinstance(row, dict):
            return row if isinstance(row, ToolCall) else FinalAnswer(None)
        return FinalAnswer(str(row[plan.field]), [], [self._row_evidence(row, plan)])

    def _compute(self, plan: Plan, state: AgentState) -> Action:
        evidence: list[Evidence] = []
        operand, note = None, "compute|question"
        if plan.field:
            row = self._row(plan, state)
            if not isinstance(row, dict):
                return row if isinstance(row, ToolCall) else FinalAnswer(None)
            evidence.append(self._row_evidence(row, plan))
            operand, note = row[plan.field], f"compute|{row['row_id']}"
        obs = _latest(state, plan.route, (note,))
        if obs is None:
            raw = keyed_rng(self.seed, "copy", state.case_id, note).random() < self.p_copy_raw
            args = build_compute_args(plan, state.question, operand, raw)
            return FinalAnswer(None) if args is None else self._issue(state, plan.route, args, note)
        if not obs.ok:
            return FinalAnswer(None)
        value = obs.result["result" if plan.route == "calculator" else "value"]
        text = format_number(value)
        evidence.append(Evidence(len(evidence) + 1, "tool", plan.route, text,
                                 json.dumps(obs.result), plan.subject))
        return FinalAnswer(text, [], evidence)


# --------------------------------------------------------------------------- guardrails


@dataclass(frozen=True)
class Guardrail:
    name: str
    schema_retry: bool = False  # validate args before the call, repair, re-ask, retry
    grounding_gate: bool = False  # verify evidence spans, re-retrieve, else abstain
    max_retries: int = 2


GUARDRAILS: dict[str, Guardrail] = {
    "baseline": Guardrail("baseline"),
    "schema_retry": Guardrail("schema_retry", schema_retry=True),
    "schema_retry_grounding_gate": Guardrail("schema_retry_grounding_gate", schema_retry=True,
                                             grounding_gate=True),
}
MAX_GATE_ROUNDS = 3


def subject_supported(subject: str, evidence_text: str, min_overlap: float = 0.5) -> bool:
    """All identifier-like subject terms (containing a digit) must appear in the evidence,
    and at least `min_overlap` of the remaining subject terms."""
    terms = list(dict.fromkeys(tokenize(subject)))
    have = set(tokenize(evidence_text))
    ids = [t for t in terms if any(ch.isdigit() for ch in t)]
    if any(t not in have for t in ids):
        return False
    rest = [t for t in terms if t not in ids]
    return not rest or sum(t in have for t in rest) / len(rest) >= min_overlap


def verify_evidence(ev: Evidence) -> str | None:
    """Return a rejection reason, or None if the evidence looks like gold evidence."""
    if ev.kind == "doc":
        if ev.status == "superseded":
            return "superseded_source"
        if normalize_answer(ev.value) not in normalize_answer(ev.span):
            return "value_not_in_span"
        if not subject_supported(ev.subject, f"{ev.title} {ev.span}"):
            return "subject_mismatch"
    elif ev.kind == "row":
        if ev.title.strip().lower() != ev.subject.strip().lower():
            return "key_mismatch"
        if ev.status != "active":
            return "inactive_row"
    return None


# --------------------------------------------------------------------------- latency


@dataclass(frozen=True)
class LatencyModel:
    """Simulated, seeded latency. No wall-clock time is measured anywhere."""

    model_call_ms: float = 180.0
    context_ms_per_item: float = 20.0  # each doc or row already in context slows a model call
    retrieval_ms: Mapping[str, float] = field(
        default_factory=lambda: {"bm25": 8.0, "char_ngram_dense": 18.0, "hybrid_rrf": 28.0})
    retrieval_ms_per_result: float = 4.0  # proportional to the requested top_k
    tool_ms: Mapping[str, float] = field(
        default_factory=lambda: {"lookup_record": 30.0, "calculator": 3.0, "unit_convert": 3.0})
    rejected_call_ms: float = 5.0  # server-side validation failure
    timeout_ms: float = 400.0  # transient error
    repair_ms: float = 2.0
    gate_ms: float = 15.0
    jitter_sigma: float = 0.2  # mean-preserving lognormal jitter per step

    def params(self) -> dict[str, float]:
        return {"model_call_ms": self.model_call_ms, "context_ms_per_item": self.context_ms_per_item,
                "retrieval_ms_per_result": self.retrieval_ms_per_result,
                "timeout_ms": self.timeout_ms, "gate_ms": self.gate_ms,
                "jitter_sigma": self.jitter_sigma}


# --------------------------------------------------------------------------- the loop


@dataclass
class Episode:
    seed: int
    case_id: str
    latency: LatencyModel
    answer: FinalAnswer | None = None
    steps: list[tuple[str, float]] = field(default_factory=list)
    invocations: list[dict] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    forced_abstention: bool = False

    @property
    def latency_ms(self) -> float:
        return float(sum(ms for _, ms in self.steps))

    def add(self, kind: str, base_ms: float) -> None:
        rng = keyed_rng(self.seed, "latency", self.case_id, len(self.steps), kind)
        s = self.latency.jitter_sigma
        self.steps.append((kind, base_ms * float(np.exp(s * rng.standard_normal() - 0.5 * s * s))))


def _context_items(state: AgentState) -> int:
    n = 0
    for o in state.history:
        if o.ok and isinstance(o.result, dict):
            n += len(o.result.get("docs", ())) + len(o.result.get("rows", ()))
    return n


class Agent:
    def __init__(self, model: Model, sandbox: ToolSandbox, guardrail: Guardrail, top_k: int,
                 latency: LatencyModel, max_steps: int = 30):
        self.model, self.sandbox, self.guardrail = model, sandbox, guardrail
        self.top_k, self.latency, self.max_steps = top_k, latency, max_steps

    def _think(self, ep: Episode, state: AgentState) -> Action:
        lat = self.latency
        ep.add("model", lat.model_call_ms + lat.context_ms_per_item * _context_items(state))
        return self.model.next_action(state)

    def _execute(self, ep: Episode, state: AgentState, call: ToolCall, system: bool) -> None:
        schema = TOOL_SCHEMAS.get(call.tool)
        repairs = reasks = 0
        if self.guardrail.schema_retry and schema is not None and not system:
            errors = validate(schema, call.args)
            while errors and repairs + reasks < self.guardrail.max_retries:
                ep.add("repair", self.latency.repair_ms)
                repaired = repair_args(call.tool, call.args)
                if not validate(schema, repaired):
                    call, errors, repairs = ToolCall(call.tool, repaired, call.note), [], repairs + 1
                    break
                state.messages.append(f"{call.tool} arguments invalid: {'; '.join(errors)}")
                state.attempt += 1
                reasks += 1
                retry = self._think(ep, state)
                if not isinstance(retry, ToolCall) or retry.tool != call.tool:
                    break
                call, errors = retry, validate(schema, retry.args)
        transient_retries = 0
        key = call.note or str(len(ep.invocations))
        while True:
            res = self.sandbox.execute(call.tool, call.args, (key, transient_retries))
            ep.add(f"tool:{call.tool}", self._tool_cost(call, res.error_kind))
            if (res.error_kind == "transient" and self.guardrail.schema_retry
                    and transient_retries < self.guardrail.max_retries):
                transient_retries += 1
                continue
            break
        state.history.append(Observation(call, res.ok, res.result, res.error, res.error_kind))
        if res.ok and call.tool == "search_docs":
            ep.retrieved.extend(d["id"] for d in res.result["docs"])
        ep.invocations.append({
            "tool": call.tool, "note": call.note, "args": call.args, "system": system,
            "valid": schema is not None and not validate(schema, call.args), "ok": res.ok,
            "error_kind": res.error_kind, "repairs": repairs, "reasks": reasks,
            "transient_retries": transient_retries,
        })

    def _tool_cost(self, call: ToolCall, error_kind: str | None) -> float:
        lat = self.latency
        if error_kind == "validation":
            return lat.rejected_call_ms
        if error_kind == "transient":
            return lat.timeout_ms
        if call.tool == "search_docs":
            per_result = lat.retrieval_ms_per_result * int(call.args["top_k"])
            return lat.retrieval_ms[self.sandbox.retriever] + per_result
        return lat.tool_ms.get(call.tool, 5.0)

    def run(self, case_id: str, question: str) -> Episode:
        state = AgentState(case_id, question, TOOL_SCHEMAS, self.top_k)
        ep = Episode(self.sandbox.seed, case_id, self.latency)
        rechecked: set[tuple[int, str]] = set()
        gate_rounds = 0
        while len(ep.steps) < self.max_steps:
            state.attempt = 0
            action = self._think(ep, state)
            if isinstance(action, ToolCall):
                self._execute(ep, state, action, system=False)
                continue
            if not self.guardrail.grounding_gate or action.abstained:
                ep.answer = action
                return ep
            ep.add("gate", self.latency.gate_ms)
            failures = [(ev, why) for ev in action.evidence if (why := verify_evidence(ev))]
            if not failures:
                action.citations = list(dict.fromkeys(
                    ev.source_id for ev in action.evidence if ev.kind == "doc"))
                ep.answer = action
                return ep
            gate_rounds += 1
            if gate_rounds >= MAX_GATE_ROUNDS:
                break
            for ev, why in failures:
                state.rejected.add((ev.hop, ev.source_id))
                state.messages.append(f"gate rejected {ev.source_id} (hop {ev.hop}): {why}")
                if ev.kind == "doc" and (ev.hop, ev.subject) not in rechecked:
                    rechecked.add((ev.hop, ev.subject))
                    args = {"query": f"{ev.subject} {ev.relation}".strip(),
                            "top_k": min(2 * self.top_k, 20)}
                    self._execute(ep, state, ToolCall("search_docs", args,
                                                      f"recheck{ev.hop}|{ev.subject}"), system=True)
        ep.answer = FinalAnswer(None)
        ep.forced_abstention = True
        return ep
