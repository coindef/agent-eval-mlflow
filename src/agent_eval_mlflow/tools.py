"""Simulated tool sandbox: JSON-schema-style validation, repair, tools, seeded perturbations."""

from __future__ import annotations

import ast
import json
import operator
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .retrieval import SearchIndex

UNITS = ("kg", "lb", "g", "km", "mi", "m", "ft", "l", "gal", "C", "F")
TABLES = ("employees", "products")

TOOL_SCHEMAS: dict[str, dict] = {
    "search_docs": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query", "top_k"],
        "additionalProperties": False,
    },
    "lookup_record": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "enum": list(TABLES)},
            "key": {"type": "string", "minLength": 1},
            "field": {"type": "string"},
        },
        "required": ["table", "key"],
        "additionalProperties": False,
    },
    "calculator": {
        "type": "object",
        "properties": {"expression": {"type": "string", "pattern": r"^[0-9+\-*/(). ]+$"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    "unit_convert": {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "enum": list(UNITS)},
            "to_unit": {"type": "string", "enum": list(UNITS)},
        },
        "required": ["value", "from_unit", "to_unit"],
        "additionalProperties": False,
    },
}

# Surface forms a model might emit instead of the enum value. Used by the model to read
# questions and by the schema_retry guardrail to repair calls.
ALIASES: dict[str, str] = {
    **{a: "kg" for a in ("kilogram", "kilograms", "kilo", "kilos", "kgs")},
    **{a: "lb" for a in ("pound", "pounds", "lbs")},
    **{a: "g" for a in ("gram", "grams")},
    **{a: "km" for a in ("kilometer", "kilometers", "kilometre", "kilometres")},
    **{a: "mi" for a in ("mile", "miles")},
    **{a: "m" for a in ("meter", "meters", "metre", "metres")},
    **{a: "ft" for a in ("foot", "feet")},
    **{a: "l" for a in ("liter", "liters", "litre", "litres")},
    **{a: "gal" for a in ("gallon", "gallons")},
    **{a: "C" for a in ("celsius", "°c", "degc")},
    **{a: "F" for a in ("fahrenheit", "°f", "degf")},
    **{a: "employees" for a in ("employee", "staff", "people")},
    **{a: "products" for a in ("product", "catalog", "catalogue")},
}

_JSON_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    # JSON Schema: an integer is any number with a zero fractional part (4.0 is valid).
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
}


def validate(schema: Mapping, value: Any, path: str = "$") -> list[str]:
    """Validate `value` against a JSON-schema subset. Returns a list of error messages.

    Supported keywords: type, properties, required, additionalProperties (bool), enum,
    minLength, maxLength, pattern, minimum, maximum, items.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None and not _JSON_TYPES[expected](value):
        return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']}")
    if _JSON_TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        for name, item in value.items():
            if name in props:
                errors.extend(validate(props[name], item, f"{path}.{name}"))
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unexpected property {name!r}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(schema["items"], item, f"{path}[{i}]"))
    return errors


_NUM_PREFIX = re.compile(r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*[A-Za-z°%]*\s*$")


def coerce_number(value: Any) -> float | None:
    """Parse '1,250', '410 kg', '−12' (Unicode minus) into a float; None if not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    m = _NUM_PREFIX.match(value.replace("−", "-"))
    return float(m.group(1).replace(",", "")) if m else None


def repair_args(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic repairs used by the schema_retry guardrail.

    Drops unknown properties, coerces numeric strings, maps enum aliases and case variants,
    and normalises calculator expressions. Missing required fields cannot be repaired here.
    """
    schema = TOOL_SCHEMAS[tool]
    props = schema["properties"]
    out: dict[str, Any] = {}
    for name, value in args.items():
        if name not in props:
            continue
        spec = props[name]
        if spec["type"] in ("number", "integer") and not _JSON_TYPES[spec["type"]](value):
            num = coerce_number(value)
            if num is not None:
                value = int(num) if spec["type"] == "integer" and num.is_integer() else num
        if "enum" in spec and value not in spec["enum"] and isinstance(value, str):
            lowered = {str(e).lower(): e for e in spec["enum"]}
            key = value.strip().lower()
            value = lowered.get(key, ALIASES.get(key, value))
        if "pattern" in spec and isinstance(value, str):
            value = _normalise_expression(value)
        out[name] = value
    return out


def _normalise_expression(expr: str) -> str:
    expr = expr.replace("−", "-").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # thousands separators
    expr = re.sub(r"(?<=[\d)\s])x(?=[\s\d(])", "*", expr)  # "18400 x 0.85"
    return expr.rstrip("= ").strip()


_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate + - * / and parentheses over numeric literals. Anything else raises ValueError."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and _JSON_TYPES["number"](node.value):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    try:
        return ev(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ValueError(str(exc)) from exc


_TO_BASE = {"kg": 1.0, "lb": 0.45359237, "g": 0.001, "km": 1000.0, "mi": 1609.344, "m": 1.0,
            "ft": 0.3048, "l": 1.0, "gal": 3.785411784}
_DIMENSION = {"kg": "mass", "lb": "mass", "g": "mass", "km": "length", "mi": "length",
              "m": "length", "ft": "length", "l": "volume", "gal": "volume", "C": "temperature",
              "F": "temperature"}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    if _DIMENSION[from_unit] != _DIMENSION[to_unit]:
        raise ValueError(f"cannot convert {from_unit} to {to_unit}")
    if _DIMENSION[from_unit] == "temperature":
        celsius = value if from_unit == "C" else (value - 32.0) * 5.0 / 9.0
        return celsius if to_unit == "C" else celsius * 9.0 / 5.0 + 32.0
    return value * _TO_BASE[from_unit] / _TO_BASE[to_unit]


def load_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _key_tokens(key: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", key.lower()))


def lookup_rows(records: Sequence[dict], table: str, key: str, limit: int = 5) -> list[dict]:
    """Fuzzy key match: exact (case-insensitive) matches first, then token-overlap matches."""
    q = _key_tokens(key)
    scored = []
    for pos, row in enumerate(records):
        if row["table"] != table:
            continue
        overlap = len(q & _key_tokens(row["key"])) / max(len(q), 1)
        exact = row["key"].strip().lower() == key.strip().lower()
        if exact or overlap > 0:
            scored.append((not exact, -overlap, pos, row))
    scored.sort(key=lambda t: t[:3])
    return [{k: v for k, v in row.items() if k != "table"} for *_, row in scored[:limit]]


@dataclass(frozen=True)
class Perturbations:
    """Seeded sandbox perturbations (probabilities per tool-call attempt)."""

    transient_error: float = 0.08  # any tool: upstream timeout instead of a result
    schema_edge: float = 0.30  # lookup_record: numeric fields returned as formatted strings
    noisy_rows: float = 0.30  # lookup_record: inject a stale duplicate row and shuffle rows


def keyed_rng(seed: int, *key: Any) -> np.random.Generator:
    """Independent, reproducible random stream for (seed, key). Uses crc32, not hash()."""
    return np.random.default_rng([seed, *(zlib.crc32(str(k).encode()) for k in key)])


@dataclass
class ToolResult:
    ok: bool
    result: Any = None
    error: str | None = None
    error_kind: str | None = None  # "validation" | "transient" | "execution"


class ToolSandbox:
    def __init__(self, index: SearchIndex, records: Sequence[dict], retriever: str, seed: int,
                 perturbations: Perturbations = Perturbations()):
        self.index, self.records, self.retriever = index, list(records), retriever
        self.seed, self.p = seed, perturbations

    def execute(self, tool: str, args: Mapping[str, Any], call_key: tuple) -> ToolResult:
        """Validate server-side, then maybe fail transiently, then run the tool."""
        if tool not in TOOL_SCHEMAS:
            return ToolResult(False, error=f"unknown tool {tool!r}", error_kind="validation")
        errors = validate(TOOL_SCHEMAS[tool], dict(args))
        if errors:
            return ToolResult(False, error="; ".join(errors), error_kind="validation")
        rng = keyed_rng(self.seed, "tool", tool, *call_key)
        if rng.random() < self.p.transient_error:
            return ToolResult(False, error="TransientError: upstream timeout", error_kind="transient")
        try:
            return ToolResult(True, getattr(self, f"_{tool}")(args, rng))
        except ValueError as exc:
            return ToolResult(False, error=str(exc), error_kind="execution")

    def _search_docs(self, args, rng) -> dict:
        hits = self.index.search(self.retriever, args["query"], int(args["top_k"]))
        return {"docs": [{"id": d.id, "title": d.title, "text": d.text, "status": d.status,
                          "score": round(s, 6)} for d, s in hits]}

    def _lookup_record(self, args, rng) -> dict:
        rows = lookup_rows(self.records, args["table"], args["key"])
        if rows and rng.random() < self.p.noisy_rows:
            stale = {k: _stale_value(v) for k, v in rows[0].items()}
            stale.update(row_id=rows[0]["row_id"] + "-stale", key=rows[0]["key"],
                         status="inactive")
            rows = rows + [stale]
            rows = [rows[i] for i in rng.permutation(len(rows))]
        if rows and rng.random() < self.p.schema_edge:
            rows = [_format_numbers(r) for r in rows]
        return {"rows": rows}

    def _calculator(self, args, rng) -> dict:
        return {"result": safe_eval(args["expression"])}

    def _unit_convert(self, args, rng) -> dict:
        value = convert_units(float(args["value"]), args["from_unit"], args["to_unit"])
        return {"value": value, "unit": args["to_unit"]}


def _stale_value(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return value
    if isinstance(value, str):
        return f"{value} (old)"
    return round(value * 0.9, 2)


_EDGE_FORMATS = {"list_price_usd": "{:,.2f}", "weight_kg": "{:,g} kg", "units_in_stock": "{:,} units"}


def _format_numbers(row: dict) -> dict:
    out = dict(row)
    for name, fmt in _EDGE_FORMATS.items():
        if isinstance(out.get(name), (int, float)):
            out[name] = fmt.format(out[name])
    return out
