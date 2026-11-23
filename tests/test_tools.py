from agent_eval_mlflow.tools import TOOL_SCHEMAS, repair_args, validate


def test_valid_calculator_call_has_no_errors():
    assert validate(TOOL_SCHEMAS["calculator"], {"expression": "(2 + 3) * 4"}) == []


def test_unknown_property_and_missing_required_are_rejected():
    errors = validate(TOOL_SCHEMAS["unit_convert"], {"value": 3, "from_unit": "kg", "extra": 1})
    assert len(errors) == 2
    assert any("to_unit" in e for e in errors)
    assert any("extra" in e for e in errors)


def test_string_number_and_enum_alias_are_rejected_then_repaired():
    bad = {"value": "410", "from_unit": "kilograms", "to_unit": "lb"}
    assert validate(TOOL_SCHEMAS["unit_convert"], bad)
    fixed = repair_args("unit_convert", bad)
    assert fixed == {"value": 410.0, "from_unit": "kg", "to_unit": "lb"}
    assert validate(TOOL_SCHEMAS["unit_convert"], fixed) == []


def test_calculator_thousands_separator_is_normalised():
    fixed = repair_args("calculator", {"expression": "1,250 * 2"})
    assert fixed == {"expression": "1250 * 2"}
