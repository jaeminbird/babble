import pytest

from babble.types import Diagnosis, object_schema, string_property

VALID = {
    "root_cause_device": "r3",
    "category": "interface_down",
    "summary": "eth1 is shut down on r3.",
    "evidence": ["show interface eth1: administratively down"],
    "fix_applied": True,
}


def test_from_tool_input_builds_a_diagnosis():
    assert Diagnosis.from_tool_input(VALID) == Diagnosis(
        root_cause_device="r3",
        category="interface_down",
        summary="eth1 is shut down on r3.",
        evidence=["show interface eth1: administratively down"],
        fix_applied=True,
    )


def test_from_tool_input_names_every_missing_field():
    with pytest.raises(ValueError) as error:
        Diagnosis.from_tool_input({"root_cause_device": "r3", "category": "other"})
    assert str(error.value) == "missing summary; missing evidence; missing fix_applied"


def test_from_tool_input_names_every_wrong_type():
    bad = dict(VALID, root_cause_device=3, evidence="one string", fix_applied="yes")
    with pytest.raises(ValueError) as error:
        Diagnosis.from_tool_input(bad)
    assert str(error.value) == (
        "root_cause_device has the wrong type; evidence has the wrong type; "
        "fix_applied has the wrong type"
    )


def test_from_tool_input_rejects_evidence_items_that_are_not_strings():
    with pytest.raises(ValueError, match="evidence must be a list of strings"):
        Diagnosis.from_tool_input(dict(VALID, evidence=["ok", 2]))


def test_object_schema_requires_every_property_by_default():
    assert object_schema({"a": string_property("A.")}) == {
        "type": "object",
        "properties": {"a": {"type": "string", "description": "A."}},
        "required": ["a"],
        "additionalProperties": False,
    }


def test_object_schema_accepts_an_explicit_required_list():
    schema = object_schema({"a": string_property("A."), "b": string_property("B.")}, ["a"])
    assert schema["required"] == ["a"]
