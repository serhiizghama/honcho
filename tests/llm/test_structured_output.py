"""Tests for repair_response_model_json in src/llm/structured_output.py.

Covers the deriver data-loss path (#993): a PromptRepresentation that fails
repair or ignores the schema must raise so the caller's retry/fallback chain
engages, while a genuine empty extraction still validates.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from src.llm.structured_output import (
    StructuredOutputError,
    repair_response_model_json,
)
from src.utils.representation import PromptRepresentation


def test_honest_empty_still_validates() -> None:
    result = repair_response_model_json('{"explicit": []}', PromptRepresentation, "m")
    assert isinstance(result, PromptRepresentation)
    assert result.explicit == []  # honest empty extraction is preserved


def test_null_explicit_coerced_to_empty() -> None:
    result = repair_response_model_json('{"explicit": null}', PromptRepresentation, "m")
    assert isinstance(result, PromptRepresentation)
    assert result.explicit == []


def test_valid_payload_preserved() -> None:
    result = repair_response_model_json(
        '{"explicit":[{"content":"the user has a dog"}]}', PromptRepresentation, "m"
    )
    assert isinstance(result, PromptRepresentation)
    assert len(result.explicit) == 1
    assert result.explicit[0].content == "the user has a dog"


def test_truncated_but_repairable_is_salvaged() -> None:
    # Missing closing quote/brackets — still recoverable to one observation.
    result = repair_response_model_json(
        '{"explicit":[{"content":"prefers tab', PromptRepresentation, "m"
    )
    assert isinstance(result, PromptRepresentation)
    assert len(result.explicit) == 1


def test_unparseable_json_raises() -> None:
    with pytest.raises(ValidationError):
        repair_response_model_json("not json at all {{{", PromptRepresentation, "m")


def test_schema_ignored_wrong_keys_raises() -> None:
    with pytest.raises(StructuredOutputError):
        repair_response_model_json('{"wrong": 1}', PromptRepresentation, "m")


def test_empty_object_raises() -> None:
    with pytest.raises(StructuredOutputError):
        repair_response_model_json("{}", PromptRepresentation, "m")


def test_deductive_only_payload_is_not_treated_as_schema_ignored() -> None:
    # `deductive` is a key the repair path recognizes; it must not trip the
    # schema-ignored guard (it validates to an empty explicit representation).
    result = repair_response_model_json(
        '{"deductive": [{"premises": ["a"], "conclusion": "b"}]}',
        PromptRepresentation,
        "m",
    )
    assert isinstance(result, PromptRepresentation)
    assert result.explicit == []


def test_other_models_still_raise_on_failure() -> None:
    # Only PromptRepresentation has the special-cased empty fallback removed;
    # every other model must keep raising, unchanged.
    class Other(BaseModel):
        a: int

    with pytest.raises(ValidationError):
        repair_response_model_json("not json", Other, "m")
