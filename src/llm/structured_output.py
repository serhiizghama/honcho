from __future__ import annotations

import json
import logging
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from src.utils.json_parser import validate_and_repair_json
from src.utils.representation import PromptRepresentation

logger = logging.getLogger(__name__)


class StructuredOutputError(ValueError):
    """Raised when structured output cannot be validated or repaired."""


def schema_instruction(response_format: type[BaseModel], *, tools_present: bool) -> str:
    """Structured-output instruction appended to the conversation for
    providers without native (or tools-compatible) schema enforcement.

    When tools are in play the wording is conditional so the model remains
    free to emit tool calls; validation then relies on parse + repair.
    """
    schema_json = json.dumps(response_format.model_json_schema(), indent=2)
    if tools_present:
        return (
            "\n\nIf not responding with a tool call, respond with valid JSON "
            f"matching this schema:\n{schema_json}"
        )
    return f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"


def repair_response_model_json(
    raw_content: str,
    response_model: type[BaseModel],
    _model: str,
) -> BaseModel:
    """Repair truncated or malformed JSON and validate against the response model.

    A ``PromptRepresentation`` used to fall back to an empty result whenever
    repair or validation failed. Because ``explicit`` has a ``default_factory``,
    that swallowed two distinct failures as if the model had legitimately found
    nothing: unparseable JSON, and well-formed JSON that ignores the schema
    entirely (e.g. ``{"wrong": 1}``). Both propagate as errors now so the
    caller's retry/fallback chain engages; a genuine ``{"explicit": []}`` still
    validates as an honest empty extraction.
    """

    repaired_data: Any = None
    try:
        final = validate_and_repair_json(raw_content)
        repaired_data = json.loads(final)

        if (
            response_model is PromptRepresentation
            and "deductive" in repaired_data
            and isinstance(repaired_data["deductive"], list)
        ):
            for item in repaired_data["deductive"]:
                if isinstance(item, dict):
                    if "conclusion" not in item and "premises" in item:
                        if item["premises"]:
                            item["conclusion"] = (
                                f"[Incomplete reasoning from premises: {item['premises'][0][:100]}...]"
                            )
                        else:
                            item["conclusion"] = (
                                "[Incomplete reasoning - conclusion missing]"
                            )
                    if "premises" not in item:
                        item["premises"] = []

        final = json.dumps(repaired_data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        final = ""

    # Well-formed JSON that shares no recognized key with the schema ignored it
    # entirely and would otherwise validate to an empty PromptRepresentation.
    if response_model is PromptRepresentation and isinstance(repaired_data, dict):
        payload_keys = set(cast("dict[str, Any]", repaired_data))
        recognized = set(response_model.model_fields) | {"deductive"}
        if not recognized & payload_keys:
            logger.error(
                "Structured output ignored the schema (keys=%s); raw=%r",
                sorted(payload_keys),
                raw_content[:500],
            )
            raise StructuredOutputError(
                "Structured output did not match the expected schema"
            )

    try:
        return response_model.model_validate_json(final)
    except ValidationError:
        if response_model is PromptRepresentation:
            logger.error(
                "Failed to repair structured output into %s; raw=%r",
                response_model.__name__,
                raw_content[:500],
            )
        raise


def validate_structured_output(
    content: object,
    response_model: type[BaseModel],
) -> BaseModel:
    if isinstance(content, response_model):
        return content
    if isinstance(content, str):
        return response_model.model_validate_json(content)
    if isinstance(content, dict):
        return response_model.model_validate(content)
    raise StructuredOutputError(
        f"Unsupported structured output payload: {type(content).__name__}"
    )


def empty_structured_output(response_model: type[BaseModel]) -> BaseModel:
    if response_model is PromptRepresentation:
        return PromptRepresentation(explicit=[])
    return response_model.model_validate({})
