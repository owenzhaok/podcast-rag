"""Strict structured-output and request-local citation validation."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from api.rag.llm import InvalidGeneration
from api.rag.models import AnswerParagraph


class GeneratedOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: Literal["answered", "insufficient_context"]
    paragraphs: list[AnswerParagraph] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "answered") != bool(self.paragraphs):
            raise ValueError("Status and paragraphs disagree")
        return self


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def validate_output(raw: str, source_ids: frozenset[str]) -> GeneratedOutput:
    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536:
            raise ValueError("Invalid output size")
        parsed = json.loads(raw, object_pairs_hook=unique_object)
        output = GeneratedOutput.model_validate(parsed)
    except (ValueError, TypeError, ValidationError, RecursionError):
        raise InvalidGeneration() from None
    for paragraph in output.paragraphs:
        if not set(paragraph.source_ids).issubset(source_ids):
            raise InvalidGeneration("invalid_citations")
    return output
