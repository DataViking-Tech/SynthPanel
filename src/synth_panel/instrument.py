"""Instrument parser — supports v1 (flat questions) and v2 (multi-round) formats.

v1 format (backward compatible):
    instrument:
      version: 1
      questions:
        - text: "..."

v2 format (multi-round):
    instrument:
      version: 2
      rounds:
        - name: discovery
          questions:
            - text: "..."
        - name: deep_dive
          depends_on: discovery
          questions:
            - text: "Based on {theme_0}, ..."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Round:
    """A single round in a multi-round instrument."""

    name: str
    questions: list[dict[str, Any]]
    depends_on: str | None = None


@dataclass
class Instrument:
    """Parsed instrument with round definitions.

    Both v1 and v2 YAML formats normalize to this structure.
    v1 instruments become a single round named "default".
    """

    version: int
    rounds: list[Round] = field(default_factory=list)

    @property
    def questions(self) -> list[dict[str, Any]]:
        """Flat question list — convenience for single-round instruments."""
        if len(self.rounds) == 1:
            return self.rounds[0].questions
        # For multi-round, return all questions across rounds
        result: list[dict[str, Any]] = []
        for r in self.rounds:
            result.extend(r.questions)
        return result

    @property
    def is_multi_round(self) -> bool:
        return len(self.rounds) > 1


class InstrumentError(ValueError):
    """Raised when an instrument definition is invalid."""


def parse_instrument(data: dict[str, Any]) -> Instrument:
    """Parse a raw instrument dict into a validated Instrument.

    Accepts either:
    - v1: dict with ``questions`` key → single "default" round
    - v2: dict with ``rounds`` key → multi-round instrument

    Raises :class:`InstrumentError` on validation failure.
    """
    version = data.get("version", 1)

    if "rounds" in data:
        return _parse_v2(data, version)

    if "questions" in data:
        return _parse_v1(data, version)

    raise InstrumentError(
        "Instrument must have either 'questions' (v1) or 'rounds' (v2) key"
    )


def _parse_v1(data: dict[str, Any], version: int) -> Instrument:
    """Parse v1 flat-questions format into a single-round instrument."""
    questions = data["questions"]
    if not isinstance(questions, list) or not questions:
        raise InstrumentError("'questions' must be a non-empty list")
    return Instrument(
        version=version,
        rounds=[Round(name="default", questions=questions)],
    )


def _parse_v2(data: dict[str, Any], version: int) -> Instrument:
    """Parse v2 multi-round format with dependency validation."""
    raw_rounds = data["rounds"]
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise InstrumentError("'rounds' must be a non-empty list")

    rounds: list[Round] = []
    seen_names: set[str] = set()

    for i, raw in enumerate(raw_rounds):
        if not isinstance(raw, dict):
            raise InstrumentError(f"Round {i} must be a mapping, got {type(raw).__name__}")

        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise InstrumentError(f"Round {i} must have a 'name' string")

        if name in seen_names:
            raise InstrumentError(f"Duplicate round name: '{name}'")

        questions = raw.get("questions")
        if not isinstance(questions, list) or not questions:
            raise InstrumentError(f"Round '{name}' must have a non-empty 'questions' list")

        depends_on = raw.get("depends_on")
        if depends_on is not None:
            if not isinstance(depends_on, str):
                raise InstrumentError(
                    f"Round '{name}': 'depends_on' must be a string, "
                    f"got {type(depends_on).__name__}"
                )
            if depends_on not in seen_names:
                raise InstrumentError(
                    f"Round '{name}': depends_on '{depends_on}' references "
                    f"an undefined or later round (only earlier rounds allowed)"
                )

        seen_names.add(name)
        rounds.append(Round(name=name, questions=questions, depends_on=depends_on))

    return Instrument(version=version, rounds=rounds)
