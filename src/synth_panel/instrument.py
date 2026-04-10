"""Instrument parser — supports v1 (flat), v2 (linear rounds), v3 (branching).

v1 format (flat questions):
    instrument:
      version: 1
      questions:
        - text: "..."

v2 format (multi-round, linear via depends_on):
    instrument:
      version: 2
      rounds:
        - name: discovery
          questions: [...]
        - name: deep_dive
          depends_on: discovery
          questions: [...]

v3 format (branching, route_when):
    instrument:
      version: 3
      rounds:
        - name: intro
          questions: [...]
          route_when:
            - if: "<condition>"
              goto: probe_pricing
            - else: wrap_up
        - name: probe_pricing
          questions: [...]
          route_when:
            - else: wrap_up
        - name: wrap_up
          questions: [...]

v1 and v2 stay valid as degenerate v3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

END_SENTINEL = "__end__"


@dataclass
class Round:
    """A single round in an instrument."""

    name: str
    questions: list[dict[str, Any]]
    depends_on: str | None = None
    route_when: list[dict[str, Any]] | None = None


@dataclass
class Instrument:
    """Parsed instrument with round definitions.

    v1, v2, and v3 YAML formats normalize to this structure.
    v1 instruments become a single round named "default".
    """

    version: int
    rounds: list[Round] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def questions(self) -> list[dict[str, Any]]:
        """Flat question list — convenience for single-round instruments."""
        if len(self.rounds) == 1:
            return self.rounds[0].questions
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

    Accepts v1 (``questions``) or v2/v3 (``rounds``). Runs the full
    DAG validation ladder before returning. Raises
    :class:`InstrumentError` on validation failure.
    """
    version = data.get("version", 1)

    if "rounds" in data:
        return _parse_rounds(data, version)

    if "questions" in data:
        return _parse_v1(data, version)

    raise InstrumentError("Instrument must have either 'questions' (v1) or 'rounds' (v2) key")


def _parse_v1(data: dict[str, Any], version: int) -> Instrument:
    """Parse v1 flat-questions format into a single-round instrument."""
    questions = data["questions"]
    if not isinstance(questions, list) or not questions:
        raise InstrumentError("'questions' must be a non-empty list")
    return Instrument(
        version=version,
        rounds=[Round(name="default", questions=questions)],
    )


def _parse_rounds(data: dict[str, Any], version: int) -> Instrument:
    """Two-pass parse for v2/v3 multi-round instruments."""
    raw_rounds = data["rounds"]
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise InstrumentError("'rounds' must be a non-empty list")

    # ---- Rung 1: Structural pass — collect names, build Round objects ----
    rounds: list[Round] = []
    name_set: set[str] = set()

    for i, raw in enumerate(raw_rounds):
        if not isinstance(raw, dict):
            raise InstrumentError(f"Round {i} must be a mapping, got {type(raw).__name__}")

        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise InstrumentError(f"Round {i} must have a 'name' string")

        if name in name_set:
            raise InstrumentError(f"Duplicate round name: '{name}'")

        questions = raw.get("questions")
        if not isinstance(questions, list) or not questions:
            raise InstrumentError(f"Round '{name}' must have a non-empty 'questions' list")

        depends_on = raw.get("depends_on")
        if depends_on is not None and not isinstance(depends_on, str):
            raise InstrumentError(f"Round '{name}': 'depends_on' must be a string, got {type(depends_on).__name__}")

        route_when = raw.get("route_when")
        if route_when is not None:
            if not isinstance(route_when, list) or not route_when:
                raise InstrumentError(f"Round '{name}': 'route_when' must be a non-empty list")
            for j, entry in enumerate(route_when):
                if not isinstance(entry, dict):
                    raise InstrumentError(f"Round '{name}': route_when[{j}] must be a mapping")

        name_set.add(name)
        rounds.append(
            Round(
                name=name,
                questions=questions,
                depends_on=depends_on,
                route_when=route_when,
            )
        )

    # ---- Rung 2: Goto resolution (forward refs allowed) ----
    for r in rounds:
        if r.depends_on is not None and r.depends_on not in name_set:
            raise InstrumentError(f"Round '{r.name}': depends_on '{r.depends_on}' does not exist")
        if r.route_when:
            _validate_route_when_targets(r, name_set)

    # ---- Rung 4: Else completeness (must be checked before reachability) ----
    for r in rounds:
        if r.route_when and "else" not in r.route_when[-1]:
            raise InstrumentError(
                f"round '{r.name}' has no else clause; add 'else: <round_name>' or 'else: {END_SENTINEL}'"
            )

    # ---- Rung 3: Acyclicity (topo sort) ----
    edges = _build_edges(rounds)
    cycle = _find_cycle(rounds, edges)
    if cycle is not None:
        raise InstrumentError(f"Cycle detected in instrument DAG: {' -> '.join(cycle)}")

    # ---- Rung 5: Reachability (warning, not error) ----
    warnings = _reachability_warnings(rounds, edges)

    return Instrument(version=version, rounds=rounds, warnings=warnings)


def _validate_route_when_targets(r: Round, name_set: set[str]) -> None:
    assert r.route_when is not None
    for j, entry in enumerate(r.route_when):
        if "else" in entry:
            target = entry["else"]
            if not isinstance(target, str):
                raise InstrumentError(f"round '{r.name}' route_when[{j}] else must be a string")
            if target != END_SENTINEL and target not in name_set:
                raise InstrumentError(f"round '{r.name}' goto '{target}' does not exist")
        elif "goto" in entry:
            target = entry["goto"]
            if not isinstance(target, str):
                raise InstrumentError(f"round '{r.name}' route_when[{j}] goto must be a string")
            if target != END_SENTINEL and target not in name_set:
                raise InstrumentError(f"round '{r.name}' goto '{target}' does not exist")
        else:
            raise InstrumentError(f"round '{r.name}' route_when[{j}] must have 'goto' or 'else'")


def _build_edges(rounds: list[Round]) -> dict[str, list[str]]:
    """Build directed flow edges (parent -> child) from depends_on and route_when."""
    edges: dict[str, list[str]] = {r.name: [] for r in rounds}
    for r in rounds:
        if r.depends_on:
            edges[r.depends_on].append(r.name)
        if r.route_when:
            for entry in r.route_when:
                target = entry.get("goto") or entry.get("else")
                if target and target != END_SENTINEL:
                    edges[r.name].append(target)
    return edges


def _find_cycle(rounds: list[Round], edges: dict[str, list[str]]) -> list[str] | None:
    """Return a cycle path as a list of round names, or None if acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {r.name: WHITE for r in rounds}
    parent: dict[str, str | None] = {r.name: None for r in rounds}
    cycle_path: list[str] | None = None

    def dfs(node: str) -> bool:
        nonlocal cycle_path
        color[node] = GRAY
        for nxt in edges.get(node, []):
            if color[nxt] == GRAY:
                # Reconstruct cycle: nxt ... node -> nxt
                path = [nxt]
                cur: str | None = node
                while cur is not None and cur != nxt:
                    path.append(cur)
                    cur = parent[cur]
                path.append(nxt)
                path.reverse()
                cycle_path = path
                return True
            if color[nxt] == WHITE:
                parent[nxt] = node
                if dfs(nxt):
                    return True
        color[node] = BLACK
        return False

    for r in rounds:
        if color[r.name] == WHITE and dfs(r.name):
            return cycle_path
    return None


def _reachability_warnings(rounds: list[Round], edges: dict[str, list[str]]) -> list[str]:
    """Return warnings for unreachable rounds.

    Entry round is rounds[0]. Traversal follows route_when/depends_on edges,
    plus an implicit linear edge from round_i to round_{i+1} when round_i has
    no route_when (preserving v2 linear semantics).
    """
    if not rounds:
        return []

    # Build traversal edges: explicit edges + implicit linear next.
    traverse: dict[str, list[str]] = {r.name: list(edges.get(r.name, [])) for r in rounds}
    for i, r in enumerate(rounds):
        if r.route_when is None and i + 1 < len(rounds):
            traverse[r.name].append(rounds[i + 1].name)

    seen: set[str] = set()
    stack = [rounds[0].name]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in traverse.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)

    return [f"unreachable round: '{r.name}'" for r in rounds if r.name not in seen]
