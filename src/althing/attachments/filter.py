"""Per-persona attachment stratification (hq-iczd / D-phase hq-x16u).

Each attachment may carry a ``filter: list[predicate]`` field. A
persona "matches" an attachment when **all** predicates hold (implicit
AND); attachments without a ``filter`` are unconditional. The predicate
engine is the same one used by v3 routing — see
:mod:`althing.routing` — passed ``valid_fields=None`` so any
persona key is acceptable. A predicate that names a key the persona
lacks evaluates as ``False``.

When a persona doesn't match an attachment, it receives the question
text without that attachment — no skip, no fallback, no placeholder
(per D-phase decision).
"""

from __future__ import annotations

from typing import Any

from althing.routing import _evaluate_predicate


def filter_attachments(
    attachments: list[dict[str, Any]],
    persona: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the subset of ``attachments`` whose filter matches ``persona``.

    An attachment without a ``filter`` key (or with an empty list) is
    always included. Otherwise every predicate in the filter must hold
    against the persona's traits.
    """
    result: list[dict[str, Any]] = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        flt = att.get("filter")
        if not flt:
            result.append(att)
            continue
        if all(_evaluate_predicate(p, persona, valid_fields=None) for p in flt):
            result.append(att)
    return result


def count_strata(
    personas: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
) -> int:
    """Number of unique attachment-set partitions across ``personas``.

    For each persona, compute the set of attachments it matches; group
    personas by that set; return the number of distinct groups. With no
    attachments or no filters at all every persona sees the same set,
    so the count is 1 (or 0 if ``personas`` is empty).

    Consumed by hq-0pbp's K≤5 caching gate at frame stage. Identity is
    by ``id(att)`` — assumes the caller passes the same attachment list
    object for every persona, which is the orchestrator's invariant.
    """
    strata: set[frozenset[int]] = set()
    for p in personas:
        if not isinstance(p, dict):
            continue
        matched = filter_attachments(attachments, p)
        strata.add(frozenset(id(a) for a in matched))
    return len(strata)
