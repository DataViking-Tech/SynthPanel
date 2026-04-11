"""Template engine for rendering dynamic questions from synthesis context.

Flattens SynthesisResult into a template context dict and renders
{theme_0}, {agreement_0}, etc. placeholders in question text fields.
Unresolvable keys render as literal placeholders (safe failure).
"""

from __future__ import annotations

import copy
import string
from typing import Any

from synth_panel.synthesis import SynthesisResult


def build_template_context(synthesis: SynthesisResult) -> dict[str, str]:
    """Flatten a SynthesisResult into a string-keyed template context.

    Returns keys like: summary, recommendation, theme_0, theme_1, ...,
    agreement_0, ..., disagreement_0, ..., surprise_0, ...
    """
    ctx: dict[str, str] = {
        "summary": synthesis.summary,
        "recommendation": synthesis.recommendation,
    }
    for prefix, items in [
        ("theme", synthesis.themes),
        ("agreement", synthesis.agreements),
        ("disagreement", synthesis.disagreements),
        ("surprise", synthesis.surprises),
    ]:
        for i, item in enumerate(items):
            ctx[f"{prefix}_{i}"] = item
    return ctx


class _SafeFormatter(string.Formatter):
    """Formatter that returns literal placeholder on missing keys."""

    def vformat(self, format_string: str, args: tuple, kwargs: dict) -> str:  # type: ignore[override]
        # Override to increase recursion limit guard
        return super().vformat(format_string, args, kwargs)

    def get_field(self, field_name: str, args: tuple, kwargs: dict) -> tuple[Any, str]:  # type: ignore[override]
        # Block dotted/bracket attribute access to prevent injection
        # Only allow simple key lookups
        first: Any
        rest: Any
        first, rest = field_name, iter([])
        try:
            import _string

            first, rest = _string.formatter_field_name_split(field_name)
        except (ImportError, ValueError):
            pass

        # Get the base value
        obj = self.get_value(first, args, kwargs)

        # If there's attribute/item access and the base was unresolved,
        # return the whole expression as a literal
        rest_list = list(rest)
        if rest_list:
            # If base key was missing (literal passthrough), return whole field as literal
            if isinstance(obj, str) and obj.startswith("{") and obj.endswith("}"):
                return "{" + field_name + "}", field_name
            # Even for resolved keys, block attribute traversal for safety
            return "{" + field_name + "}", field_name

        return obj, str(first)

    def get_value(self, key: int | str, args: tuple, kwargs: dict) -> str:  # type: ignore[override]
        if isinstance(key, str):
            try:
                return kwargs[key]
            except KeyError:
                return "{" + key + "}"
        return super().get_value(key, args, kwargs)

    def format_field(self, value: Any, format_spec: str) -> str:
        # If value is a literal placeholder passthrough, return as-is
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            return value  # ignore format spec on unresolved keys
        return super().format_field(value, format_spec)


_formatter = _SafeFormatter()


def render_questions(questions: list[dict[str, Any]], context: dict[str, str]) -> list[dict[str, Any]]:
    """Deep-copy questions and render template placeholders in text fields.

    Only string values in 'text' and 'follow_ups' (recursively) are rendered.
    """
    rendered = copy.deepcopy(questions)
    for q in rendered:
        if "text" in q and isinstance(q["text"], str):
            q["text"] = _formatter.format(q["text"], **context)
        if "follow_ups" in q and isinstance(q["follow_ups"], list):
            for fu in q["follow_ups"]:
                if isinstance(fu, str):
                    idx = q["follow_ups"].index(fu)
                    q["follow_ups"][idx] = _formatter.format(fu, **context)
                elif isinstance(fu, dict) and "text" in fu and isinstance(fu["text"], str):
                    fu["text"] = _formatter.format(fu["text"], **context)
    return rendered


def validate_template(text: str, context: dict[str, str]) -> list[str]:
    """Return list of unresolvable placeholder keys in text.

    Parses the format string and checks each field name against context.
    Returns empty list if all keys are resolvable.
    """
    unresolvable: list[str] = []
    for _, field_name, _, _ in _formatter.parse(text):
        if field_name is None:
            continue
        # Handle dotted access like {theme_0.upper} — only check base key
        base_key = field_name.split(".")[0].split("[")[0]
        if base_key and base_key not in context:
            unresolvable.append(base_key)
    return unresolvable
