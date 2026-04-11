"""Shared prompt builders for persona system prompts and question formatting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

# Synthesis prompt version — increment when the prompt changes materially.
SYNTHESIS_PROMPT_VERSION = 1

SYNTHESIS_PROMPT = (
    "You are a research analyst synthesizing responses from a focus group panel. "
    "Multiple panelists answered the same set of questions. Your job is to identify "
    "patterns, agreements, disagreements, and surprising insights across their responses.\n\n"
    "Analyze the panelist responses below and produce a structured synthesis with:\n"
    "- summary: A concise overview of the key findings (2-4 sentences)\n"
    "- themes: The main themes that emerged across responses\n"
    "- agreements: Points where panelists broadly agreed\n"
    "- disagreements: Points where panelists diverged or disagreed\n"
    "- surprises: Unexpected or notable findings\n"
    "- recommendation: A brief actionable recommendation based on the findings\n\n"
    "Be specific and cite panelist names when relevant. Focus on substance, not meta-commentary."
)


def persona_system_prompt(persona: dict[str, Any]) -> str:
    """Build a system prompt from a persona definition."""
    parts = [f"You are role-playing as {persona.get('name', 'an anonymous person')}."]
    if persona.get("age"):
        parts.append(f"Age: {persona['age']}.")
    if persona.get("occupation"):
        parts.append(f"Occupation: {persona['occupation']}.")
    if persona.get("background"):
        parts.append(f"Background: {persona['background']}.")
    if persona.get("personality_traits"):
        traits = persona["personality_traits"]
        if isinstance(traits, list):
            traits = ", ".join(str(t) for t in traits)
        parts.append(f"Personality traits: {traits}.")
    parts.append(
        "Answer questions in character. Be authentic to this persona's "
        "perspective, experiences, and communication style. "
        "Give concise, direct answers."
    )
    return " ".join(parts)


def load_prompt_template(path: str) -> str:
    """Load a prompt template from a file path.

    The template should contain Python format-string placeholders such as
    ``{name}``, ``{age}``, ``{occupation}``, ``{background}``, and
    ``{personality_traits}``.  Any key present in the persona dict can be
    referenced.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return p.read_text(encoding="utf-8")


class _DefaultDict(defaultdict):
    """A defaultdict that returns '{key}' for missing keys during format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def persona_system_prompt_from_template(
    persona: dict[str, Any],
    template: str,
) -> str:
    """Build a system prompt by applying a template to the persona dict.

    Uses Python ``str.format_map`` with a tolerant dict that leaves
    unknown placeholders as literal ``{key}`` strings.  The
    ``personality_traits`` field is pre-joined with ", " if it's a list.
    """
    ctx = dict(persona)
    traits = ctx.get("personality_traits")
    if isinstance(traits, list):
        ctx["personality_traits"] = ", ".join(str(t) for t in traits)
    return template.format_map(_DefaultDict(str, ctx))


def build_question_prompt(question: dict[str, Any]) -> str:
    """Build a user prompt from a question definition."""
    text = question.get("text", question) if isinstance(question, dict) else str(question)
    return str(text)
