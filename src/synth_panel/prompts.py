"""Shared prompt builders for persona system prompts and question formatting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
from jinja2.sandbox import SandboxedEnvironment

if TYPE_CHECKING:
    pass

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

# Markers that distinguish Jinja2 templates from legacy Python format strings.
_JINJA2_MARKERS = ("{{", "{%", "{#")


def is_jinja2_template(template: str) -> bool:
    """Return True if the string contains Jinja2 syntax markers."""
    return any(marker in template for marker in _JINJA2_MARKERS)


def compile_jinja2_template(template_str: str) -> jinja2.Template:
    """Compile a Jinja2 template string using SandboxedEnvironment.

    Validates syntax at compile time — callers should do this once at
    load time, not per persona render.  Raises
    :class:`jinja2.TemplateSyntaxError` on invalid syntax.
    """
    env = SandboxedEnvironment()
    return env.from_string(template_str)


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

    Supports both Jinja2 (``{{ name }}``, ``{% if ... %}``) and legacy
    Python format-string (``{name}``) syntax — detected automatically by
    :func:`is_jinja2_template`.  Returns the raw template string; callers
    should call :func:`compile_jinja2_template` if Jinja2 is detected so
    validation happens at load time rather than per render.
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
    template: str | jinja2.Template,
) -> str:
    """Build a system prompt by applying a template to the persona dict.

    Accepts either:

    - A pre-compiled :class:`jinja2.Template` (preferred — compile once
      via :func:`compile_jinja2_template` at load time for best performance).
    - A raw ``str`` with Jinja2 syntax (``{{ name }}``, ``{% if ... %}``
      etc.) — auto-detected and compiled inline on each call.
    - A raw ``str`` with legacy Python ``{name}``-style placeholders —
      rendered via ``str.format_map`` with a tolerant dict that leaves
      unknown keys as literal ``{key}`` strings.

    The ``personality_traits`` field is pre-joined with ``", "`` when using
    the legacy format path.  Jinja2 templates receive the raw list and can
    use ``{{ personality_traits | join(", ") }}`` for the same effect.
    """
    if isinstance(template, jinja2.Template):
        return template.render(**persona)
    if is_jinja2_template(template):
        return compile_jinja2_template(template).render(**persona)
    ctx = dict(persona)
    traits = ctx.get("personality_traits")
    if isinstance(traits, list):
        ctx["personality_traits"] = ", ".join(str(t) for t in traits)
    return template.format_map(_DefaultDict(str, ctx))


def build_question_prompt(question: dict[str, Any]) -> str:
    """Build a user prompt from a question definition."""
    text = question.get("text", question) if isinstance(question, dict) else str(question)
    return str(text)
