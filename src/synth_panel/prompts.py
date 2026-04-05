"""Shared prompt builders for persona system prompts and question formatting."""

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


def build_question_prompt(question: dict[str, Any]) -> str:
    """Build a user prompt from a question definition."""
    text = question.get("text", question) if isinstance(question, dict) else str(question)
    return str(text)
