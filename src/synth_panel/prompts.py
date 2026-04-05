"""Shared prompt builders for persona system prompts and question formatting."""

from typing import Any


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
