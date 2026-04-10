"""Persona variant generation via LLM perturbation (sp-5on.14).

Generates K persona variants along 4 orthogonal axes using an LLM
perturbation pass. Each variant changes exactly ONE axis (controlled
experiment) and carries provenance metadata for downstream analysis.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import InputMessage, TextBlock
from synth_panel.structured.output import StructuredOutputConfig, StructuredOutputEngine

logger = logging.getLogger(__name__)


class PerturbationAxis(Enum):
    """The four orthogonal perturbation dimensions."""

    TRAIT_SWAP = "trait_swap"
    MOOD_CONTEXT = "mood_context"
    DEMOGRAPHIC_SHIFT = "demographic_shift"
    BACKGROUND_REPHRASE = "background_rephrase"


ALL_AXES: list[PerturbationAxis] = list(PerturbationAxis)


@dataclass(frozen=True)
class PerturbationRecord:
    """Metadata about a single perturbation applied to a persona."""

    axis: PerturbationAxis
    original_field: str
    original_value: str
    perturbed_value: str
    change_description: str


@dataclass
class PersonaVariant:
    """A perturbed copy of a persona with provenance tracking."""

    persona: dict[str, Any]
    source_persona_name: str
    variant_index: int
    perturbation: PerturbationRecord
    variant_name: str = ""

    def __post_init__(self) -> None:
        if not self.variant_name:
            self.variant_name = f"{self.source_persona_name} (v{self.variant_index})"
        self.persona["name"] = self.variant_name
        self.persona["_variant_of"] = self.source_persona_name
        self.persona["_perturbation_axis"] = self.perturbation.axis.value


@dataclass
class VariantSet:
    """All variants generated for a single base persona."""

    source_persona: dict[str, Any]
    variants: list[PersonaVariant]

    @property
    def source_name(self) -> str:
        return self.source_persona.get("name", "unknown")

    @property
    def k(self) -> int:
        return len(self.variants)


_PERTURBATION_PROMPTS: dict[PerturbationAxis, str] = {
    PerturbationAxis.TRAIT_SWAP: (
        "You are modifying a research persona for a sensitivity analysis.\n\n"
        "ORIGINAL PERSONA:\n"
        "Name: {name}\n"
        "Age: {age}\n"
        "Occupation: {occupation}\n"
        "Background: {background}\n"
        "Personality traits: {traits}\n\n"
        "TASK: Replace exactly 1-2 personality traits with plausible alternatives "
        "that a real person in this role might have. The replacement traits should "
        "be meaningfully different (not synonyms). Keep all other fields identical.\n\n"
        "Example: 'analytical' -> 'intuitive', 'risk-averse' -> 'adventurous'"
    ),
    PerturbationAxis.MOOD_CONTEXT: (
        "You are modifying a research persona for a sensitivity analysis.\n\n"
        "ORIGINAL PERSONA:\n"
        "Name: {name}\n"
        "Age: {age}\n"
        "Occupation: {occupation}\n"
        "Background: {background}\n"
        "Personality traits: {traits}\n\n"
        "TASK: Add a brief situational context (1-2 sentences) to the background "
        "that describes a recent experience or current mood state. This should be "
        "something that could plausibly affect how this person responds to survey "
        "questions. Keep name, age, occupation, and personality traits identical.\n\n"
        "Examples: 'Just received a promotion and is feeling optimistic about new tools.' "
        "'Had a frustrating experience with a vendor last week.' "
        "'Currently evaluating whether to switch jobs.'"
    ),
    PerturbationAxis.DEMOGRAPHIC_SHIFT: (
        "You are modifying a research persona for a sensitivity analysis.\n\n"
        "ORIGINAL PERSONA:\n"
        "Name: {name}\n"
        "Age: {age}\n"
        "Occupation: {occupation}\n"
        "Background: {background}\n"
        "Personality traits: {traits}\n\n"
        "TASK: Shift the persona's age by 5-10 years (either direction) and adjust "
        "the occupation/seniority to match the new age. The background should be "
        "lightly adjusted so it remains coherent with the new age and seniority. "
        "Keep personality traits identical. Keep the same name."
    ),
    PerturbationAxis.BACKGROUND_REPHRASE: (
        "You are modifying a research persona for a sensitivity analysis.\n\n"
        "ORIGINAL PERSONA:\n"
        "Name: {name}\n"
        "Age: {age}\n"
        "Occupation: {occupation}\n"
        "Background: {background}\n"
        "Personality traits: {traits}\n\n"
        "TASK: Rephrase the background paragraph to emphasize different aspects of "
        "this person's experience. The core facts should remain the same, but the "
        "framing and emphasis should shift. For example, if the original emphasizes "
        "technical expertise, the rephrase might emphasize management challenges. "
        "Keep name, age, occupation, and personality traits identical."
    ),
}

_PERTURBATION_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Persona name (must match original).",
        },
        "age": {
            "type": "integer",
            "description": "Persona age (may change for demographic_shift).",
        },
        "occupation": {
            "type": "string",
            "description": "Persona occupation (may change for demographic_shift).",
        },
        "background": {
            "type": "string",
            "description": "Full background paragraph (modified per the task).",
        },
        "personality_traits": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of personality traits (may change for trait_swap).",
        },
        "change_description": {
            "type": "string",
            "description": "One sentence describing what you changed and why.",
        },
        "original_field": {
            "type": "string",
            "enum": ["personality_traits", "background", "age", "occupation"],
            "description": "The primary field that was modified.",
        },
        "original_value": {
            "type": "string",
            "description": "The original value of the primary field before modification.",
        },
    },
    "required": [
        "name", "age", "occupation", "background", "personality_traits",
        "change_description", "original_field", "original_value",
    ],
}


def _format_prompt(axis: PerturbationAxis, persona: dict[str, Any]) -> str:
    """Fill the perturbation prompt template with persona fields."""
    traits = persona.get("personality_traits", [])
    traits_str = ", ".join(str(t) for t in traits) if traits else "(not specified)"
    return _PERTURBATION_PROMPTS[axis].format(
        name=persona.get("name", "(not specified)"),
        age=persona.get("age", "(not specified)"),
        occupation=persona.get("occupation", "(not specified)"),
        background=persona.get("background", "(not specified)"),
        traits=traits_str,
    )


def generate_variants(
    persona: dict[str, Any],
    client: LLMClient,
    *,
    k: int = 5,
    axes: list[PerturbationAxis] | None = None,
    model: str = "haiku",
    max_tokens: int = 1024,
) -> VariantSet:
    """Generate K persona variants via LLM perturbation.

    Each variant perturbs exactly ONE axis (controlled experiment). Axes are
    cycled round-robin across the K variants.

    Args:
        persona: Original persona dict. Must have at least "name".
        client: LLM client for generating perturbations.
        k: Number of variants to generate. Default 5. Minimum 1, maximum 20.
        axes: Which perturbation axes to use. Default: all four.
        model: Model for perturbation generation. Default "haiku" (cheap).
        max_tokens: Max tokens per perturbation call. Default 1024.

    Returns:
        VariantSet with K PersonaVariant objects (fewer if some fail).

    Raises:
        ValueError: If k < 1 or k > 20, or persona missing "name".
    """
    if k < 1 or k > 20:
        raise ValueError(f"k must be between 1 and 20, got {k}")
    if "name" not in persona:
        raise ValueError("persona must have a 'name' field")

    resolved_axes = axes if axes is not None else ALL_AXES
    engine = StructuredOutputEngine(client)
    config = StructuredOutputConfig(
        schema=_PERTURBATION_EXTRACTION_SCHEMA,
        tool_name="perturb_persona",
        tool_description="Generate a perturbed version of the persona.",
    )

    variants: list[PersonaVariant] = []
    source_name = persona["name"]

    for i in range(k):
        axis = resolved_axes[i % len(resolved_axes)]
        prompt = _format_prompt(axis, persona)

        result = engine.extract(
            model=model,
            max_tokens=max_tokens,
            messages=[InputMessage(role="user", content=[TextBlock(text=prompt)])],
            config=config,
        )

        if result.is_fallback:
            logger.warning(
                "Variant %d/%d for %r (axis=%s) failed extraction, skipping",
                i, k, source_name, axis.value,
            )
            continue

        data = result.data
        variant_persona = {
            "name": data.get("name", source_name),
            "age": data.get("age", persona.get("age")),
            "occupation": data.get("occupation", persona.get("occupation")),
            "background": data.get("background", persona.get("background")),
            "personality_traits": data.get("personality_traits", persona.get("personality_traits", [])),
        }

        # Determine the perturbed value based on original_field
        original_field = data.get("original_field", "")
        perturbed_value = str(variant_persona.get(original_field, ""))

        record = PerturbationRecord(
            axis=axis,
            original_field=original_field,
            original_value=data.get("original_value", ""),
            perturbed_value=perturbed_value,
            change_description=data.get("change_description", ""),
        )

        variant = PersonaVariant(
            persona=variant_persona,
            source_persona_name=source_name,
            variant_index=i,
            perturbation=record,
        )
        variants.append(variant)

    return VariantSet(source_persona=persona, variants=variants)


def generate_panel_variants(
    personas: list[dict[str, Any]],
    client: LLMClient,
    *,
    k: int = 5,
    axes: list[PerturbationAxis] | None = None,
    model: str = "haiku",
    max_workers: int | None = None,
) -> list[VariantSet]:
    """Generate variants for every persona in a panel.

    Args:
        personas: List of persona dicts.
        client: LLM client.
        k: Variants per persona. Default 5.
        axes: Perturbation axes. Default: all four.
        model: Model for generation. Default "haiku".
        max_workers: Parallelism for persona-level generation.
            Default: min(len(personas), 10).

    Returns:
        List of VariantSet, one per input persona, in input order.
    """
    workers = max_workers if max_workers is not None else min(len(personas), 10)

    def _gen(p: dict[str, Any]) -> VariantSet:
        return generate_variants(p, client, k=k, axes=axes, model=model)

    if workers <= 1 or len(personas) <= 1:
        return [_gen(p) for p in personas]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_gen, p) for p in personas]
        return [f.result() for f in futures]
