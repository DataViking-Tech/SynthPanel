"""Tests for synth_panel.perturbation -- sp-5on.14."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from synth_panel.llm.models import (
    CompletionResponse,
    TokenUsage,
    ToolInvocationBlock,
)
from synth_panel.perturbation import (
    ALL_AXES,
    PersonaVariant,
    PerturbationAxis,
    PerturbationRecord,
    VariantSet,
    generate_panel_variants,
    generate_variants,
)

# --- Fixtures ---------------------------------------------------------------

_BASE_PERSONA = {
    "name": "Sarah Chen",
    "age": 34,
    "occupation": "Product Manager",
    "background": "Works at a mid-size SaaS company. 8 years in product roles.",
    "personality_traits": ["analytical", "pragmatic", "detail-oriented"],
}


def _make_extraction_response(
    data: dict,
    tool_name: str = "perturb_persona",
) -> CompletionResponse:
    """Build a CompletionResponse with a tool invocation containing `data`."""
    return CompletionResponse(
        id="r1",
        model="test",
        content=[
            ToolInvocationBlock(id="tc1", name=tool_name, input=data),
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )


def _make_fallback_response() -> CompletionResponse:
    """Build a CompletionResponse with no tool call (triggers fallback)."""
    from synth_panel.llm.models import TextBlock

    return CompletionResponse(
        id="r1",
        model="test",
        content=[TextBlock(text="I couldn't do it")],
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )


def _valid_extraction(
    *,
    axis: PerturbationAxis = PerturbationAxis.TRAIT_SWAP,
) -> dict:
    """Return a valid structured extraction result for a given axis."""
    if axis == PerturbationAxis.TRAIT_SWAP:
        return {
            "name": "Sarah Chen",
            "age": 34,
            "occupation": "Product Manager",
            "background": "Works at a mid-size SaaS company. 8 years in product roles.",
            "personality_traits": ["intuitive", "pragmatic", "detail-oriented"],
            "change_description": "Swapped analytical for intuitive",
            "original_field": "personality_traits",
            "original_value": "analytical, pragmatic, detail-oriented",
        }
    elif axis == PerturbationAxis.MOOD_CONTEXT:
        return {
            "name": "Sarah Chen",
            "age": 34,
            "occupation": "Product Manager",
            "background": "Works at a mid-size SaaS company. 8 years in product roles. Just got a promotion.",
            "personality_traits": ["analytical", "pragmatic", "detail-oriented"],
            "change_description": "Added mood context about recent promotion",
            "original_field": "background",
            "original_value": "Works at a mid-size SaaS company. 8 years in product roles.",
        }
    elif axis == PerturbationAxis.DEMOGRAPHIC_SHIFT:
        return {
            "name": "Sarah Chen",
            "age": 28,
            "occupation": "Associate Product Manager",
            "background": "Works at a mid-size SaaS company. 3 years in product roles.",
            "personality_traits": ["analytical", "pragmatic", "detail-oriented"],
            "change_description": "Shifted age down by 6 years, adjusted seniority",
            "original_field": "age",
            "original_value": "34",
        }
    else:  # BACKGROUND_REPHRASE
        return {
            "name": "Sarah Chen",
            "age": 34,
            "occupation": "Product Manager",
            "background": "An experienced product leader at a growing SaaS company, managing cross-functional teams.",
            "personality_traits": ["analytical", "pragmatic", "detail-oriented"],
            "change_description": "Rephrased to emphasize leadership over tenure",
            "original_field": "background",
            "original_value": "Works at a mid-size SaaS company. 8 years in product roles.",
        }


def _mock_client_for_axes(axes_sequence: list[PerturbationAxis]) -> MagicMock:
    """Build a mock LLMClient that returns valid extractions for a sequence of axes."""
    client = MagicMock()
    responses = [_make_extraction_response(_valid_extraction(axis=ax)) for ax in axes_sequence]
    client.send.side_effect = responses
    return client


# --- PerturbationAxis -------------------------------------------------------


class TestPerturbationAxis:
    def test_all_axes_has_four(self):
        assert len(ALL_AXES) == 4

    def test_enum_values(self):
        assert PerturbationAxis.TRAIT_SWAP.value == "trait_swap"
        assert PerturbationAxis.MOOD_CONTEXT.value == "mood_context"
        assert PerturbationAxis.DEMOGRAPHIC_SHIFT.value == "demographic_shift"
        assert PerturbationAxis.BACKGROUND_REPHRASE.value == "background_rephrase"


# --- PersonaVariant ---------------------------------------------------------


class TestPersonaVariant:
    def test_variant_name_auto_generated(self):
        record = PerturbationRecord(
            axis=PerturbationAxis.TRAIT_SWAP,
            original_field="personality_traits",
            original_value="analytical, pragmatic",
            perturbed_value="intuitive, pragmatic",
            change_description="Swapped analytical for intuitive",
        )
        variant = PersonaVariant(
            persona=dict(_BASE_PERSONA),
            source_persona_name="Sarah Chen",
            variant_index=0,
            perturbation=record,
        )
        assert variant.variant_name == "Sarah Chen (v0)"
        assert variant.persona["name"] == "Sarah Chen (v0)"
        assert variant.persona["_variant_of"] == "Sarah Chen"
        assert variant.persona["_perturbation_axis"] == "trait_swap"


# --- VariantSet -------------------------------------------------------------


class TestVariantSet:
    def test_source_name(self):
        vs = VariantSet(source_persona=_BASE_PERSONA, variants=[])
        assert vs.source_name == "Sarah Chen"
        assert vs.k == 0


# --- generate_variants (unit, mocked LLM) -----------------------------------


class TestGenerateVariants:
    def test_rejects_k_zero(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="k"):
            generate_variants(_BASE_PERSONA, client, k=0)

    def test_rejects_k_over_20(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="k"):
            generate_variants(_BASE_PERSONA, client, k=21)

    def test_rejects_missing_name(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="name"):
            generate_variants({"age": 30}, client)

    def test_axes_cycle_round_robin(self):
        """With K=5 and 4 axes, axes should cycle: 0,1,2,3,0."""
        expected_axes = [
            PerturbationAxis.TRAIT_SWAP,
            PerturbationAxis.MOOD_CONTEXT,
            PerturbationAxis.DEMOGRAPHIC_SHIFT,
            PerturbationAxis.BACKGROUND_REPHRASE,
            PerturbationAxis.TRAIT_SWAP,
        ]
        client = _mock_client_for_axes(expected_axes)

        result = generate_variants(dict(_BASE_PERSONA), client, k=5)

        assert len(result.variants) == 5
        for i, expected_axis in enumerate(expected_axes):
            assert result.variants[i].perturbation.axis == expected_axis

    def test_single_axis_mode(self):
        """When axes=[TRAIT_SWAP], all K variants should use TRAIT_SWAP."""
        client = _mock_client_for_axes([PerturbationAxis.TRAIT_SWAP] * 3)

        result = generate_variants(
            dict(_BASE_PERSONA),
            client,
            k=3,
            axes=[PerturbationAxis.TRAIT_SWAP],
        )

        assert len(result.variants) == 3
        for v in result.variants:
            assert v.perturbation.axis == PerturbationAxis.TRAIT_SWAP

    def test_fallback_variants_excluded(self):
        """If extraction fails (is_fallback=True), that variant is dropped."""
        axes_seq = [
            PerturbationAxis.TRAIT_SWAP,
            PerturbationAxis.MOOD_CONTEXT,
            PerturbationAxis.DEMOGRAPHIC_SHIFT,  # will be fallback
            PerturbationAxis.BACKGROUND_REPHRASE,
            PerturbationAxis.TRAIT_SWAP,
        ]

        # Build responses: 3rd call returns a fallback (no tool call,
        # retries exhausted)
        responses = []
        for i, ax in enumerate(axes_seq):
            if i == 2:
                # The StructuredOutputEngine retries internally; we need the
                # client.send to return text responses for all retry attempts
                # on this call. With retry_limit=2, that's 3 text responses.
                # But since StructuredOutputEngine is the one calling
                # client.send, we need to interleave.
                responses.append(_make_fallback_response())
                responses.append(_make_fallback_response())
                responses.append(_make_fallback_response())
            else:
                responses.append(_make_extraction_response(_valid_extraction(axis=ax)))

        client = MagicMock()
        client.send.side_effect = responses

        result = generate_variants(dict(_BASE_PERSONA), client, k=5)

        # Variant at index 2 was skipped due to fallback
        assert len(result.variants) == 4
        axes_in_result = [v.perturbation.axis for v in result.variants]
        assert PerturbationAxis.DEMOGRAPHIC_SHIFT not in axes_in_result

    def test_variant_metadata_fields(self):
        """Variants have correct provenance metadata stamped in persona dict."""
        client = _mock_client_for_axes([PerturbationAxis.TRAIT_SWAP])

        result = generate_variants(dict(_BASE_PERSONA), client, k=1)

        assert len(result.variants) == 1
        v = result.variants[0]
        assert v.persona["_variant_of"] == "Sarah Chen"
        assert v.persona["_perturbation_axis"] == "trait_swap"
        assert v.variant_index == 0
        assert v.source_persona_name == "Sarah Chen"

    def test_variant_set_properties(self):
        """VariantSet source_name and k track correctly."""
        client = _mock_client_for_axes([PerturbationAxis.TRAIT_SWAP, PerturbationAxis.MOOD_CONTEXT])

        result = generate_variants(dict(_BASE_PERSONA), client, k=2)

        assert result.source_name == "Sarah Chen"
        assert result.k == 2


# --- generate_panel_variants ------------------------------------------------


class TestGeneratePanelVariants:
    def test_returns_one_variant_set_per_persona(self):
        """Output length matches input persona count."""
        personas = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 40},
            {"name": "Carol", "age": 50},
        ]

        # Each persona gets k=2 variants, each needing one LLM call
        all_axes = [PerturbationAxis.TRAIT_SWAP, PerturbationAxis.MOOD_CONTEXT]
        responses = []
        for p in personas:
            for ax in all_axes:
                data = _valid_extraction(axis=ax)
                data["name"] = p["name"]
                responses.append(_make_extraction_response(data))

        client = MagicMock()
        client.send.side_effect = responses

        result = generate_panel_variants(personas, client, k=2, max_workers=1)

        assert len(result) == 3

    def test_preserves_input_order(self):
        """VariantSets are in the same order as input personas."""
        personas = [
            {"name": "Zara", "age": 25},
            {"name": "Alice", "age": 35},
        ]

        responses = []
        for p in personas:
            data = _valid_extraction(axis=PerturbationAxis.TRAIT_SWAP)
            data["name"] = p["name"]
            responses.append(_make_extraction_response(data))

        client = MagicMock()
        client.send.side_effect = responses

        result = generate_panel_variants(personas, client, k=1, max_workers=1)

        assert result[0].source_name == "Zara"
        assert result[1].source_name == "Alice"
