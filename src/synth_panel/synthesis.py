"""Panel synthesis: aggregate panelist responses into research findings.

Takes individual panelist responses from a panel run and synthesizes them
into a structured SynthesisResult via an LLM call using tool-use forcing
(StructuredOutputEngine).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synth_panel.cost import (
    ZERO_USAGE,
    CostEstimate,
    TokenUsage,
    estimate_cost,
    lookup_pricing,
)
from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import InputMessage, TextBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.prompts import SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_VERSION
from synth_panel.structured import StructuredOutputConfig, StructuredOutputEngine


def _convert_llm_usage(llm_usage: LLMTokenUsage) -> TokenUsage:
    """Convert LLM-layer TokenUsage to cost-layer TokenUsage."""
    return TokenUsage(
        input_tokens=llm_usage.input_tokens,
        output_tokens=llm_usage.output_tokens,
        cache_creation_input_tokens=llm_usage.cache_write_tokens,
        cache_read_input_tokens=llm_usage.cache_read_tokens,
    )


# JSON Schema for the synthesis result, used by StructuredOutputEngine.
_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A concise summary of the panel findings (2-4 sentences).",
        },
        "themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key themes that emerged across panelist responses.",
        },
        "agreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Points where panelists broadly agreed.",
        },
        "disagreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Points where panelists disagreed or diverged.",
        },
        "surprises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unexpected or notable findings from the panel.",
        },
        "recommendation": {
            "type": "string",
            "description": "A brief actionable recommendation based on the panel findings.",
        },
    },
    "required": ["summary", "themes", "agreements", "disagreements", "surprises", "recommendation"],
}

_DEFAULT_MODEL = "sonnet"
_MAX_TOKENS = 4096


@dataclass
class SynthesisResult:
    """Aggregated synthesis of a panel run."""

    summary: str
    themes: list[str]
    agreements: list[str]
    disagreements: list[str]
    surprises: list[str]
    recommendation: str
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)
    cost: CostEstimate = field(default_factory=CostEstimate)
    model: str = _DEFAULT_MODEL
    synthesis_prompt_version: int = SYNTHESIS_PROMPT_VERSION
    is_fallback: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "summary": self.summary,
            "themes": self.themes,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "surprises": self.surprises,
            "recommendation": self.recommendation,
            "usage": self.usage.to_dict(),
            "cost": self.cost.format_usd(),
            "model": self.model,
            "prompt_version": self.synthesis_prompt_version,
        }


def _format_panelist_data(
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
) -> str:
    """Format panelist results and questions into a structured text block for the LLM."""
    parts: list[str] = []

    # Questions asked
    parts.append("## Questions Asked")
    for i, q in enumerate(questions, 1):
        text = q.get("text", q) if isinstance(q, dict) else str(q)
        parts.append(f"{i}. {text}")

    parts.append("")
    parts.append("## Panelist Responses")

    for result in panelist_results:
        name = result.persona_name
        parts.append(f"\n### {name}")
        for resp in result.responses:
            q_text = resp.get("question", "")
            answer = resp.get("response", "")
            is_follow_up = resp.get("follow_up", False)
            prefix = "  Follow-up" if is_follow_up else "  Q"
            if isinstance(answer, dict):
                answer = json.dumps(answer, indent=2)
            parts.append(f"{prefix}: {q_text}")
            parts.append(f"  A: {answer}")

    return "\n".join(parts)


def synthesize_panel(
    client: LLMClient,
    panelist_results: list[Any],
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    custom_prompt: str | None = None,
) -> SynthesisResult:
    """Synthesize panelist responses into a structured research finding.

    Args:
        client: LLM client for the synthesis call.
        panelist_results: List of PanelistResult from run_panel_parallel.
        questions: The questions that were asked.
        model: Model to use for synthesis (default: sonnet).
        custom_prompt: Override the default synthesis prompt.

    Returns:
        SynthesisResult with structured findings and cost tracking.
    """
    resolved_model = model or _DEFAULT_MODEL
    prompt_text = custom_prompt or SYNTHESIS_PROMPT

    # Format the panelist data into a user message
    panelist_data = _format_panelist_data(panelist_results, questions)
    user_content = f"{prompt_text}\n\n{panelist_data}"

    messages = [
        InputMessage(role="user", content=[TextBlock(text=user_content)]),
    ]

    config = StructuredOutputConfig(
        schema=_SYNTHESIS_SCHEMA,
        tool_name="synthesize",
        tool_description="Provide structured synthesis of the panel responses.",
    )

    engine = StructuredOutputEngine(client)
    result = engine.extract(
        model=resolved_model,
        max_tokens=_MAX_TOKENS,
        messages=messages,
        config=config,
    )

    # Convert usage
    usage = _convert_llm_usage(result.response.usage)
    pricing, _ = lookup_pricing(resolved_model)
    cost = estimate_cost(usage, pricing)

    if result.is_fallback:
        return SynthesisResult(
            summary="Synthesis failed — see error field.",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            usage=usage,
            cost=cost,
            model=resolved_model,
            is_fallback=True,
            error=result.error,
        )

    data = result.data
    return SynthesisResult(
        summary=data.get("summary", ""),
        themes=data.get("themes", []),
        agreements=data.get("agreements", []),
        disagreements=data.get("disagreements", []),
        surprises=data.get("surprises", []),
        recommendation=data.get("recommendation", ""),
        usage=usage,
        cost=cost,
        model=resolved_model,
    )
