"""sy-547: end-to-end enforcement of typed response_schema.

Two layers:

* Orchestrator integration — a question with an enum/scale
  ``response_schema`` coerces the free-text answer to a typed value,
  persists BOTH raw (``response``) and typed (``response_typed``) plus the
  schema kind, and flags unmappable answers with ``schema_unmapped``.
* Poll-summary integration — a saved result carrying the persisted
  ``response_schema`` + ``response_typed`` buckets the question as ``enum``
  (not ``text``) and counts unmappable answers as unparseable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from synth_panel.cost import TokenUsage as CostTokenUsage
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import ConversationMessage
from synth_panel.poll_summary import build_poll_summary
from synth_panel.runtime import TurnSummary


def _system(p: dict[str, Any]) -> str:
    return f"You are {p['name']}"


def _question(q: dict[str, Any]) -> str:
    return q["text"]


def _turn(text: str) -> TurnSummary:
    usage = CostTokenUsage(input_tokens=10, output_tokens=5)
    msg = ConversationMessage(role="assistant", content=[{"type": "text", "text": text}], usage=usage)
    return TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)


# ---------------------------------------------------------------------------
# Orchestrator coercion
# ---------------------------------------------------------------------------


@patch("synth_panel.orchestrator.AgentRuntime")
def test_enum_answer_is_coerced_and_persisted(mock_runtime_cls: MagicMock) -> None:
    """The #547 repro: 'Blue.' for an enum coerces to 'blue', raw kept."""
    questions = [
        {
            "text": "Pick exactly one color and output only that.",
            "response_schema": {"type": "enum", "options": ["red", "green", "blue"]},
        }
    ]

    def runtime_factory(*args, **kwargs):
        runtime = MagicMock()
        runtime.run_turn.side_effect = lambda prompt: _turn("Blue.")
        return runtime

    mock_runtime_cls.side_effect = runtime_factory

    results, _reg, _sess = run_panel_parallel(
        client=MagicMock(),
        personas=[{"name": "P0"}],
        questions=questions,
        model="claude-sonnet-4-6",
        system_prompt_fn=_system,
        question_prompt_fn=_question,
        max_workers=1,
    )

    resp = results[0].responses[0]
    assert resp["response"] == "Blue."  # raw text preserved
    assert resp["response_typed"] == "blue"  # coerced typed value
    assert resp["response_schema"]["type"] == "enum"
    assert "schema_unmapped" not in resp


@patch("synth_panel.orchestrator.AgentRuntime")
def test_unmappable_enum_answer_is_flagged(mock_runtime_cls: MagicMock) -> None:
    questions = [
        {
            "text": "Pick a color.",
            "response_schema": {"type": "enum", "options": ["red", "green", "blue"]},
        }
    ]

    def runtime_factory(*args, **kwargs):
        runtime = MagicMock()
        runtime.run_turn.side_effect = lambda prompt: _turn("I'd go with a nice teal.")
        return runtime

    mock_runtime_cls.side_effect = runtime_factory

    results, _reg, _sess = run_panel_parallel(
        client=MagicMock(),
        personas=[{"name": "P0"}],
        questions=questions,
        model="claude-sonnet-4-6",
        system_prompt_fn=_system,
        question_prompt_fn=_question,
        max_workers=1,
    )

    resp = results[0].responses[0]
    assert resp["response"] == "I'd go with a nice teal."
    assert resp.get("schema_unmapped") is True
    assert "response_typed" not in resp


@patch("synth_panel.orchestrator.AgentRuntime")
def test_scale_answer_coerced_to_int(mock_runtime_cls: MagicMock) -> None:
    questions = [{"text": "Rate 1-5.", "response_schema": {"type": "scale", "min": 1, "max": 5}}]

    def runtime_factory(*args, **kwargs):
        runtime = MagicMock()
        runtime.run_turn.side_effect = lambda prompt: _turn("I'd say 4 out of 5.")
        return runtime

    mock_runtime_cls.side_effect = runtime_factory

    results, _reg, _sess = run_panel_parallel(
        client=MagicMock(),
        personas=[{"name": "P0"}],
        questions=questions,
        model="claude-sonnet-4-6",
        system_prompt_fn=_system,
        question_prompt_fn=_question,
        max_workers=1,
    )

    resp = results[0].responses[0]
    assert resp["response_typed"] == 4


@patch("synth_panel.orchestrator.AgentRuntime")
def test_text_schema_left_untouched(mock_runtime_cls: MagicMock) -> None:
    questions = [{"text": "Tell me a story.", "response_schema": {"type": "text"}}]

    def runtime_factory(*args, **kwargs):
        runtime = MagicMock()
        runtime.run_turn.side_effect = lambda prompt: _turn("Once upon a time...")
        return runtime

    mock_runtime_cls.side_effect = runtime_factory

    results, _reg, _sess = run_panel_parallel(
        client=MagicMock(),
        personas=[{"name": "P0"}],
        questions=questions,
        model="claude-sonnet-4-6",
        system_prompt_fn=_system,
        question_prompt_fn=_question,
        max_workers=1,
    )

    resp = results[0].responses[0]
    assert "response_typed" not in resp
    assert "response_schema" not in resp
    assert "schema_unmapped" not in resp


# ---------------------------------------------------------------------------
# Poll-summary buckets the persisted enum schema
# ---------------------------------------------------------------------------


def _saved_envelope() -> dict[str, Any]:
    """A saved result shaped like ``save_panel_result`` writes after sy-547,
    carrying the question's response_schema + per-response response_typed."""
    return {
        "persona_count": 3,
        "question_count": 1,
        "questions": [
            {
                "text": "Pick exactly one color and output only that.",
                "response_schema": {"type": "enum", "options": ["red", "green", "blue"]},
            }
        ],
        "results": [
            {
                "persona": "P0",
                "responses": [
                    {
                        "question": "Pick...",
                        "response": "Blue.",
                        "response_typed": "blue",
                        "response_schema": {"type": "enum", "options": ["red", "green", "blue"]},
                    }
                ],
            },
            {
                "persona": "P1",
                "responses": [
                    {
                        "question": "Pick...",
                        "response": "blue",
                        "response_typed": "blue",
                        "response_schema": {"type": "enum", "options": ["red", "green", "blue"]},
                    }
                ],
            },
            {
                "persona": "P2",
                "responses": [
                    {
                        "question": "Pick...",
                        "response": "teal",
                        "schema_unmapped": True,
                        "response_schema": {"type": "enum", "options": ["red", "green", "blue"]},
                    }
                ],
            },
        ],
    }


def test_poll_summary_buckets_persisted_enum_as_enum() -> None:
    summary = build_poll_summary(_saved_envelope())
    q0 = summary.questions[0]
    assert q0.kind == "enum"
    # Two panelists coerced to "blue"; the third (teal) is unparseable.
    assert q0.first_choice_counts == {"blue": 2}
    assert q0.winner == "blue"
    assert q0.n_unparseable == 1


# ---------------------------------------------------------------------------
# sy-546: blend-member drop detection
# ---------------------------------------------------------------------------


class _FakePR:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses


class _FakeMR:
    def __init__(self, model: str, panelist_results: list[_FakePR]) -> None:
        self.model = model
        self.panelist_results = panelist_results


class _FakeEnsemble:
    def __init__(self, model_results: list[_FakeMR]) -> None:
        self.model_results = model_results


def test_blend_dropped_models_detects_all_error_member() -> None:
    from synth_panel.cli.commands import _blend_dropped_models

    good = _FakeMR("good-model", [_FakePR([{"question": "Q", "response": "an answer"}])])
    # Every response is an error / inline error string → dropped.
    bad = _FakeMR(
        "bad-model",
        [
            _FakePR([{"question": "Q", "response": "[error: OpenRouter API error 404]", "error": True}]),
            _FakePR([{"question": "Q", "response": "[error: OpenRouter API error 404]", "error": True}]),
        ],
    )
    ensemble = _FakeEnsemble([good, bad])

    assert _blend_dropped_models(ensemble) == ["bad-model"]


def test_blend_dropped_models_empty_when_all_healthy() -> None:
    from synth_panel.cli.commands import _blend_dropped_models

    a = _FakeMR("a", [_FakePR([{"question": "Q", "response": "x"}])])
    b = _FakeMR("b", [_FakePR([{"question": "Q", "response": "y"}])])
    assert _blend_dropped_models(_FakeEnsemble([a, b])) == []
