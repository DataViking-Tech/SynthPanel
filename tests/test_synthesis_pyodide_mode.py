"""Tests for the v1.2.0 (sy-huo) pyodide_safe_mode / async-DI surface on
``synthesize_panel``.

Three paths are covered:

1. **Degenerate path** — ``judge_enabled=False`` returns a sync
   :class:`SynthesisResult` with no LLM call, no thread spawn, and no
   trafilatura / lxml import. This is the canonical Cloudflare Python
   Workers (pyodide) invocation.
2. **Async-DI path** — passing an ``llm_client`` that conforms to
   :class:`AsyncLLMClient` turns the function into an awaitable that
   uses the consumer's async client end-to-end. The internal
   althing ``LLMClient`` (which uses ``threading.Lock`` + Semaphore)
   is never instantiated.
3. **Pyodide guard** — ``pyodide_safe_mode=True`` without an opt-out
   raises a clear ValueError rather than silently spawning a thread
   the runtime can't run.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from althing.cost import ZERO_USAGE, TokenUsage
from althing.ensemble import (
    AsyncCompletion,
    AsyncLLMClient,
    SynthesisResult,
    synthesize_panel,
)
from althing.orchestrator import PanelistResult

_QUESTIONS = [
    {"text": "What do you think of the product?"},
    {"text": "Would you recommend it?"},
]

_PANELIST_RESULTS = [
    PanelistResult(
        persona_name="Alice",
        responses=[
            {"question": "What do you think of the product?", "response": "I love the simplicity."},
            {"question": "Would you recommend it?", "response": "Yes, absolutely."},
        ],
        usage=ZERO_USAGE,
    ),
    PanelistResult(
        persona_name="Bob",
        responses=[
            {"question": "What do you think of the product?", "response": "It's decent but overpriced."},
            {"question": "Would you recommend it?", "response": "Only if they lower the price."},
        ],
        usage=ZERO_USAGE,
    ),
]


# ---------------------------------------------------------------------------
# Degenerate (judge_enabled=False) path.
# ---------------------------------------------------------------------------


class TestJudgeDisabled:
    def test_returns_degenerate_synthesis_result(self) -> None:
        """judge_enabled=False produces a real SynthesisResult, not None."""
        result = synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            judge_enabled=False,
        )
        assert isinstance(result, SynthesisResult)
        # Empty LLM-derived fields — caller should not mistake this for synthesis.
        assert result.themes == []
        assert result.agreements == []
        assert result.disagreements == []
        assert result.surprises == []
        assert result.recommendation == ""
        # Summary surfaces the disabled state for traceability.
        assert "judge_enabled=False" in result.summary.lower() or "judge disabled" in result.summary.lower()
        # No LLM call → no usage, no cost.
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.cost.input_cost == 0.0
        assert result.cost.output_cost == 0.0

    def test_no_thread_spawn(self) -> None:
        """The degenerate path must not spawn a thread (pyodide constraint)."""
        before = threading.active_count()
        synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            judge_enabled=False,
            pyodide_safe_mode=True,
        )
        # Allow daemon threads spun up by the test runner to settle, but
        # synthesize_panel itself must not introduce new ones.
        assert threading.active_count() == before

    def test_no_trafilatura_lxml_import(self) -> None:
        """Calling synthesize_panel with judge_enabled=False must not
        import trafilatura or lxml. Boardroom validates this on the
        pyodide runtime; we mirror the assertion here so CI catches
        any regression that pulls them in eagerly.
        """
        # Drop any pre-imported state from earlier tests.
        for mod in list(sys.modules):
            if mod == "trafilatura" or mod.startswith("trafilatura.") or mod == "lxml" or mod.startswith("lxml."):
                del sys.modules[mod]

        synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            judge_enabled=False,
            pyodide_safe_mode=True,
        )

        assert "trafilatura" not in sys.modules, (
            "synthesize_panel(..., judge_enabled=False) must not import trafilatura"
        )
        assert "lxml" not in sys.modules, "synthesize_panel(..., judge_enabled=False) must not import lxml"

    def test_accepts_async_llm_client_but_ignores_it(self) -> None:
        """When judge_enabled=False, an injected llm_client is accepted
        but never called — there's no LLM call to route.
        """

        class TrackingClient:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, *, prompt: str, model: str, max_tokens: int = 4096) -> AsyncCompletion:
                self.calls += 1
                return AsyncCompletion(text="{}", usage=None)

        tracking = TrackingClient()
        result = synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            pyodide_safe_mode=True,
            llm_client=tracking,  # type: ignore[arg-type]
            judge_enabled=False,
        )
        assert isinstance(result, SynthesisResult)
        assert tracking.calls == 0, "injected client must not be called when judge_enabled=False"


# ---------------------------------------------------------------------------
# Async DI (llm_client=<AsyncLLMClient>) path.
# ---------------------------------------------------------------------------


_FAKE_SYNTHESIS_JSON = """\
{
  "summary": "Panelists agreed on simplicity but split on price.",
  "themes": ["simplicity", "price sensitivity"],
  "agreements": ["The product is simple to use"],
  "disagreements": ["Whether the price is justified"],
  "surprises": ["Bob suggested a cheaper tier"],
  "recommendation": "Introduce a small-team pricing tier."
}
"""


class _FakeAsyncClient:
    """In-test AsyncLLMClient that records every call."""

    def __init__(self, text: str = _FAKE_SYNTHESIS_JSON, usage: TokenUsage | None = None) -> None:
        self.text = text
        self.usage = usage
        self.calls: list[dict[str, object]] = []

    async def complete(self, *, prompt: str, model: str, max_tokens: int = 4096) -> AsyncCompletion:
        self.calls.append({"prompt": prompt, "model": model, "max_tokens": max_tokens})
        return AsyncCompletion(text=self.text, usage=self.usage)


class TestAsyncDI:
    def test_returns_coroutine(self) -> None:
        """When llm_client is provided + judge_enabled=True, the function
        returns an awaitable so the consumer's event loop drives the call."""
        fake = _FakeAsyncClient()
        coro = synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            llm_client=fake,  # type: ignore[arg-type]
        )
        assert asyncio.iscoroutine(coro), "async-DI path must return a coroutine"
        coro.close()

    def test_async_client_is_called(self) -> None:
        """The injected async client must be invoked end-to-end —
        althing must not fall back to its internal threading-based
        LLMClient when an async DI client is provided."""
        fake = _FakeAsyncClient()
        coro = synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            llm_client=fake,  # type: ignore[arg-type]
        )
        result = asyncio.run(coro)  # type: ignore[arg-type]

        assert isinstance(result, SynthesisResult)
        assert len(fake.calls) == 1, "injected async client must be called exactly once"
        # Resolved model defaults to sonnet when neither model nor panelist_model is given.
        assert fake.calls[0]["model"] == "sonnet"
        # The prompt must include the panelist data so the judge has something to synthesize.
        assert "Alice" in fake.calls[0]["prompt"]  # type: ignore[operator]
        assert "Bob" in fake.calls[0]["prompt"]  # type: ignore[operator]

    def test_async_path_parses_json_response(self) -> None:
        """Synthesis fields populate from the injected client's JSON output."""
        fake = _FakeAsyncClient()
        result = asyncio.run(
            synthesize_panel(
                None,
                _PANELIST_RESULTS,
                _QUESTIONS,
                llm_client=fake,  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
        )
        assert result.summary == "Panelists agreed on simplicity but split on price."
        assert result.themes == ["simplicity", "price sensitivity"]
        assert result.agreements == ["The product is simple to use"]
        assert result.disagreements == ["Whether the price is justified"]
        assert result.surprises == ["Bob suggested a cheaper tier"]
        assert result.recommendation == "Introduce a small-team pricing tier."
        assert not result.is_fallback

    def test_async_path_tolerates_markdown_fences(self) -> None:
        """Some async backends wrap JSON in ```json fences despite the
        prompt instructions — synthesis must still parse cleanly."""
        fenced = f"```json\n{_FAKE_SYNTHESIS_JSON}\n```"
        fake = _FakeAsyncClient(text=fenced)
        result = asyncio.run(
            synthesize_panel(
                None,
                _PANELIST_RESULTS,
                _QUESTIONS,
                llm_client=fake,  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
        )
        assert not result.is_fallback
        assert result.themes == ["simplicity", "price sensitivity"]

    def test_async_path_fallback_on_garbage(self) -> None:
        """Non-JSON output yields a fallback SynthesisResult, not a crash."""
        fake = _FakeAsyncClient(text="this is not JSON, sorry")
        result = asyncio.run(
            synthesize_panel(
                None,
                _PANELIST_RESULTS,
                _QUESTIONS,
                llm_client=fake,  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
        )
        assert result.is_fallback
        assert result.error == "async_client_returned_unparseable_output"

    def test_async_path_records_usage_when_provided(self) -> None:
        """When the async client populates usage, althing propagates
        it onto the result so cost dashboards stay accurate."""
        usage = TokenUsage(input_tokens=1000, output_tokens=400)
        fake = _FakeAsyncClient(usage=usage)
        result = asyncio.run(
            synthesize_panel(
                None,
                _PANELIST_RESULTS,
                _QUESTIONS,
                llm_client=fake,  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
        )
        assert result.usage.input_tokens == 1000
        assert result.usage.output_tokens == 400

    def test_async_client_satisfies_protocol(self) -> None:
        """A bare-bones consumer adapter must isinstance-check as AsyncLLMClient.

        runtime_checkable Protocols only check method presence, so this
        pins the public contract: any object with a `complete` callable
        is a valid AsyncLLMClient.
        """
        fake = _FakeAsyncClient()
        assert isinstance(fake, AsyncLLMClient)


# ---------------------------------------------------------------------------
# pyodide_safe_mode guard.
# ---------------------------------------------------------------------------


class TestPyodideSafeMode:
    def test_pyodide_safe_mode_alone_raises(self) -> None:
        """pyodide_safe_mode=True without judge_enabled=False or an
        async llm_client must refuse loudly — silently falling through
        to the threading-based client would crash inside the Worker."""
        with pytest.raises(ValueError, match="pyodide_safe_mode"):
            synthesize_panel(
                None,
                _PANELIST_RESULTS,
                _QUESTIONS,
                pyodide_safe_mode=True,
            )

    def test_pyodide_safe_mode_with_async_client_returns_coroutine(self) -> None:
        """pyodide_safe_mode=True + async client is the canonical
        Workers-with-judge invocation. Returns coroutine."""
        fake = _FakeAsyncClient()
        coro = synthesize_panel(
            None,
            _PANELIST_RESULTS,
            _QUESTIONS,
            pyodide_safe_mode=True,
            llm_client=fake,  # type: ignore[arg-type]
        )
        assert asyncio.iscoroutine(coro)
        coro.close()


# ---------------------------------------------------------------------------
# Backwards-compat — defaults must preserve v1.1.0 behavior.
# ---------------------------------------------------------------------------


def test_v1_1_0_default_signature_unchanged() -> None:
    """Existing callers passing only (client, panelist_results, questions)
    must see no behavior change. Documented at the bead level —
    additive-only release."""
    from unittest.mock import MagicMock

    from althing.llm.models import CompletionResponse, ToolInvocationBlock
    from althing.llm.models import TokenUsage as LLMTokenUsage

    response_data = {
        "summary": "x",
        "themes": ["t"],
        "agreements": ["a"],
        "disagreements": ["d"],
        "surprises": ["s"],
        "recommendation": "r",
    }
    mock_client = MagicMock()
    mock_client.send.return_value = CompletionResponse(
        id="synth-1",
        model="claude-sonnet-4-6",
        content=[ToolInvocationBlock(id="tc1", name="synthesize", input=response_data)],
        usage=LLMTokenUsage(input_tokens=10, output_tokens=20),
    )

    result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)
    assert isinstance(result, SynthesisResult)
    assert not result.is_fallback
    assert result.summary == "x"
