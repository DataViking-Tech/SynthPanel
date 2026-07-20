"""sy-550: a `type: url` attachment that yields no usable content must be a
hard error by default — the persona never answers blind on missing content,
the failure is counted in the failure rate, and per-attachment fetch status is
persisted for auditability. ``--allow-empty-attachments`` restores best-effort
placeholder behaviour.

These tests drive the real orchestrator (``run_panel_parallel``) with a
``type: url`` attachment and monkeypatch the fetch ladder's ``extract`` to fail,
so the URLBlock-lowering path is exercised end to end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from althing import orchestrator as orch_mod
from althing.fetch import lower as lower_mod
from althing.fetch.perimeter import PerimeterDeny
from althing.llm.models import CompletionResponse, StopReason, TextBlock
from althing.llm.models import TokenUsage as LLMTokenUsage
from althing.orchestrator import run_panel_parallel


def _text_response(text: str = "Hello!") -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="claude-sonnet",
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.send = MagicMock(return_value=_text_response())
    return client


def _system_prompt(persona: dict[str, Any]) -> str:
    return f"You are {persona.get('name', 'Anonymous')}."


def _question_prompt(question: dict[str, Any]) -> str:
    return question.get("text", str(question)) if isinstance(question, dict) else str(question)


def _url_question() -> dict[str, Any]:
    return {
        "text": "React to this page",
        "attachments": [{"type": "url", "url": "http://localhost:4321/", "fetch_mode": "markdown"}],
    }


def _deny(url, cfg):
    raise PerimeterDeny("no safe address for 'localhost': loopback")


def test_failed_url_attachment_is_error_by_default(monkeypatch):
    """Default: the perimeter-denied fetch fails the affected question — the
    response is flagged as an error and carries the attachment_fetch_error
    marker, instead of silently sending empty content to the model."""
    monkeypatch.setattr(lower_mod, "extract", _deny)

    results, _registry, _sessions = run_panel_parallel(
        client=_mock_client(),
        personas=[{"name": "Alice"}, {"name": "Bob"}],
        questions=[_url_question()],
        model="sonnet",
        system_prompt_fn=_system_prompt,
        question_prompt_fn=_question_prompt,
        max_workers=2,
    )

    assert len(results) == 2
    for pr in results:
        resp = pr.responses[0]
        assert resp.get("error") is True
        marker = resp.get("attachment_fetch_error")
        assert marker is not None
        assert marker["url"] == "http://localhost:4321/"
        assert "loopback" in marker["reason"]
        # status is recorded for auditability
        status = resp.get("attachment_fetch_status")
        assert status and status[0]["status"] == "failed"


def test_allow_empty_attachments_proceeds_best_effort(monkeypatch):
    """With --allow-empty-attachments the run proceeds: the failed fetch
    becomes a placeholder and the model is still called (no error flag), but
    the failed fetch status is still recorded."""
    monkeypatch.setattr(lower_mod, "extract", _deny)

    client = _mock_client()
    results, _registry, _sessions = run_panel_parallel(
        client=client,
        personas=[{"name": "Alice"}],
        questions=[_url_question()],
        model="sonnet",
        system_prompt_fn=_system_prompt,
        question_prompt_fn=_question_prompt,
        allow_empty_attachments=True,
    )

    resp = results[0].responses[0]
    assert resp.get("error") is not True
    # the model WAS called best-effort
    assert client.send.called
    status = resp.get("attachment_fetch_status")
    assert status and status[0]["status"] == "failed"
    assert "loopback" in status[0]["reason"]


def test_successful_fetch_records_ok_status(monkeypatch):
    """A successful fetch records an ok status and the response is healthy."""
    from althing.fetch.ladder import AttachmentIntent, LadderResult

    def ok(url, cfg):
        return LadderResult(
            url=url,
            intent=AttachmentIntent.TEXT,
            text="The real on-page content the persona should react to.",
            text_mode="markdown",
            screenshot=None,
            screenshot_mode=None,
            final_url=url,
            redirect_chain=[],
            stale=False,
            fetched=True,
        )

    monkeypatch.setattr(lower_mod, "extract", ok)

    results, _registry, _sessions = run_panel_parallel(
        client=_mock_client(),
        personas=[{"name": "Alice"}],
        questions=[{"text": "React", "attachments": [{"type": "url", "url": "https://example.com/p"}]}],
        model="sonnet",
        system_prompt_fn=_system_prompt,
        question_prompt_fn=_question_prompt,
    )

    resp = results[0].responses[0]
    assert resp.get("error") is not True
    status = resp.get("attachment_fetch_status")
    assert status and status[0]["status"] == "ok"


def test_attachment_fetch_error_is_exported():
    """The orchestrator imports AttachmentFetchError for the marker branch."""
    assert hasattr(orch_mod, "AttachmentFetchError")
