"""sy-546: model reachability pre-flight.

A bad slug that 404s on every call must be caught before spending, with a
fail-fast message naming it. Transient / auth / credential failures are
inconclusive and never block the run.
"""

from __future__ import annotations

from althing.llm.errors import LLMError, LLMErrorCategory
from althing.llm.models import CompletionRequest
from althing.preflight import (
    PreflightReport,
    preflight_models,
)

BAD_SLUG = "openrouter/google/gemini-2.0-flash-001"
GOOD_SLUG = "openrouter/openai/gpt-4o-mini"


class _FakeClient:
    """Stand-in for LLMClient.send keyed by per-model behaviour."""

    def __init__(self, behaviour: dict[str, object]) -> None:
        self._behaviour = behaviour
        self.calls: list[str] = []

    def send(self, request: CompletionRequest):
        self.calls.append(request.model)
        outcome = self._behaviour.get(request.model, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # truthy sentinel — preflight ignores the value


def test_bad_slug_is_flagged_unreachable() -> None:
    client = _FakeClient(
        {
            BAD_SLUG: LLMError(
                "OpenRouter API error 404: No endpoints found for google/gemini-2.0-flash-001.",
                LLMErrorCategory.BAD_REQUEST,
                status_code=404,
            ),
            GOOD_SLUG: "ok",
        }
    )
    report = preflight_models([GOOD_SLUG, BAD_SLUG], client=client)

    assert not report.ok
    bad = report.unreachable
    assert [p.model for p in bad] == [BAD_SLUG]
    msg = report.failure_message()
    assert BAD_SLUG in msg
    assert "unreachable" in msg.lower()


def test_all_reachable_is_ok() -> None:
    client = _FakeClient({GOOD_SLUG: "ok", "openrouter/anthropic/claude-haiku-4.5": "ok"})
    report = preflight_models([GOOD_SLUG, "openrouter/anthropic/claude-haiku-4.5"], client=client)
    assert report.ok
    assert report.unreachable == []


def test_rate_limit_is_inconclusive_not_unreachable() -> None:
    # A 429 must NOT abort the run — it isn't a property of the slug.
    client = _FakeClient({GOOD_SLUG: LLMError("rate limited", LLMErrorCategory.RATE_LIMIT, status_code=429)})
    report = preflight_models([GOOD_SLUG], client=client)
    assert report.ok
    statuses = {p.model: p.status for p in report.probes}
    assert statuses[GOOD_SLUG] == "inconclusive"


def test_missing_credentials_is_inconclusive() -> None:
    client = _FakeClient({GOOD_SLUG: LLMError("no key", LLMErrorCategory.MISSING_CREDENTIALS)})
    report = preflight_models([GOOD_SLUG], client=client)
    assert report.ok


def test_distinct_models_probed_once_each() -> None:
    client = _FakeClient({GOOD_SLUG: "ok"})
    preflight_models([GOOD_SLUG, GOOD_SLUG, GOOD_SLUG], client=client)
    assert client.calls.count(GOOD_SLUG) == 1


def test_empty_models_returns_empty_report() -> None:
    report = preflight_models([], client=_FakeClient({}))
    assert isinstance(report, PreflightReport)
    assert report.ok
    assert report.probes == []
