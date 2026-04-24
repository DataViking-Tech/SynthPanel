"""Tests for the SynthBench best-model picker (sp-zq3).

Covers fetch + cache semantics (mirrors ``test_registry_cache.py``),
ranking by topic score vs SPS, ensemble fallback, offline mode, and
parse errors.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from synth_panel import synthbench
from synth_panel.synthbench import (
    CACHE_TTL,
    SYNTHBENCH_OFFLINE_ENV,
    SYNTHBENCH_REFRESH_ENV,
    SYNTHBENCH_URL_ENV,
    cache_path,
    load_leaderboard,
    parse_target,
    rank_entries,
    read_cache,
    recommend,
    write_cache,
)

URL = "https://example.test/leaderboard.json"


def _entry(
    *,
    model: str,
    sps: float,
    topic_scores: dict[str, float] | None = None,
    dataset: str = "globalopinionqa",
    provider: str = "anthropic",
    framework: str = "synthpanel",
    is_ensemble: bool = False,
    n: int = 100,
    jsd: float = 0.1,
    cost_per_100q: float = 0.5,
    run_count: int = 5,
    config_id: str | None = None,
) -> dict[str, Any]:
    return {
        "config_id": config_id or f"cfg-{model}",
        "model": model,
        "provider": provider,
        "dataset": dataset,
        "framework": framework,
        "is_ensemble": is_ensemble,
        "sps": sps,
        "n": n,
        "jsd": jsd,
        "topic_scores": topic_scores or {},
        "cost_per_100q": cost_per_100q,
        "run_count": run_count,
    }


SAMPLE: dict[str, Any] = {
    "generated_at": "2026-04-24T00:00:00Z",
    "synthbench_version": "1.0",
    "entries": [
        _entry(
            model="claude-haiku-4-5-20251001",
            sps=0.82,
            topic_scores={"Economy & Work": 0.85, "Technology & Digital Life": 0.71},
        ),
        _entry(
            model="claude-sonnet-4-6",
            sps=0.78,
            topic_scores={"Economy & Work": 0.82, "Technology & Digital Life": 0.80},
        ),
        _entry(
            model="gemini-2.5-flash",
            sps=0.80,
            topic_scores={"Economy & Work": 0.79, "Technology & Digital Life": 0.90},
            provider="google",
        ),
        # wrong dataset — filtered out
        _entry(
            model="grok-3",
            sps=0.99,
            dataset="gss",
            provider="xai",
        ),
    ],
}


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(SYNTHBENCH_URL_ENV, URL)
    monkeypatch.delenv(SYNTHBENCH_OFFLINE_ENV, raising=False)
    monkeypatch.delenv(SYNTHBENCH_REFRESH_ENV, raising=False)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _explode(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected network call: {request.method} {request.url}")


# ---------- parse_target ----------


def test_parse_target_topic_only() -> None:
    assert parse_target("Economy & Work") == ("Economy & Work", "globalopinionqa")


def test_parse_target_topic_and_dataset() -> None:
    assert parse_target("Economy & Work:gss") == ("Economy & Work", "gss")


def test_parse_target_dataset_only_with_leading_colon() -> None:
    assert parse_target(":gss") == (None, "gss")


def test_parse_target_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_target("   ")


# ---------- rank_entries ----------


def test_rank_entries_by_sps_filters_dataset() -> None:
    ranked = rank_entries(SAMPLE, topic=None, dataset="globalopinionqa")
    models = [e["model"] for e, _ in ranked]
    # Sorted by SPS desc: 0.82 haiku, 0.80 gemini, 0.78 sonnet. grok-3 filtered out.
    assert models == [
        "claude-haiku-4-5-20251001",
        "gemini-2.5-flash",
        "claude-sonnet-4-6",
    ]


def test_rank_entries_by_topic_score() -> None:
    ranked = rank_entries(SAMPLE, topic="Technology & Digital Life", dataset="globalopinionqa")
    models = [e["model"] for e, _ in ranked]
    # 0.90 gemini, 0.80 sonnet, 0.71 haiku
    assert models[0] == "gemini-2.5-flash"


def test_rank_entries_topic_case_insensitive() -> None:
    ranked = rank_entries(SAMPLE, topic="economy & work", dataset="globalopinionqa")
    assert ranked[0][0]["model"] == "claude-haiku-4-5-20251001"


def test_rank_entries_skips_entries_missing_topic_score() -> None:
    leaderboard = {"entries": [_entry(model="x", sps=0.5, topic_scores={})]}
    assert rank_entries(leaderboard, topic="Health & Science", dataset="globalopinionqa") == []


# ---------- recommend (inline leaderboard) ----------


def test_recommend_picks_top_by_sps() -> None:
    rec = recommend(":globalopinionqa", leaderboard=SAMPLE)
    assert rec is not None
    assert rec.model == "claude-haiku-4-5-20251001"
    assert rec.dataset == "globalopinionqa"
    assert rec.topic is None
    assert rec.sps == pytest.approx(0.82)
    assert rec.provider == "anthropic"
    assert rec.n == 100


def test_recommend_picks_top_by_topic() -> None:
    rec = recommend("Technology & Digital Life", leaderboard=SAMPLE)
    assert rec is not None
    assert rec.model == "gemini-2.5-flash"
    assert rec.topic == "Technology & Digital Life"
    assert rec.sps == pytest.approx(0.90)


def test_recommend_returns_none_for_empty_leaderboard() -> None:
    assert recommend("anything", leaderboard={"entries": []}) is None


def test_recommend_resolves_short_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the default alias table (avoid user's ~/.synthpanel/aliases.yaml).
    from synth_panel.llm import aliases

    monkeypatch.setattr(aliases, "_ALIASES_FILE", Path("/nonexistent/aliases.yaml"))
    monkeypatch.delenv("SYNTHPANEL_MODEL_ALIASES", raising=False)
    aliases._reset_cache()
    board = {"entries": [_entry(model="haiku", sps=0.9)]}
    rec = recommend(":globalopinionqa", leaderboard=board)
    assert rec is not None
    # 'haiku' alias resolves to canonical model id.
    assert rec.model == "claude-haiku-4-5-20251001"
    assert rec.raw_model == "haiku"


def test_recommend_ensemble_falls_back_to_config_base(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_panel.llm import aliases

    monkeypatch.setattr(aliases, "_ALIASES_FILE", Path("/nonexistent/aliases.yaml"))
    monkeypatch.delenv("SYNTHPANEL_MODEL_ALIASES", raising=False)
    aliases._reset_cache()
    board = {
        "entries": [
            _entry(
                model="",
                sps=0.95,
                framework="product",
                is_ensemble=True,
                config_id="synthpanel:haiku",
            )
        ]
    }
    rec = recommend(":globalopinionqa", leaderboard=board)
    assert rec is not None
    assert rec.is_ensemble is True
    assert rec.model == "claude-haiku-4-5-20251001"


def test_recommend_low_confidence_flag() -> None:
    board = {"entries": [_entry(model="haiku", sps=0.9, run_count=1)]}
    rec = recommend(":globalopinionqa", leaderboard=board)
    assert rec is not None
    assert rec.low_confidence is True


def test_recommend_format_line_has_expected_pieces() -> None:
    rec = recommend("Economy & Work", leaderboard=SAMPLE)
    assert rec is not None
    line = rec.format_line()
    assert "synthbench" in line
    assert "claude-haiku-4-5-20251001" in line
    assert "SPS" in line
    assert "source=synthbench.org" in line


# ---------- load_leaderboard (cache + network) ----------


def test_fresh_cache_hit_skips_network() -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')
    with _client(_explode) as client:
        loaded = load_leaderboard(client=client)
    assert loaded is not None
    assert loaded.leaderboard == SAMPLE


def test_stale_cache_304_keeps_cached_leaderboard() -> None:
    stale = datetime.now(timezone.utc) - CACHE_TTL - timedelta(hours=1)
    write_cache(SAMPLE, source_url=URL, etag='"abc"', fetched_at=stale)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("if-none-match") == '"abc"'
        return httpx.Response(304)

    with _client(handler) as client:
        loaded = load_leaderboard(client=client)
    assert loaded is not None
    assert loaded.leaderboard == SAMPLE
    cached = read_cache()
    assert cached is not None
    assert (datetime.now(timezone.utc) - cached.fetched_at) < timedelta(minutes=1)


def test_stale_cache_200_overwrites_with_new_payload() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    write_cache(SAMPLE, source_url=URL, etag='"old"', fetched_at=stale)
    updated = {"entries": [_entry(model="opus", sps=0.99)]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=updated, headers={"ETag": '"new"'})

    with _client(handler) as client:
        loaded = load_leaderboard(client=client)
    assert loaded is not None
    assert loaded.leaderboard == updated
    cached = read_cache()
    assert cached is not None
    assert cached.etag == '"new"'


def test_stale_cache_network_fail_returns_stale_with_warning() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    write_cache(SAMPLE, source_url=URL, etag='"abc"', fetched_at=stale)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    warnings: list[str] = []
    with _client(handler) as client:
        loaded = load_leaderboard(client=client, warn=warnings.append)
    assert loaded is not None
    assert loaded.leaderboard == SAMPLE
    assert any("stale cache" in w for w in warnings)


def test_no_cache_and_fetch_fail_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    warnings: list[str] = []
    with _client(handler) as client:
        loaded = load_leaderboard(client=client, warn=warnings.append)
    assert loaded is None
    assert any("synthbench unavailable" in w for w in warnings)


def test_no_cache_and_http_404_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    warnings: list[str] = []
    with _client(handler) as client:
        loaded = load_leaderboard(client=client, warn=warnings.append)
    assert loaded is None


def test_offline_env_prevents_any_network(monkeypatch: pytest.MonkeyPatch) -> None:
    write_cache(
        SAMPLE,
        source_url=URL,
        etag='"abc"',
        fetched_at=datetime.now(timezone.utc) - timedelta(hours=48),
    )
    monkeypatch.setenv(SYNTHBENCH_OFFLINE_ENV, "1")
    with _client(_explode) as client:
        loaded = load_leaderboard(client=client)
    assert loaded is not None
    assert loaded.leaderboard == SAMPLE


def test_offline_with_no_cache_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNTHBENCH_OFFLINE_ENV, "1")
    with _client(_explode) as client:
        loaded = load_leaderboard(client=client)
    assert loaded is None


def test_refresh_env_bypasses_fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')
    monkeypatch.setenv(SYNTHBENCH_REFRESH_ENV, "1")
    new_payload = {"entries": []}
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(200, json=new_payload)

    with _client(handler) as client:
        loaded = load_leaderboard(client=client)
    assert loaded is not None
    assert loaded.leaderboard == new_payload
    assert "if-none-match" not in seen_headers[0]


# ---------- recommend with real cache lookup ----------


def test_recommend_through_cache_layer() -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')
    with _client(_explode) as client:
        rec = recommend("Economy & Work", client=client)
    assert rec is not None
    assert rec.model == "claude-haiku-4-5-20251001"
    assert rec.cache_age_hours < 1.0


def test_recommend_returns_none_when_leaderboard_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    warnings: list[str] = []
    with _client(handler) as client:
        rec = recommend("anything", client=client, warn=warnings.append)
    assert rec is None


# ---------- module-level constant sanity ----------


def test_default_url_points_at_synthbench_org() -> None:
    # Documented in NOTES on sp-zq3.
    assert synthbench.DEFAULT_SYNTHBENCH_URL == "https://synthbench.org/data/leaderboard.json"


def test_cache_path_honors_data_dir_env(tmp_path: Path) -> None:
    assert cache_path() == tmp_path / "synthbench-cache.json"
