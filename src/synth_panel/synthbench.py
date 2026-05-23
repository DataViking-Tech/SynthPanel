"""SynthBench leaderboard integration (sp-zq3).

Fetches the public SynthBench leaderboard (``leaderboard.json``) and picks
the best-ranked model for a given topic or dataset. Wired into the
``synthpanel panel run --best-model-for`` CLI flag so the SPS score on
synthbench.org can inform model selection without an extra manual step.

Behaviour mirrors the pack registry cache (``registry/cache.py``):

- 24-hour TTL, stored at ``$SYNTH_PANEL_DATA_DIR/synthbench-cache.json``
  (default ``~/.synthpanel/synthbench-cache.json``).
- Conditional GET via ``If-None-Match`` when the cached ETag is present.
- Graceful offline fallback: a stale cache satisfies the call with a
  stderr warning; no cache + network error returns ``None`` so the CLI
  can emit a friendly "synthbench unavailable" line and still run.
- ``SYNTHPANEL_SYNTHBENCH_URL`` env var overrides the fetch URL (tests).
- ``SYNTHPANEL_SYNTHBENCH_OFFLINE=1`` forces cache-only mode.
- ``SYNTHPANEL_SYNTHBENCH_REFRESH=1`` bypasses the TTL + ETag.

sy-nkh: when the live URL is unreachable AND no user cache exists,
:func:`load_leaderboard` falls back to a bundled snapshot at
``data/synthbench-snapshot.json``. The snapshot ships with the package
so ``--best-model-for`` produces a real recommendation on a fresh
install even while ``synthbench.org/data/leaderboard.json`` is 404'ing
(GH #494). The warning message that surfaces this includes the
``SYNTHPANEL_SYNTHBENCH_URL`` override so users with a working internal
mirror have a clear path forward.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SYNTHBENCH_URL = "https://synthbench.org/data/leaderboard.json"
SYNTHBENCH_URL_ENV = "SYNTHPANEL_SYNTHBENCH_URL"
SYNTHBENCH_OFFLINE_ENV = "SYNTHPANEL_SYNTHBENCH_OFFLINE"
SYNTHBENCH_REFRESH_ENV = "SYNTHPANEL_SYNTHBENCH_REFRESH"
DATA_DIR_ENV = "SYNTH_PANEL_DATA_DIR"
CACHE_FILENAME = "synthbench-cache.json"
CACHE_TTL = timedelta(hours=24)
FETCH_TIMEOUT = 10.0
DEFAULT_DATASET = "globalopinionqa"
MIN_RUN_COUNT = 3

# sy-nkh: bundled snapshot path — shipped alongside the package so a
# fresh install on a broken URL still yields a recommendation. Resolved
# at import time from this module's directory so it works whether
# synthpanel is installed as a wheel (data lives next to the .py files)
# or run from a source checkout. The ``data/`` subpackage is declared
# in pyproject.toml's ``[tool.setuptools.package-data]`` block.
_BUNDLED_SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "synthbench-snapshot.json"

WarnFn = Callable[[str], None]


class SynthBenchFetchError(Exception):
    """Network, HTTP-status, or JSON-parse failure during leaderboard fetch."""


@dataclass(frozen=True)
class CachedLeaderboard:
    fetched_at: datetime
    source_url: str
    etag: str | None
    leaderboard: dict[str, Any]


@dataclass(frozen=True)
class Recommendation:
    """A resolved best-model recommendation for a topic/dataset."""

    model: str
    raw_model: str
    provider: str
    dataset: str
    topic: str | None
    sps: float
    jsd: float | None
    n: int | None
    cost_per_100q: float | None
    run_count: int
    framework: str | None
    is_ensemble: bool
    fetched_at: datetime
    cache_age_hours: float
    low_confidence: bool
    runnable: bool = True
    """Whether :attr:`model` is a runnable provider model id (gh-519).

    SynthBench product/ensemble rows can surface a human-readable display
    label (e.g. ``"SynthPanel (Gemini Flash Lite)"``) in their ``model``
    field rather than a provider-runnable id. Stamping such a label onto
    ``--model`` fails at call time with a provider "not a valid model ID"
    error. ``recommend`` sets this to ``False`` when neither the entry's
    model field nor an inferred config base resolves to a runnable id, so
    the CLI can refuse with an actionable message instead of producing a
    non-runnable selection. Defaults to ``True`` so test fixtures that
    construct :class:`Recommendation` directly keep their prior shape.
    """
    source: str = "live"
    """Provenance discriminator — one of :data:`RECOMMENDATION_SOURCES`.

    Agents should branch on this to decide how to phrase the
    recommendation in downstream prose: ``live`` and ``cache`` reflect
    current upstream data; ``stale-cache`` reflects the user's last
    successful fetch (potentially old); ``bundled-snapshot`` reflects
    the package release date and predates any post-release leaderboard
    movement. See ``docs/recommended-models.md`` for the full table.
    """

    def format_line(self) -> str:
        """Render a one-line summary suitable for stderr."""
        focus = f"{self.dataset}/{self.topic}" if self.topic else self.dataset
        parts = [
            f"synthbench: best model for {focus} → {self.model}",
            f"SPS {self.sps:.3f}",
        ]
        if self.jsd is not None:
            parts.append(f"JSD {self.jsd:.3f}")
        if self.n is not None:
            parts.append(f"n={self.n}")
        if self.cost_per_100q is not None:
            parts.append(f"${self.cost_per_100q:.3f}/100q")
        age = max(0, int(self.cache_age_hours))
        parts.append(f"cached {age}h ago")
        parts.append(f"source={self.source}")
        return " · ".join(parts)


def synthbench_url() -> str:
    return os.environ.get(SYNTHBENCH_URL_ENV, DEFAULT_SYNTHBENCH_URL)


def _data_dir() -> Path:
    d = Path(os.environ.get(DATA_DIR_ENV, "~/.synthpanel")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path() -> Path:
    return _data_dir() / CACHE_FILENAME


def _parse_iso(value: str) -> datetime | None:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def read_cache(path: Path | None = None) -> CachedLeaderboard | None:
    target = path or cache_path()
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    fetched = _parse_iso(str(raw.get("fetched_at", "")))
    leaderboard = raw.get("leaderboard")
    source = raw.get("source_url")
    if fetched is None or not isinstance(leaderboard, dict) or not isinstance(source, str):
        return None
    etag = raw.get("etag")
    if not isinstance(etag, str) and etag is not None:
        etag = None
    return CachedLeaderboard(
        fetched_at=fetched,
        source_url=source,
        etag=etag,
        leaderboard=leaderboard,
    )


def write_cache(
    leaderboard: dict[str, Any],
    *,
    source_url: str,
    etag: str | None = None,
    fetched_at: datetime | None = None,
    path: Path | None = None,
) -> None:
    target = path or cache_path()
    stamp = fetched_at or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "fetched_at": stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": source_url,
        "etag": etag,
        "leaderboard": leaderboard,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)


def _is_fresh(cached: CachedLeaderboard, now: datetime | None = None) -> bool:
    moment = now or datetime.now(timezone.utc)
    return (moment - cached.fetched_at) < CACHE_TTL


def _env_flag(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def _default_warn(message: str) -> None:
    print(message, file=sys.stderr)


def _override_hint(target_url: str) -> str:
    """Return the actionable corrective suffix for unreachable-URL warnings.

    sy-nkh: the v1.5.0 warning ended with ``no recommendation`` which left
    the user with no next move. This suffix names the env var override
    and tags the bundled-snapshot fallback so the message is actionable
    by itself.
    """
    return (
        f"override the URL via ${SYNTHBENCH_URL_ENV} if you have a mirror — "
        f"current default {target_url} is tracked in GH #494"
    )


def _load_bundled_snapshot(
    path: Path | None = None,
) -> tuple[dict[str, Any], datetime] | None:
    """Read the package-bundled leaderboard snapshot (sy-nkh).

    Returns ``(leaderboard, fetched_at)`` or ``None`` when the snapshot
    file is missing or malformed. ``fetched_at`` is derived from the
    embedded ``_meta.snapshot_date`` so cache-age accounting is
    meaningful (users see "cached 30d ago" not "cached 0h ago" on a
    stale bundled fallback).

    The snapshot file is the last-resort fallback path. Never raises —
    a malformed snapshot just returns ``None`` and the caller continues
    to its no-recommendation branch.
    """
    target = path or _BUNDLED_SNAPSHOT_PATH
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        return None

    meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
    snapshot_iso = meta.get("snapshot_date") or raw.get("generated_at")
    fetched = _parse_iso(str(snapshot_iso)) if isinstance(snapshot_iso, str) else None
    if fetched is None:
        # A snapshot without a parseable date is still usable; just
        # stamp it at the epoch so the "stale" badge always shows.
        fetched = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return raw, fetched


def _fetch_http(
    url: str,
    *,
    etag: str | None,
    client: httpx.Client | None,
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """GET leaderboard, optionally conditional on ``etag``.

    Returns ``(data, etag, not_modified)``.
    """
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    owns_client = client is None
    active = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        try:
            resp = active.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise SynthBenchFetchError(f"network error fetching {url}: {exc}") from exc
        if resp.status_code == 304:
            return None, etag, True
        if resp.status_code != 200:
            raise SynthBenchFetchError(f"unexpected HTTP {resp.status_code} from {url}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise SynthBenchFetchError(f"malformed JSON from {url}: {exc}") from exc
        if not isinstance(body, dict):
            raise SynthBenchFetchError(f"expected JSON object from {url}, got {type(body).__name__}")
        new_etag = resp.headers.get("ETag") or resp.headers.get("etag")
        return body, new_etag, False
    finally:
        if owns_client:
            active.close()


RECOMMENDATION_SOURCES: tuple[str, ...] = (
    "live",
    "cache",
    "stale-cache",
    "bundled-snapshot",
)
"""Closed enum of provenance labels emitted alongside a recommendation.

- ``live`` — the leaderboard was fetched over HTTP this run (or the
  upstream returned 304 confirming the cached copy is still current).
- ``cache`` — the user's on-disk cache was fresh (< ``CACHE_TTL``), so
  no network call happened.
- ``stale-cache`` — the user's cache exceeded its TTL and the network
  fetch failed; the stale cache was used as a fallback.
- ``bundled-snapshot`` — the package-bundled snapshot was used because
  no user cache existed and the live fetch failed (or offline mode was
  requested without a user cache).

Agents inspecting the recommendation should treat ``live`` and ``cache``
as current, ``stale-cache`` as user-dated, and ``bundled-snapshot`` as
package-dated. See ``docs/recommended-models.md``.
"""


@dataclass(frozen=True)
class LoadedLeaderboard:
    leaderboard: dict[str, Any]
    fetched_at: datetime
    source: str = "live"
    """One of :data:`RECOMMENDATION_SOURCES`. Defaults to ``"live"`` so
    test fixtures that construct ``LoadedLeaderboard`` directly without
    going through :func:`load_leaderboard` keep their existing shape."""


def load_leaderboard(
    *,
    client: httpx.Client | None = None,
    url: str | None = None,
    refresh: bool | None = None,
    offline: bool | None = None,
    warn: WarnFn | None = None,
) -> LoadedLeaderboard | None:
    """Return the leaderboard dict (cache-aware). ``None`` on total failure.

    - Offline: cache only, or bundled snapshot if no cache (sy-nkh).
    - Refresh: bypass TTL + ETag, force GET.
    - Fresh cache: zero network.
    - Stale cache: conditional GET (304 → keep cache; 200 → overwrite;
      network error → warn + stale cache).
    - No cache + network error: bundled snapshot fallback + warn
      (sy-nkh — fresh installs see a recommendation while the canonical
      URL is 404'ing). Returns ``None`` only when even the bundled
      snapshot is unreadable.
    """
    target_url = url or synthbench_url()
    emit = warn or _default_warn
    offline_flag = offline if offline is not None else _env_flag(SYNTHBENCH_OFFLINE_ENV)
    refresh_flag = refresh if refresh is not None else _env_flag(SYNTHBENCH_REFRESH_ENV)

    cached = read_cache()

    if offline_flag:
        if cached is not None:
            # Offline + cache hit: data is from the user's cache, even
            # if it's older than CACHE_TTL — there's no way to refresh
            # while offline. Tag as ``cache`` if it's still fresh by TTL
            # standards, ``stale-cache`` otherwise so agents can decide
            # whether to trust the recommendation.
            label = "cache" if _is_fresh(cached) else "stale-cache"
            return LoadedLeaderboard(cached.leaderboard, cached.fetched_at, source=label)
        # sy-nkh: offline mode previously gave up entirely without a cache;
        # surface the bundled snapshot here too so air-gapped users get a
        # usable recommendation on the first ever call.
        bundled = _load_bundled_snapshot()
        if bundled is not None:
            data, snapshot_at = bundled
            emit(
                f"synthpanel: synthbench offline mode with no user cache — "
                f"using bundled snapshot from {snapshot_at.date().isoformat()} "
                f"(source=bundled-snapshot; see docs/recommended-models.md)"
            )
            return LoadedLeaderboard(data, snapshot_at, source="bundled-snapshot")
        return None

    url_matches = cached is not None and cached.source_url == target_url
    if cached is not None and url_matches and not refresh_flag and _is_fresh(cached):
        return LoadedLeaderboard(cached.leaderboard, cached.fetched_at, source="cache")

    conditional_etag: str | None = None
    if cached is not None and url_matches and not refresh_flag:
        conditional_etag = cached.etag

    try:
        data, new_etag, not_modified = _fetch_http(
            target_url, etag=conditional_etag, client=client, timeout=FETCH_TIMEOUT
        )
    except SynthBenchFetchError as exc:
        if cached is not None:
            emit(
                f"synthpanel: synthbench fetch failed ({exc}); using stale cache from "
                f"{cached.fetched_at.isoformat()} (source=stale-cache)"
            )
            return LoadedLeaderboard(cached.leaderboard, cached.fetched_at, source="stale-cache")
        # sy-nkh: no cache + network error → bundled snapshot fallback.
        # The canonical URL has been 404 since v1.5.0 ship (GH #494) so
        # without this, every fresh install's --best-model-for is a
        # no-op. Warn loudly with the env-var override so users with a
        # working mirror have a one-line fix.
        bundled = _load_bundled_snapshot()
        if bundled is not None:
            data_snap, snapshot_at = bundled
            emit(
                f"synthpanel: synthbench unavailable ({exc}); "
                f"using bundled snapshot from {snapshot_at.date().isoformat()} "
                f"(source=bundled-snapshot) — {_override_hint(target_url)}"
            )
            return LoadedLeaderboard(data_snap, snapshot_at, source="bundled-snapshot")
        emit(f"synthpanel: synthbench unavailable ({exc}) and no bundled snapshot found; {_override_hint(target_url)}")
        return None

    if not_modified and cached is not None:
        # 304 confirms the cached copy is still authoritative as of this
        # moment — treat as ``live`` so agents see the upstream
        # acknowledged it, not as ``cache`` (which would imply we never
        # talked to the upstream this run).
        now = datetime.now(timezone.utc)
        write_cache(cached.leaderboard, source_url=target_url, etag=cached.etag, fetched_at=now)
        return LoadedLeaderboard(cached.leaderboard, now, source="live")

    assert data is not None
    now = datetime.now(timezone.utc)
    write_cache(data, source_url=target_url, etag=new_etag, fetched_at=now)
    return LoadedLeaderboard(data, now, source="live")


# ---------- recommendation logic ----------


def parse_target(spec: str) -> tuple[str | None, str]:
    """Parse a ``--best-model-for`` spec into ``(topic, dataset)``.

    Formats accepted:

    - ``"TOPIC"``                  → topic, default dataset
    - ``"TOPIC:DATASET"``          → topic, dataset
    - ``":DATASET"`` / ``"DATASET"`` (no topic keyword): when the spec
      has no colon and matches a known dataset sentinel, treat as
      dataset-only. Otherwise treat as topic + default dataset.

    The simplest rule ships: colon-separated ``topic:dataset``. Any
    non-empty token before the colon is the topic, after is the dataset.
    A bare string is treated as a topic against the default dataset.
    """
    if not spec or not spec.strip():
        raise ValueError("empty --best-model-for target")
    text = spec.strip()
    if ":" in text:
        topic_part, dataset_part = text.split(":", 1)
        topic = topic_part.strip() or None
        dataset = dataset_part.strip() or DEFAULT_DATASET
        return topic, dataset
    return text, DEFAULT_DATASET


def _topic_score(entry: dict[str, Any], topic: str) -> float | None:
    topics = entry.get("topic_scores")
    if not isinstance(topics, dict):
        return None
    # Exact match first.
    if topic in topics:
        val = topics[topic]
        return float(val) if isinstance(val, (int, float)) else None
    # Case-insensitive fallback.
    lower = topic.lower()
    for key, val in topics.items():
        if isinstance(key, str) and key.lower() == lower and isinstance(val, (int, float)):
            return float(val)
    return None


def _sps(entry: dict[str, Any]) -> float | None:
    val = entry.get("sps")
    return float(val) if isinstance(val, (int, float)) else None


def _as_float(entry: dict[str, Any], key: str) -> float | None:
    val = entry.get(key)
    return float(val) if isinstance(val, (int, float)) else None


def _as_int(entry: dict[str, Any], key: str) -> int | None:
    val = entry.get(key)
    if isinstance(val, bool):
        return None
    return int(val) if isinstance(val, (int, float)) else None


def rank_entries(
    leaderboard: dict[str, Any],
    *,
    topic: str | None,
    dataset: str,
) -> list[tuple[dict[str, Any], float]]:
    """Filter + rank entries. Returns ``[(entry, score)]`` descending."""
    raw = leaderboard.get("entries")
    entries = raw if isinstance(raw, list) else []
    ranked: list[tuple[dict[str, Any], float]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("dataset") != dataset:
            continue
        if topic:
            score = _topic_score(entry, topic)
        else:
            score = _sps(entry)
        if score is None:
            continue
        ranked.append((entry, score))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def _resolve_model_string(raw: str) -> str:
    """Resolve leaderboard's ``model`` string through the alias table.

    Leaderboard strings may be raw model ids ("claude-haiku-4-5-20251001"),
    short aliases ("haiku"), or product-ish labels ("SynthPanel (Sonnet 4)").
    We pass raw + lowercase through ``resolve_alias``; if neither resolves
    to something different, the raw string is returned unchanged and the
    caller decides what to do with it.
    """
    from synth_panel.llm.aliases import resolve_alias

    if not raw:
        return raw
    resolved = resolve_alias(raw)
    if resolved != raw:
        return resolved
    lowered = raw.strip().lower()
    resolved2 = resolve_alias(lowered)
    if resolved2 != lowered:
        return resolved2
    return raw


def is_runnable_model_id(model: str) -> bool:
    """Return ``True`` if *model* looks like a runnable provider model id.

    SynthBench leaderboard rows — especially product/ensemble configs —
    sometimes carry a human-readable display label in their ``model``
    field (e.g. ``"SynthPanel (Gemini Flash Lite)"``) rather than a
    provider-runnable id. A label like that, stamped onto ``--model``,
    fails at call time with a provider "not a valid model ID" error
    (gh-519).

    The discriminator is structural rather than an allow-list: real model
    ids (``claude-haiku-4-5-20251001``, ``gemini-2.5-flash``, ``grok-3``,
    ``gpt-4o-mini``, ``openrouter/...``, short aliases like ``haiku``, and
    ``ollama:``/``local:`` specs) are single whitespace-free tokens, while
    display labels contain spaces and/or parentheses. The OpenAI-compatible
    and local backends accept arbitrary bare ids, so enumerating known
    prefixes would wrongly reject valid local/self-hosted models — instead
    we reject anything empty, whitespace-bearing, or paren-bearing and
    accept the rest.
    """
    if not model:
        return False
    text = model.strip()
    if not text:
        return False
    if any(ch.isspace() for ch in text):
        return False
    return "(" not in text and ")" not in text


# Canonical provider id prefixes (mirrors the ``model_prefixes`` declared on
# each provider's ProviderConfig in synth_panel.llm.providers.*). Kept local so
# resolving a recommendation doesn't have to import the full LLM client stack;
# only used as a *recognition* heuristic, never for routing.
_KNOWN_PROVIDER_PREFIXES: tuple[str, ...] = (
    "claude-",
    "gemini-",
    "grok-",
    "gpt-",
    "openrouter/",
)


def _is_recognized_model_id(model: str) -> bool:
    """Stricter than :func:`is_runnable_model_id` — used to gate config-id-derived
    base models (gh-519).

    ``_underlying_base`` extracts a token from ``config_id`` by splitting on
    separators, which for hyphenated ids like
    ``synthpanel-gemini-flash-lite-ba37570c`` yields a meaningless hash
    fragment (``ba37570c``). That fragment is whitespace-free so it passes
    :func:`is_runnable_model_id`, yet it is not a real model id. We only adopt
    a config-derived base when it matches a known provider prefix or is a
    registered short alias — otherwise we keep the original (display) label so
    the caller refuses rather than swapping in garbage.
    """
    if not is_runnable_model_id(model):
        return False
    text = model.strip()
    if any(text.startswith(prefix) for prefix in _KNOWN_PROVIDER_PREFIXES):
        return True
    # Registered short alias (resolve_alias maps it to a different canonical id).
    from synth_panel.llm.aliases import resolve_alias

    return resolve_alias(text) != text


def _underlying_base(entry: dict[str, Any]) -> str | None:
    """If an entry is an ensemble/product config, try to surface a base model."""
    config_id = entry.get("config_id")
    if not isinstance(config_id, str):
        return None
    # Heuristic: config ids for product configs tend to encode the underlying
    # base after a separator. Accept dash or slash. Fall back to None.
    for sep in (":", "/", "__", "-"):
        if sep in config_id:
            tail = config_id.split(sep)[-1].strip()
            if tail and tail.lower() not in ("ensemble", "product", "synthpanel"):
                return tail
    return config_id or None


def recommend(
    spec: str,
    *,
    leaderboard: dict[str, Any] | None = None,
    fetched_at: datetime | None = None,
    client: httpx.Client | None = None,
    url: str | None = None,
    refresh: bool | None = None,
    offline: bool | None = None,
    warn: WarnFn | None = None,
) -> Recommendation | None:
    """Pick the best model for ``spec`` (``"topic"`` or ``"topic:dataset"``).

    When ``leaderboard`` is supplied it is used directly (callers who
    already hold a leaderboard dict — e.g. tests). Otherwise the cache
    layer is consulted. Returns ``None`` if no candidate entry is found
    or the leaderboard is unavailable.
    """
    topic, dataset = parse_target(spec)

    source = "live"
    if leaderboard is None:
        loaded = load_leaderboard(client=client, url=url, refresh=refresh, offline=offline, warn=warn)
        if loaded is None:
            return None
        leaderboard = loaded.leaderboard
        fetched_at = loaded.fetched_at
        source = loaded.source
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc)

    ranked = rank_entries(leaderboard, topic=topic, dataset=dataset)
    if not ranked:
        return None

    entry, score = ranked[0]
    raw_model = str(entry.get("model") or "")
    resolved_model = _resolve_model_string(raw_model)
    framework = entry.get("framework") if isinstance(entry.get("framework"), str) else None
    is_ensemble = bool(entry.get("is_ensemble"))

    # Product/ensemble configs aren't runnable as-is. When the entry's
    # model field is empty OR a non-runnable display label (gh-519: e.g.
    # "SynthPanel (Gemini Flash Lite)"), try to infer a runnable base model
    # from config_id instead. Only adopt the inferred base when it itself
    # resolves to a runnable id — otherwise keep the original string so the
    # caller can report it verbatim and refuse rather than silently swap in
    # a config-id hash fragment.
    if (is_ensemble or framework == "product") and not is_runnable_model_id(resolved_model):
        base = _underlying_base(entry)
        if base:
            candidate = _resolve_model_string(base)
            if _is_recognized_model_id(candidate):
                resolved_model = candidate

    final_model = resolved_model or raw_model
    run_count = _as_int(entry, "run_count") or 0
    cache_age_hours = max(0.0, (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0)

    return Recommendation(
        model=final_model,
        raw_model=raw_model,
        provider=str(entry.get("provider") or ""),
        dataset=dataset,
        topic=topic,
        sps=float(score),
        jsd=_as_float(entry, "jsd"),
        n=_as_int(entry, "n"),
        cost_per_100q=_as_float(entry, "cost_per_100q"),
        run_count=run_count,
        framework=framework,
        is_ensemble=is_ensemble,
        fetched_at=fetched_at,
        cache_age_hours=cache_age_hours,
        low_confidence=run_count > 0 and run_count < MIN_RUN_COUNT,
        runnable=is_runnable_model_id(final_model),
        source=source,
    )
