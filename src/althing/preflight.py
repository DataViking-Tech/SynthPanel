"""Model reachability pre-flight (sy-546).

An ensemble / weighted / blend run lists several models via ``--models``.
When one slug is bogus (e.g. ``openrouter/google/gemini-2.0-flash-001``,
which 404s on every OpenRouter call), the historical behaviour was a
silent degrade: every call to that member errored, the run completed on
the survivors, and a ``--blend`` quietly became a 2-model blend. The 404
is deterministic and knowable before any persona work, so this module
probes each distinct model once with a 1-token completion and classifies
the result:

* reachable — the probe succeeded.
* unreachable — the provider rejected the model itself (404 / "no
  endpoints" / model-not-found style ``BAD_REQUEST``). This is the
  fail-fast signal.
* inconclusive — a transient or environmental failure (rate limit,
  server error, transport, or missing credentials). We do NOT block the
  run on these: they are not a property of the slug, and a flaky probe
  must not abort an otherwise-valid panel.

:func:`preflight_models` runs the probes (concurrently, via the shared
``LLMClient`` throttles) and returns a :class:`PreflightReport`. The CLI
calls it before spending — on the real run AND on ``--dry-run`` — and
aborts with a message naming the bad slug(s) when any model is
unreachable.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from althing.llm.client import LLMClient
from althing.llm.errors import LLMError, LLMErrorCategory
from althing.llm.models import CompletionRequest, InputMessage, TextBlock

logger = logging.getLogger(__name__)

# Error categories that mean "the slug is wrong" (fail fast) rather than
# "the environment hiccuped" (don't block). 404 / no-endpoints / unknown
# model land in BAD_REQUEST via classify_http_status (400-499 except
# 401/403/429).
_UNREACHABLE_CATEGORIES = frozenset({LLMErrorCategory.BAD_REQUEST})

# Categories we treat as inconclusive — the probe could not prove the
# slug bad, so we let the run proceed and surface the issue at call time.
_INCONCLUSIVE_CATEGORIES = frozenset(
    {
        LLMErrorCategory.RATE_LIMIT,
        LLMErrorCategory.SERVER_ERROR,
        LLMErrorCategory.TRANSPORT,
        LLMErrorCategory.AUTHENTICATION,
        LLMErrorCategory.MISSING_CREDENTIALS,
        LLMErrorCategory.DESERIALIZATION,
        LLMErrorCategory.RETRIES_EXHAUSTED,
    }
)


@dataclass(frozen=True)
class ModelProbe:
    """Outcome of probing a single model slug."""

    model: str
    status: str  # "reachable" | "unreachable" | "inconclusive"
    detail: str | None = None

    @property
    def unreachable(self) -> bool:
        return self.status == "unreachable"


@dataclass
class PreflightReport:
    """Aggregate result of probing every model in a run."""

    probes: list[ModelProbe] = field(default_factory=list)

    @property
    def unreachable(self) -> list[ModelProbe]:
        return [p for p in self.probes if p.unreachable]

    @property
    def ok(self) -> bool:
        """True iff no model was proven unreachable."""
        return not self.unreachable

    def failure_message(self) -> str:
        """Render the actionable abort message naming the bad slug(s)."""
        bad = self.unreachable
        lines = [
            f"Pre-flight failed: {len(bad)} model(s) in --models are unreachable "
            "(the provider rejected the slug, not a transient error):",
        ]
        for p in bad:
            detail = f" — {p.detail}" if p.detail else ""
            lines.append(f"  - {p.model}: model not found / no endpoints{detail}")
        lines.append(
            "Fix the slug(s) above (check the provider's model catalog) or drop "
            "them from --models. Pass --skip-preflight to bypass this check."
        )
        return "\n".join(lines)


def _probe_one(client: LLMClient, model: str) -> ModelProbe:
    """Probe a single model with a minimal 1-token completion."""
    request = CompletionRequest(
        model=model,
        max_tokens=1,
        messages=[InputMessage(role="user", content=[TextBlock(text="ping")])],
        temperature=0.0,
        cache_enabled=False,
    )
    try:
        client.send(request)
    except LLMError as exc:
        if exc.category in _UNREACHABLE_CATEGORIES:
            logger.info("preflight: %s unreachable (%s)", model, exc.category.value)
            return ModelProbe(model=model, status="unreachable", detail=str(exc)[:200])
        # Anything else is inconclusive — don't block the run on a flake or
        # an environment problem the operator can see in the real error.
        logger.info("preflight: %s inconclusive (%s)", model, exc.category.value)
        return ModelProbe(model=model, status="inconclusive", detail=str(exc)[:200])
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("preflight: %s inconclusive (%s)", model, type(exc).__name__)
        return ModelProbe(model=model, status="inconclusive", detail=str(exc)[:200])
    return ModelProbe(model=model, status="reachable")


def preflight_models(
    models: list[str],
    *,
    client: LLMClient | None = None,
) -> PreflightReport:
    """Probe each distinct slug in *models* once and return a report.

    Order in the report follows first appearance in *models*. Probes run
    concurrently; the shared :class:`LLMClient` applies any configured
    concurrency / rate-limit throttles.
    """
    distinct: list[str] = list(dict.fromkeys(m for m in models if m))
    if not distinct:
        return PreflightReport()

    probe_client = client or LLMClient()
    results: dict[str, ModelProbe] = {}
    max_workers = min(len(distinct), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_probe_one, probe_client, m): m for m in distinct}
        for fut in futures:
            probe = fut.result()
            results[probe.model] = probe

    return PreflightReport(probes=[results[m] for m in distinct])
