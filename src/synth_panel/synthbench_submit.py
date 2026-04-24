"""Opt-in submission of calibrated panel runs to SynthBench (sp-ezz).

SynthBench's submission API (``POST {SYNTHBENCH_API_URL}/submit``) expects a
benchmark-shaped payload: ``{benchmark, config, aggregate, per_question, ...}``.
A bare SynthPanel run does not produce this shape — it is qualitative output.
A run made with ``--calibrate-against gss:HAPPY`` (sp-inline-calibration) DOES
produce per-question JSD against a known human baseline, which is the
SynthBench currency. So the submission flow is gated on that flag.

This module is responsible for three things:

1.  **Consent** — first time a user opts in we print a one-screen privacy
    notice and record acceptance at ``~/.synthpanel/synthbench-consent.json``
    so subsequent runs do not re-prompt. ``--yes`` bypasses the prompt for
    CI / non-interactive use.
2.  **Payload transformation** — turn the SynthPanel ``convergence`` report
    plus the loaded baseline payload into a SynthBench-shaped JSON dict.
3.  **HTTP POST** — bearer-auth POST to ``/submit`` with a bounded timeout,
    returning a structured :class:`SubmissionResult`. The panel run itself
    never fails on submission errors; the submitter logs a warning instead.

The module deliberately does not import from :mod:`synth_panel.convergence`
beyond the public ``ConvergenceTracker.cumulative_distributions()`` helper
so a future refactor of the tracker's internals does not ripple here.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from synth_panel.__version__ import __version__

DEFAULT_API_URL = "https://api.synthbench.org"
API_URL_ENV = "SYNTHBENCH_API_URL"
API_KEY_ENV = "SYNTHBENCH_API_KEY"
ACCOUNT_URL = "https://synthbench.org/account"
SUBMIT_TIMEOUT = 30.0

CONSENT_DIR = Path("~/.synthpanel").expanduser()
CONSENT_PATH = CONSENT_DIR / "synthbench-consent.json"
CONSENT_VERSION = 1

CONSENT_NOTICE = """
SynthBench submission consent
─────────────────────────────
You opted in to upload this calibrated panel run to SynthBench's public
benchmark leaderboard.

What gets uploaded:
  • Per-question categorical response distributions (the model_distribution
    used to compute calibration JSD).
  • The calibration spec (e.g. 'gss:HAPPY'), extractor label, and panel
    sample size n.
  • Run config: model identifier(s), persona pack name, instrument name.
  • The SynthPanel client version.

What does NOT get uploaded:
  • Free-text panelist responses or follow-ups.
  • Persona definitions, system prompts, or any persona attributes.
  • API keys, file paths, or local environment data.

Do not use --submit-to-synthbench with confidential personas, proprietary
instruments, or topics you would not publish on a public leaderboard.

This consent is recorded at ~/.synthpanel/synthbench-consent.json so you
will not be re-prompted. Pass --yes to bypass this prompt in CI.
"""


# ---------------------------------------------------------------------------
# Errors and result types
# ---------------------------------------------------------------------------


class SynthBenchSubmissionError(Exception):
    """Raised for terminal submission failures the CLI should surface."""


@dataclass
class SubmissionResult:
    """Outcome of a single ``/submit`` POST.

    ``submission_id`` and ``leaderboard_url`` are populated only on success.
    ``status`` mirrors what the server returned (``accepted``, ``validating``,
    ``rejected``, etc.); on a transport failure it is ``"error"``.
    """

    accepted: bool
    status: str
    submission_id: str | None = None
    leaderboard_url: str | None = None
    error: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Consent handling
# ---------------------------------------------------------------------------


def _load_consent() -> dict[str, Any] | None:
    """Return the recorded consent dict, or ``None`` if no record exists."""
    try:
        with CONSENT_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # A corrupted consent file is treated as absent so the user is
        # re-prompted rather than silently submitting on stale state.
        return None
    if not isinstance(data, dict):
        return None
    return data


def consent_recorded() -> bool:
    """``True`` when a valid consent record (matching ``CONSENT_VERSION``) exists."""
    data = _load_consent()
    if data is None:
        return False
    return data.get("version") == CONSENT_VERSION and data.get("accepted") is True


def record_consent() -> None:
    """Persist a positive consent record at ``CONSENT_PATH``."""
    CONSENT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CONSENT_VERSION,
        "accepted": True,
        "client_version": __version__,
    }
    with CONSENT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def prompt_consent(*, stream: Any = None, input_fn: Any = None) -> bool:
    """Show the consent notice and read ``y/N`` from stdin.

    ``stream`` and ``input_fn`` are injection points for tests. Default
    ``stream`` is stderr (so the prompt does not pollute JSON stdout) and
    default ``input_fn`` is the builtin :func:`input`.
    """
    out = stream if stream is not None else sys.stderr
    reader = input_fn if input_fn is not None else input
    print(CONSENT_NOTICE, file=out)
    print("Continue? [y/N] ", end="", file=out, flush=True)
    try:
        answer = reader("")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


# ---------------------------------------------------------------------------
# Payload transformation
# ---------------------------------------------------------------------------


def _normalize_distribution(dist: dict[str, Any]) -> dict[str, float]:
    """Coerce + renormalize a distribution to floats summing to 1.0.

    Returns an empty dict when the input has no positive mass — callers
    should treat that as "no usable distribution" rather than divide-by-zero.
    """
    coerced = {str(k): float(v) for k, v in dist.items() if v is not None}
    total = sum(v for v in coerced.values() if v > 0)
    if total <= 0:
        return {}
    return {k: v / total for k, v in coerced.items() if v > 0}


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean; ``None`` when the list is empty so we can omit the field."""
    if not values:
        return None
    return sum(values) / len(values)


def build_submission_payload(
    *,
    panel_extra: dict[str, Any],
    calibration_spec: str,
    baseline_payload: dict[str, Any] | None,
    model_distributions: dict[str, dict[str, float]],
    panelist_model: str | None,
    instrument_name: str | None,
    persona_pack_name: str | None,
) -> dict[str, Any]:
    """Assemble a SynthBench ``/submit`` payload from a calibrated panel run.

    The shape is best-effort against the SynthBench Tier 2 schema documented
    in ``SUBMISSIONS.md`` (see bead sp-ezz). When the server cannot validate
    a field — for example, ``mean_tau`` for non-rank questions — we omit the
    field rather than send a placeholder; the server can choose to accept or
    reject. Server-side rejections are surfaced verbatim by :func:`submit`.
    """
    convergence = panel_extra.get("convergence") or {}
    per_question_in = convergence.get("per_question") or {}
    final_n = convergence.get("final_n", 0)

    human_dists: dict[str, dict[str, float]] = {}
    if isinstance(baseline_payload, dict):
        # Supported shapes:
        #   {"human_distribution": {...}}                          (single-question)
        #   {"per_question": {key: {"human_distribution": ...}}}   (multi-question)
        if isinstance(baseline_payload.get("human_distribution"), dict):
            for key in per_question_in:
                human_dists[key] = _normalize_distribution(baseline_payload["human_distribution"])
        elif isinstance(baseline_payload.get("per_question"), dict):
            for key, sub in baseline_payload["per_question"].items():
                if isinstance(sub, dict) and isinstance(sub.get("human_distribution"), dict):
                    human_dists[key] = _normalize_distribution(sub["human_distribution"])

    per_question_out: dict[str, Any] = {}
    jsd_values: list[float] = []
    for key, q_data in per_question_in.items():
        calib = q_data.get("calibration") if isinstance(q_data, dict) else None
        if not isinstance(calib, dict):
            # Skip questions for which calibration was not computed (e.g.
            # disjoint supports surfaced via alignment_error are still
            # included since calib will exist with jsd=1.0). Absence here
            # means the question was not on the calibration path at all.
            continue
        model_dist = _normalize_distribution(model_distributions.get(key, {}))
        human_dist = human_dists.get(key, {})
        if not model_dist or not human_dist:
            continue
        jsd_value = float(calib.get("jsd", 0.0))
        entry = {
            "model_distribution": model_dist,
            "human_distribution": human_dist,
            "jsd": jsd_value,
            "n": final_n,
            "extractor": calib.get("extractor"),
            "auto_derived": bool(calib.get("auto_derived", False)),
        }
        if "alignment_error" in calib:
            entry["alignment_error"] = calib["alignment_error"]
        per_question_out[key] = entry
        jsd_values.append(jsd_value)

    aggregate: dict[str, Any] = {"n": final_n}
    mean_jsd = _mean(jsd_values)
    if mean_jsd is not None:
        aggregate["mean_jsd"] = mean_jsd

    config: dict[str, Any] = {
        "calibration_spec": calibration_spec,
        "n": final_n,
        "client": "synthpanel",
        "client_version": __version__,
    }
    if panelist_model:
        config["panelist_model"] = panelist_model
    if instrument_name:
        config["instrument"] = instrument_name
    if persona_pack_name:
        config["persona_pack"] = persona_pack_name

    return {
        "benchmark": "synthbench",
        "config": config,
        "aggregate": aggregate,
        "per_question": per_question_out,
    }


def is_submittable(panel_extra: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(ok, reason)`` describing whether ``panel_extra`` is submittable.

    A panel run is submittable only when convergence ran, the calibration
    block carried a baseline_spec, and at least one per-question entry has
    a calibration JSD (otherwise SynthBench has nothing to score).
    """
    if panel_extra.get("run_invalid"):
        return False, "panel run was marked invalid; refusing to submit"
    convergence = panel_extra.get("convergence")
    if not isinstance(convergence, dict):
        return False, "no convergence report attached (run with --calibrate-against)"
    per_q = convergence.get("per_question") or {}
    has_calibration = any(isinstance(q, dict) and isinstance(q.get("calibration"), dict) for q in per_q.values())
    if not has_calibration:
        return False, "no per-question calibration data (no JSD computed)"
    return True, None


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def _api_base() -> str:
    return os.environ.get(API_URL_ENV, DEFAULT_API_URL).rstrip("/")


def submit(
    payload: dict[str, Any],
    *,
    api_key: str,
    api_url: str | None = None,
    client: httpx.Client | None = None,
) -> SubmissionResult:
    """POST ``payload`` to ``{api_url}/submit`` with bearer auth.

    On 2xx the server response is parsed as JSON and an accepted result is
    returned. On any other status the body is captured into ``error`` so
    the user sees the server's specific Tier-2 validation error rather than
    a generic HTTP code. Network errors surface as ``status="error"``.

    ``client`` is an injection point for tests (use ``httpx.MockTransport``).
    """
    base = (api_url or _api_base()).rstrip("/")
    url = f"{base}/submit"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"synthpanel/{__version__}",
    }
    owns_client = client is None
    client = client or httpx.Client(timeout=SUBMIT_TIMEOUT)
    try:
        try:
            resp = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            return SubmissionResult(accepted=False, status="error", error=str(exc))

        body: dict[str, Any]
        try:
            parsed = resp.json()
            body = parsed if isinstance(parsed, dict) else {"raw": parsed}
        except ValueError:
            body = {"raw": resp.text}

        if 200 <= resp.status_code < 300:
            submission_id = body.get("id") or body.get("submission_id")
            status = str(body.get("status", "accepted"))
            leaderboard_url = body.get("leaderboard_url") or body.get("url")
            if leaderboard_url is None and submission_id:
                # Conventional public URL when the server does not echo one.
                leaderboard_url = f"https://synthbench.org/submit/{submission_id}"
            return SubmissionResult(
                accepted=True,
                status=status,
                submission_id=str(submission_id) if submission_id else None,
                leaderboard_url=leaderboard_url,
                raw_response=body,
            )

        # Surface the server's validation error verbatim so the user can act
        # on it. Common cases: 401 (bad key), 422 (Tier-2 schema mismatch).
        err_msg = body.get("error") or body.get("detail") or body.get("message")
        return SubmissionResult(
            accepted=False,
            status=f"http_{resp.status_code}",
            error=str(err_msg) if err_msg else f"HTTP {resp.status_code}: {resp.text[:200]}",
            raw_response=body,
        )
    finally:
        if owns_client:
            client.close()


# ---------------------------------------------------------------------------
# Top-level CLI entrypoint
# ---------------------------------------------------------------------------


def submit_panel_result(
    *,
    panel_extra: dict[str, Any],
    calibration_spec: str,
    baseline_payload: dict[str, Any] | None,
    model_distributions: dict[str, dict[str, float]],
    panelist_model: str | None = None,
    instrument_name: str | None = None,
    persona_pack_name: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    skip_consent: bool = False,
    client: httpx.Client | None = None,
    stderr: Any = None,
) -> SubmissionResult:
    """Top-level entry called from the CLI after a panel run completes.

    Returns a :class:`SubmissionResult`. The caller is responsible for the
    "warn-but-don't-fail" semantic — this function never raises on a
    submission error so a slow SynthBench cannot turn a successful panel
    run into a failed CLI exit.
    """
    out = stderr if stderr is not None else sys.stderr

    ok, reason = is_submittable(panel_extra)
    if not ok:
        return SubmissionResult(accepted=False, status="not_submittable", error=reason)

    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        return SubmissionResult(
            accepted=False,
            status="missing_api_key",
            error=f"{API_KEY_ENV} not set; mint a key at {ACCOUNT_URL}",
        )

    if not skip_consent and not consent_recorded():
        if not prompt_consent(stream=out):
            return SubmissionResult(
                accepted=False,
                status="consent_declined",
                error="user declined SynthBench upload consent",
            )
        record_consent()

    payload = build_submission_payload(
        panel_extra=panel_extra,
        calibration_spec=calibration_spec,
        baseline_payload=baseline_payload,
        model_distributions=model_distributions,
        panelist_model=panelist_model,
        instrument_name=instrument_name,
        persona_pack_name=persona_pack_name,
    )
    if not payload["per_question"]:
        return SubmissionResult(
            accepted=False,
            status="empty_payload",
            error="no per-question calibration entries had both a model and human distribution",
        )
    return submit(payload, api_key=key, api_url=api_url, client=client)
