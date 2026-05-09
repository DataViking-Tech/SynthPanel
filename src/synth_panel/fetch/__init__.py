"""URL fetcher subsystem for question/panel attachments.

Implements the I-phase design from hq-hqlp:

- ``perimeter`` — SSRF-hardened HTTP fetcher with magic-byte sniffing,
  size caps, and DNS-rebinding mitigation via pinned-IP transport.
- ``cache`` — content-addressable on-disk cache with LRU eviction and
  per-run in-memory L1.
- ``ladder`` — content-type ladder driven by per-question
  ``attachment_intent``: markdown negotiation → trafilatura HTML →
  Playwright screenshot (visual-only opt-in).

The public API surface is intentionally small; modules are imported on
demand to keep the heavy text-extraction and browser dependencies off
the import path of CLI commands that never touch attachments.
"""

from __future__ import annotations

from synth_panel.fetch.perimeter import (
    ContentTooLarge,
    FetchResult,
    PerimeterDeny,
    ResolvedTarget,
    safe_fetch,
    safe_resolve,
    sniff_and_validate,
)

__all__ = [
    "ContentTooLarge",
    "FetchResult",
    "PerimeterDeny",
    "ResolvedTarget",
    "safe_fetch",
    "safe_resolve",
    "sniff_and_validate",
]
