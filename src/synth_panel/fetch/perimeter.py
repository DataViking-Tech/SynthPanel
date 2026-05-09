"""SSRF-hardened HTTP fetcher (D-phase design hq-hqlp §3).

This module is the trust boundary for any URL synthpanel reaches on
behalf of an LLM panel run. The model may be coaxed into asking for an
arbitrary URL, so every fetch must:

- Resolve hostnames once and reject private / loopback / link-local /
  reserved / multicast / IMDS / CGNAT / IPv6-ULA / IPv4-mapped targets.
- Connect to the *resolved* IP for every redirect hop, defeating
  DNS-rebinding races where a public name later flips to RFC1918.
- Walk redirects manually with a small cap and re-validate per hop.
- Stream bytes with a hard size cap and an httpx Timeout that keeps a
  hostile server from holding a slot indefinitely.
- Sniff magic bytes and reject responses whose actual content type
  disagrees with the declared header or sits outside the allowlist.

The implementation deliberately vendors ~150 LOC of perimeter logic
rather than depending on a third-party SSRF wrapper — keeping the
audit surface inside this repository was the explicit hq-hqlp call.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterable


# Size caps per content kind, in bytes. From hq-hqlp §3.
HTML_MAX_BYTES = 8 * 1024 * 1024  # 8 MiB
PDF_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB
IMAGE_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB

DEFAULT_MAX_BYTES = HTML_MAX_BYTES

# httpx timeout per hop. Total wall-clock budget ≈ 18s.
DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=2.0)

DEFAULT_MAX_REDIRECTS = 3

# Content-type allowlist (lowercased, no parameters). The ladder may
# narrow this further; the perimeter rejects anything outside.
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/html",
        "text/markdown",
        "text/plain",
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)

# Networks that must never be reached from a synthpanel fetch. The
# stdlib ``ipaddress`` flags most of these via ``is_private`` etc., but
# we add explicit nets to be defence-in-depth against future changes.
_EXPLICIT_DENY_V4 = (
    ipaddress.ip_network("169.254.169.254/32"),  # AWS / GCP IMDS
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
)
_EXPLICIT_DENY_V6 = (
    ipaddress.ip_network("fc00::/7"),  # Unique local addresses
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6
)

# Magic-byte → content-type table (used as a fallback when puremagic is
# unavailable, and as a sanity check when it is). Keys are byte
# prefixes; the first matching prefix wins.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class PerimeterDeny(Exception):
    """Raised when a fetch target violates the security perimeter."""


class ContentTooLarge(Exception):
    """Raised when the streaming body exceeds the configured cap."""


@dataclass(frozen=True)
class ResolvedTarget:
    """Outcome of ``safe_resolve`` for a hostname.

    ``ip`` is the textual address that callers must connect to;
    ``host`` is the original hostname (preserved so the SNI / Host
    header can still match the certificate).
    """

    host: str
    ip: str
    family: int  # socket.AF_INET / AF_INET6


@dataclass
class FetchResult:
    """Outcome of a successful ``safe_fetch`` call."""

    url: str  # final URL after redirects
    status_code: int
    content_type: str  # validated, lowercased, no parameters
    declared_content_type: str  # raw Content-Type header, as received
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    redirect_chain: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# safe_resolve
# ---------------------------------------------------------------------------


def _ip_is_denied(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a short reason string if ``ip`` is denied, else ``None``."""
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_private:
        return "private"
    if isinstance(ip, ipaddress.IPv4Address):
        for net in _EXPLICIT_DENY_V4:
            if ip in net:
                return f"explicit-deny:{net}"
    else:
        for net in _EXPLICIT_DENY_V6:
            if ip in net:
                return f"explicit-deny:{net}"
        # IPv4-mapped: also cross-check the embedded IPv4.
        if ip.ipv4_mapped is not None:
            mapped_reason = _ip_is_denied(ip.ipv4_mapped)
            if mapped_reason is not None:
                return f"ipv4-mapped:{mapped_reason}"
    return None


def safe_resolve(host: str) -> ResolvedTarget:
    """Resolve ``host`` to a single IP, rejecting unsafe targets.

    The first address returned by ``socket.getaddrinfo`` that survives
    the denylist is pinned and returned. ``PerimeterDeny`` is raised if
    every address is unsafe or if resolution itself fails.

    Hostnames that are already IP literals are validated in-place
    without DNS lookup so that callers can skip the resolver when they
    have already pinned an address.
    """
    if not host:
        raise PerimeterDeny("empty host")

    # Strip surrounding brackets from IPv6 literals (URL form).
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host

    # If ``host`` is already an IP literal, validate without DNS.
    try:
        ip_obj = ipaddress.ip_address(bare)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        reason = _ip_is_denied(ip_obj)
        if reason is not None:
            raise PerimeterDeny(f"target {bare} denied: {reason}")
        family = socket.AF_INET6 if isinstance(ip_obj, ipaddress.IPv6Address) else socket.AF_INET
        return ResolvedTarget(host=host, ip=str(ip_obj), family=family)

    try:
        infos = socket.getaddrinfo(bare, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PerimeterDeny(f"DNS resolution failed for {host!r}: {exc}") from exc

    last_reason: str | None = None
    for family, _stype, _proto, _canon, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            last_reason = f"unparseable address {ip_str!r}"
            continue
        reason = _ip_is_denied(ip_obj)
        if reason is None:
            return ResolvedTarget(host=host, ip=str(ip_obj), family=family)
        last_reason = reason

    raise PerimeterDeny(f"no safe address for {host!r}: {last_reason or 'no usable records'}")


# ---------------------------------------------------------------------------
# sniff_and_validate
# ---------------------------------------------------------------------------


def _normalise_content_type(raw: str) -> str:
    """Lowercase + strip parameters from a Content-Type header value."""
    primary = raw.split(";", 1)[0].strip().lower()
    return primary


def _puremagic_sniff(head: bytes) -> str | None:
    """Best-effort sniff via puremagic. Returns ``None`` on failure."""
    if not head:
        return None
    try:
        import puremagic  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        guesses = puremagic.magic_string(head)
    except Exception:
        return None
    if not guesses:
        return None
    # puremagic returns a list of namedtuples; the first has the
    # highest confidence. ``mime_type`` may be empty for some matches.
    best = guesses[0]
    mime = getattr(best, "mime_type", None) or ""
    return mime.lower() or None


def _heuristic_sniff(head: bytes) -> str | None:
    """Coarse magic-byte sniff used when puremagic is unavailable."""
    if not head:
        return None
    for prefix, mime in _MAGIC_PREFIXES:
        if head.startswith(prefix):
            return mime
    # RIFF....WEBP: 12-byte structured prefix.
    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    # HTML — fairly permissive.
    snippet = head[:512].lstrip().lower()
    if snippet.startswith(b"<!doctype html") or snippet.startswith(b"<html"):
        return "text/html"
    return None


def sniff_and_validate(
    stream_head: bytes,
    declared_type: str,
    allow: Iterable[str] | None = None,
) -> str:
    """Sniff magic bytes and reconcile with the ``Content-Type`` header.

    Returns the final, lowercased content type. Raises ``PerimeterDeny``
    if the declared and sniffed types disagree, or if the result is
    outside ``allow``.

    ``stream_head`` should be the first ~512 bytes of the body.
    """
    allow_set: frozenset[str] = frozenset(t.lower() for t in allow) if allow is not None else ALLOWED_CONTENT_TYPES

    declared = _normalise_content_type(declared_type or "")
    sniffed = _puremagic_sniff(stream_head) or _heuristic_sniff(stream_head)

    # Text bodies (plain / markdown / minimal HTML) often defy magic-byte
    # sniffing — there is nothing to match. Fall through to the declared
    # type when the body is plausibly text.
    if sniffed is None and declared.startswith("text/"):
        sniffed = declared

    if sniffed is None:
        raise PerimeterDeny(f"unable to sniff content type (declared {declared!r}); refusing opaque payload")

    if sniffed not in allow_set:
        raise PerimeterDeny(f"sniffed content type {sniffed!r} not in allowlist")

    # Reconcile sniffed vs declared. ``text/html`` and ``application/xhtml+xml``
    # are commonly interchangeable; treat declared/sniffed mismatches that
    # both stay inside ``text/*`` as a soft pass when one is text/plain.
    if declared and declared != sniffed:
        # text/* leniency: a server may serve markdown as text/plain.
        text_pair = {declared, sniffed} <= {"text/plain", "text/markdown", "text/html"}
        if not text_pair:
            raise PerimeterDeny(f"declared content type {declared!r} disagrees with sniffed {sniffed!r}")

    return sniffed


# ---------------------------------------------------------------------------
# safe_fetch — DNS-pinned httpx transport
# ---------------------------------------------------------------------------


def _build_pinned_transport(target: ResolvedTarget) -> httpx.BaseTransport:
    """Return an ``httpx`` transport that connects only to ``target.ip``.

    The TLS handshake is still driven by httpcore based on the request
    URL's hostname, so SNI / certificate verification continue to use
    the *original* host name. The custom backend simply ensures the
    underlying socket connects to our pinned IP, defeating DNS rebinding
    that might otherwise flip a public name to RFC1918 between the
    ``safe_resolve`` check and the actual connect().
    """
    # httpcore is httpx's transport backend; we reach into the private
    # ``_backends.sync`` module deliberately. The import is local so a
    # pinned httpcore version isn't required at module import time.
    from httpcore._backends.sync import SyncBackend  # type: ignore[import-not-found]

    pinned_ip = target.ip

    class _PinnedSyncBackend(SyncBackend):  # pragma: no cover - thin override
        def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options=None,
        ):
            return super().connect_tcp(
                pinned_ip,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )

    transport = httpx.HTTPTransport(verify=True, retries=0)
    # The internal connection pool exposes ``_network_backend``; swap
    # it for our pinned backend. This mutation is intentional and
    # confined to this transport instance.
    pool = getattr(transport, "_pool", None)
    if pool is not None and hasattr(pool, "_network_backend"):
        pool._network_backend = _PinnedSyncBackend()
    return transport


def _stream_body(response: httpx.Response, max_bytes: int) -> bytes:
    """Drain ``response`` into bytes, raising before exceeding ``max_bytes``."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ContentTooLarge(f"response body exceeded cap of {max_bytes} bytes (got at least {total})")
        chunks.append(chunk)
    return b"".join(chunks)


def _check_advertised_size(headers: httpx.Headers, max_bytes: int) -> None:
    """Reject a response whose ``Content-Length`` already overshoots."""
    raw = headers.get("content-length")
    if raw is None:
        return
    try:
        advertised = int(raw)
    except ValueError:
        return
    if advertised > max_bytes:
        raise ContentTooLarge(f"server advertised {advertised} bytes, exceeds cap of {max_bytes}")


def _resolve_redirect(base_url: str, location: str) -> str:
    """Resolve a ``Location`` header against ``base_url``."""
    from urllib.parse import urljoin

    return urljoin(base_url, location)


def safe_fetch(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: httpx.Timeout | None = None,
    allow_types: Iterable[str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    user_agent: str = "synthpanel-fetch/1.0 (+https://synthpanel.dev)",
    accept: str = "text/markdown, text/html;q=0.9, */*;q=0.5",
    extra_headers: dict[str, str] | None = None,
) -> FetchResult:
    """Fetch ``url`` through the security perimeter.

    Parameters
    ----------
    url:
        The starting URL. Must use ``http`` or ``https``.
    max_bytes:
        Hard cap on the response body. Streaming aborts as soon as the
        cap is exceeded so a hostile server cannot exhaust memory.
    timeout:
        ``httpx.Timeout`` controlling connect/read/write/pool timeouts.
        Defaults to a tight 3/10/3/2 budget.
    allow_types:
        Iterable of content types to accept. Defaults to the module
        ``ALLOWED_CONTENT_TYPES`` set.
    max_redirects:
        Maximum redirect hops. Each hop is re-resolved and re-pinned.

    Returns
    -------
    FetchResult
        Body and validated metadata for the final response.

    Raises
    ------
    PerimeterDeny
        Any perimeter violation (private IP, content-type mismatch,
        redirect loop, scheme not in {http, https}, etc.).
    ContentTooLarge
        Streaming exceeded ``max_bytes`` or the advertised Content-Length
        was already too large.
    """
    allow = frozenset(t.lower() for t in allow_types) if allow_types is not None else ALLOWED_CONTENT_TYPES
    timeout_cfg = timeout or DEFAULT_TIMEOUT

    redirect_chain: list[str] = []
    current_url = url
    headers = {
        "User-Agent": user_agent,
        "Accept": accept,
    }
    if extra_headers:
        headers.update(extra_headers)

    for hop in range(max_redirects + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https"):
            raise PerimeterDeny(f"unsupported scheme {parsed.scheme!r} in {current_url!r}")
        if not parsed.hostname:
            raise PerimeterDeny(f"missing host in {current_url!r}")

        target = safe_resolve(parsed.hostname)
        transport = _build_pinned_transport(target)

        with (
            httpx.Client(transport=transport, timeout=timeout_cfg, follow_redirects=False) as client,
            client.stream("GET", current_url, headers=headers) as response,
        ):
            if 300 <= response.status_code < 400:
                location = response.headers.get("location")
                if not location:
                    raise PerimeterDeny(f"{response.status_code} redirect without Location header")
                if hop >= max_redirects:
                    raise PerimeterDeny(f"redirect cap of {max_redirects} exceeded at {current_url!r}")
                redirect_chain.append(current_url)
                next_url = _resolve_redirect(current_url, location)
                if next_url in redirect_chain or next_url == current_url:
                    raise PerimeterDeny(f"redirect loop at {next_url!r}")
                current_url = next_url
                continue

            if response.status_code >= 400:
                raise PerimeterDeny(f"HTTP {response.status_code} from {current_url!r}")

            _check_advertised_size(response.headers, max_bytes)
            declared = response.headers.get("content-type", "")
            body = _stream_body(response, max_bytes)
            content_type = sniff_and_validate(body[:512], declared, allow)

            # Normalise the final URL — drop fragments, keep query.
            final = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path or "/",
                    parsed.params,
                    parsed.query,
                    "",
                )
            )
            return FetchResult(
                url=final,
                status_code=response.status_code,
                content_type=content_type,
                declared_content_type=declared,
                body=body,
                headers={k.lower(): v for k, v in response.headers.items()},
                redirect_chain=list(redirect_chain),
            )

    # Should be unreachable — the loop returns or raises on every path.
    raise PerimeterDeny(f"redirect cap of {max_redirects} exhausted without response")
