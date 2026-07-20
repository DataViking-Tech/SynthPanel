"""PDF attachment ingest decision tree.

Implements the design in hq-31t8 / I-phase hq-glz6: pick between native
PDF submission, text extraction, or page-as-image rendering using cheap
local probes before paying any LLM cost.

Decision tree (defaults — every threshold is configurable via
:class:`PdfOptions`):

* encrypted PDF → :class:`PdfEncryptedError` (Anthropic native does not
  support password-protected PDFs).
* file size > 32 MiB or page count > native cap (600 / 100 on 200k-ctx
  models) → text-extract path with
  ``"pdf_too_large_for_native_visual; using_text_extraction"`` warning.
  If the same PDF is also scanned (text density below threshold), reject
  with :class:`PdfOversizeScannedError` — page-as-image of that many
  pages would blow the cost cliff.
* avg chars/page < 32 → SCANNED → page-as-image at 150 DPI (PNG).
* 32 ≤ avg chars/page < 200 AND image coverage ≥ 0.5 → MIXED → native
  PDF API (Anthropic does both extract + image, pay the safety net).
* avg chars/page ≥ 200 within native limits → NATIVE (default).

Submission mode (recommendation only — actual upload lives in the LLM
adapter): inline base64 below 4 MiB, Files API ``file_id`` above.

The 32-chars/page scanned threshold matches the cross-rig precedent in
``traitprint_cloud/.../document-parser.ts`` (``SCANNED_PDF_MIN_CHARS``)
so panels share the same scanned/text-bearing classification regardless
of which rig ingested the PDF.

``pypdfium2`` and ``Pillow`` are imported lazily so callers that never
touch PDFs do not pay the wheel cost. Install with
``pip install althing[pdf]``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PdfError(Exception):
    """Base class for PDF ingest errors."""


class PdfEncryptedError(PdfError):
    """Raised when a PDF is encrypted/password-protected.

    Anthropic's native PDF support explicitly rejects encrypted PDFs and
    we have no in-pipeline way to recover, so the caller must unlock the
    PDF before resubmitting.
    """


class PdfOversizeScannedError(PdfError):
    """Raised when a scanned PDF also exceeds native limits.

    Rendering >600 scanned pages at 150 DPI is roughly 2.3M tokens —
    well past any context window and an obvious cost cliff. We refuse
    rather than guess at a truncation.
    """


class PdfMissingDependencyError(PdfError):
    """Raised when ``pypdfium2`` (or ``Pillow``) is not installed.

    Install with ``pip install althing[pdf]``.
    """


# ---------------------------------------------------------------------------
# Options + result types
# ---------------------------------------------------------------------------


class PdfStrategy(Enum):
    """How a PDF should be presented to the model."""

    NATIVE = "native"
    TEXT_EXTRACT = "text_extract"
    PAGE_AS_IMAGE = "page_as_image"


class SubmissionMode(Enum):
    """Recommended transport for the bytes of a NATIVE-strategy PDF.

    The actual upload to the Files API or base64 encoding happens in the
    LLM adapter; the planner only emits a recommendation so cost
    previews can show the right path. Files API is preferred above
    4 MiB to keep request payloads small.
    """

    INLINE_BASE64 = "inline_base64"
    FILES_API = "files_api"


@dataclass(frozen=True)
class PdfOptions:
    """Configurable thresholds for the decision tree.

    All defaults match the D-phase design (hq-31t8 §2). Override per
    call when an instrument needs different limits — for example, a
    panel run pinned to a 200k-context model should set
    ``native_max_pages=100``.
    """

    native_max_pages: int = 600
    native_max_pages_200k_ctx: int = 100
    native_max_bytes: int = 32 * 1024 * 1024
    scanned_density_threshold: int = 32
    mixed_density_low: int = 32
    mixed_density_high: int = 200
    mixed_image_coverage: float = 0.5
    files_api_threshold_bytes: int = 4 * 1024 * 1024
    render_dpi: int = 150
    probe_pages: int = 5
    chunk_token_limit: int = 80_000
    cost_cliff_page_threshold: int = 20
    truncate_pages: int | None = None


DEFAULT_OPTIONS = PdfOptions()


@dataclass(frozen=True)
class PdfProbe:
    """Cheap local-probe results, prior to any submission decision."""

    page_count: int
    file_size_bytes: int
    avg_chars_per_page: float
    image_coverage: float | None
    is_encrypted: bool


@dataclass(frozen=True)
class PdfPlan:
    """Decision-tree output describing how to submit this PDF."""

    strategy: PdfStrategy
    submission: SubmissionMode | None
    probe: PdfProbe
    warnings: tuple[str, ...] = ()
    estimated_input_tokens: int = 0
    cost_cliff: bool = False
    options: PdfOptions = field(default_factory=lambda: DEFAULT_OPTIONS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_pdf(
    data: bytes,
    *,
    options: PdfOptions = DEFAULT_OPTIONS,
    password: str | None = None,
) -> PdfProbe:
    """Run cheap local probes on ``data``.

    Opens the PDF with ``pypdfium2``, samples up to
    ``options.probe_pages`` pages for text density, and computes image
    coverage only when the density falls in the ambiguous band
    (cheap/expensive split — see hq-31t8 §4).

    Raises :class:`PdfEncryptedError` if the PDF is encrypted and no
    correct password was supplied. Other PDFium errors are wrapped as
    :class:`PdfError`.
    """
    pdfium = _import_pdfium()
    file_size = len(data)
    doc = _open_document(pdfium, data, password)
    try:
        page_count = len(doc)
        if page_count == 0:
            return PdfProbe(
                page_count=0,
                file_size_bytes=file_size,
                avg_chars_per_page=0.0,
                image_coverage=None,
                is_encrypted=False,
            )

        sample_n = min(page_count, max(1, options.probe_pages))
        total_chars = 0
        for i in range(sample_n):
            page = doc.get_page(i)
            try:
                textpage = page.get_textpage()
                try:
                    total_chars += textpage.count_chars()
                finally:
                    textpage.close()
            finally:
                page.close()
        avg_chars = total_chars / sample_n

        image_coverage: float | None = None
        if options.mixed_density_low <= avg_chars < options.mixed_density_high:
            image_coverage = _measure_image_coverage(doc, sample_n)

        return PdfProbe(
            page_count=page_count,
            file_size_bytes=file_size,
            avg_chars_per_page=avg_chars,
            image_coverage=image_coverage,
            is_encrypted=False,
        )
    finally:
        doc.close()


def plan_pdf(
    data: bytes,
    *,
    options: PdfOptions = DEFAULT_OPTIONS,
    model_context: int = 1_000_000,
    password: str | None = None,
) -> PdfPlan:
    """Run the full ingest decision tree against ``data``.

    Returns a :class:`PdfPlan` with the chosen strategy, submission
    recommendation, warnings, and a token-cost estimate suitable for
    surfacing in instrument-load cost previews.

    ``model_context`` selects the native page cap (200k-context models
    use ``options.native_max_pages_200k_ctx``).
    """
    probe = probe_pdf(data, options=options, password=password)
    return _decide(probe, options=options, model_context=model_context)


def extract_text_chunks(
    data: bytes,
    *,
    options: PdfOptions = DEFAULT_OPTIONS,
    password: str | None = None,
) -> list[str]:
    """Extract per-page text and chunk it at ``options.chunk_token_limit``.

    Used for the TEXT_EXTRACT path. Chunks preserve page boundaries:
    each chunk holds whole pages joined by ``\\n\\n``. Token count is
    approximated as ``len(chars) // 4`` — good enough for the chunking
    boundary, but the planner's ``estimated_input_tokens`` is what gets
    surfaced to the cost preview.
    """
    pdfium = _import_pdfium()
    doc = _open_document(pdfium, data, password)
    try:
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0
        char_budget = max(1024, options.chunk_token_limit * 4)
        for i in range(len(doc)):
            page = doc.get_page(i)
            try:
                textpage = page.get_textpage()
                try:
                    page_text = textpage.get_text_range()
                finally:
                    textpage.close()
            finally:
                page.close()
            if current and current_chars + len(page_text) > char_budget:
                chunks.append("\n\n".join(current))
                current = []
                current_chars = 0
            current.append(page_text)
            current_chars += len(page_text)
        if current:
            chunks.append("\n\n".join(current))
        return chunks
    finally:
        doc.close()


def render_pages_as_png(
    data: bytes,
    *,
    options: PdfOptions = DEFAULT_OPTIONS,
    password: str | None = None,
) -> list[bytes]:
    """Render every page to a PNG byte string at ``options.render_dpi``.

    Used for the PAGE_AS_IMAGE path (scanned PDFs). Requires Pillow —
    raises :class:`PdfMissingDependencyError` otherwise. The result is
    list-of-bytes (one PNG per page) so callers can batch them into
    image content blocks in document order.
    """
    pdfium = _import_pdfium()
    _import_pillow()  # surface a typed PdfMissingDependencyError before we open the doc
    doc = _open_document(pdfium, data, password)
    try:
        scale = options.render_dpi / 72.0
        out: list[bytes] = []
        for i in range(len(doc)):
            page = doc.get_page(i)
            try:
                bitmap = page.render(scale=scale)
                try:
                    pil = bitmap.to_pil()
                finally:
                    bitmap.close()
            finally:
                page.close()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _import_pdfium():
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise PdfMissingDependencyError(
            "pypdfium2 is required for PDF attachments. Install with `pip install althing[pdf]`."
        ) from exc
    return pdfium


def _import_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise PdfMissingDependencyError(
            "Pillow is required to render PDF pages as images. Install with `pip install althing[pdf]`."
        ) from exc
    return Image


def _open_document(pdfium, data: bytes, password: str | None):
    """Open ``data`` as a PDF, mapping pdfium errors to typed errors.

    PDFium reports password failures via a numeric error code on
    :class:`pypdfium2.PdfiumError`. We translate the code into either
    :class:`PdfEncryptedError` (codes 4 = bad password, 5 = unsupported
    security scheme) or :class:`PdfError` for everything else.
    """
    try:
        return pdfium.PdfDocument(io.BytesIO(data), password=password, autoclose=True)
    except pdfium.PdfiumError as exc:
        err_code = getattr(exc, "err_code", None)
        if err_code in (4, 5) or _looks_like_password_error(str(exc)):
            raise PdfEncryptedError(
                "PDF is encrypted/password-protected; Anthropic native PDF support cannot ingest encrypted PDFs."
            ) from exc
        raise PdfError(f"Failed to open PDF: {exc}") from exc


def _looks_like_password_error(msg: str) -> bool:
    lowered = msg.lower()
    return "password" in lowered or "security" in lowered


def _measure_image_coverage(doc, sample_n: int) -> float:
    """Image-coverage corroborator for the ambiguous text-density band.

    Returns the fraction of sampled pages whose XObject image bboxes
    cover at least 95% of the page rectangle (PyMuPDF discussion #1653
    threshold). We look at one page at a time so a single huge image
    on one page does not skew the verdict.
    """
    if sample_n == 0:
        return 0.0
    image_pages = 0
    for i in range(sample_n):
        page = doc.get_page(i)
        try:
            page_w = page.get_width()
            page_h = page.get_height()
            page_area = max(page_w * page_h, 1.0)
            covered_area = 0.0
            for obj in page.get_objects(filter=()):
                try:
                    if not _is_image_object(obj):
                        continue
                    left, bottom, right, top = obj.get_bounds()
                    covered_area += max(0.0, (right - left) * (top - bottom))
                except Exception:
                    continue
            if covered_area / page_area >= 0.95:
                image_pages += 1
        finally:
            page.close()
    return image_pages / sample_n


def _is_image_object(obj) -> bool:
    """Heuristic test for "this PdfObject is a raster image".

    pypdfium2 exposes object type via the FPDF_PAGEOBJ_* constants
    (``type`` attribute). Type 3 = FPDF_PAGEOBJ_IMAGE. We also accept
    :class:`pypdfium2.PdfImage` instances directly in case the
    high-level wrapper is used.
    """
    try:
        import pypdfium2 as pdfium

        if isinstance(obj, pdfium.PdfImage):
            return True
    except ImportError:
        pass
    return getattr(obj, "type", None) == 3


def _decide(
    probe: PdfProbe,
    *,
    options: PdfOptions,
    model_context: int,
) -> PdfPlan:
    """Pure decision tree from a probe to a plan.

    Separated from :func:`probe_pdf` so the logic is testable without
    constructing real PDFs.
    """
    native_cap = options.native_max_pages_200k_ctx if model_context <= 200_000 else options.native_max_pages
    is_oversize = probe.file_size_bytes > options.native_max_bytes or probe.page_count > native_cap
    is_scanned = probe.avg_chars_per_page < options.scanned_density_threshold
    is_mixed = (
        options.mixed_density_low <= probe.avg_chars_per_page < options.mixed_density_high
        and (probe.image_coverage or 0.0) >= options.mixed_image_coverage
    )

    warnings: list[str] = []

    if is_oversize and is_scanned:
        raise PdfOversizeScannedError(
            f"Scanned PDF exceeds native limits (pages={probe.page_count}, "
            f"size={probe.file_size_bytes} bytes, density={probe.avg_chars_per_page:.1f} chars/page); "
            "rendering all pages as images would overflow context. "
            "Pre-process (OCR + extract) and resubmit, or set "
            "PdfOptions.truncate_pages explicitly."
        )

    if is_oversize:
        if options.truncate_pages is None:
            warnings.append("pdf_too_large_for_native_visual; using_text_extraction")
            strategy = PdfStrategy.TEXT_EXTRACT
            submission = None
        else:
            warnings.append(f"pdf_truncated_to_first_{options.truncate_pages}_pages; explicit truncate_pages opt-in")
            strategy = PdfStrategy.NATIVE
            submission = _pick_submission(probe.file_size_bytes, options)
    elif is_scanned:
        strategy = PdfStrategy.PAGE_AS_IMAGE
        submission = SubmissionMode.INLINE_BASE64
    elif is_mixed:
        strategy = PdfStrategy.NATIVE
        submission = _pick_submission(probe.file_size_bytes, options)
    else:
        strategy = PdfStrategy.NATIVE
        submission = _pick_submission(probe.file_size_bytes, options)

    estimated_tokens = _estimate_tokens(strategy, probe, options)
    cost_cliff = _is_cost_cliff(strategy, probe, options)

    return PdfPlan(
        strategy=strategy,
        submission=submission,
        probe=probe,
        warnings=tuple(warnings),
        estimated_input_tokens=estimated_tokens,
        cost_cliff=cost_cliff,
        options=options,
    )


def _pick_submission(file_size_bytes: int, options: PdfOptions) -> SubmissionMode:
    if file_size_bytes > options.files_api_threshold_bytes:
        return SubmissionMode.FILES_API
    return SubmissionMode.INLINE_BASE64


def _estimate_tokens(
    strategy: PdfStrategy,
    probe: PdfProbe,
    options: PdfOptions,
) -> int:
    """Rough per-PDF input-token estimate for cost previews.

    Per-page tokens: native ~2,250 (text + image), text-extract ~500,
    page-as-image at 150 DPI on letter-size ~2,900 (D-phase §5).
    """
    if probe.page_count <= 0:
        return 0
    pages = probe.page_count
    if options.truncate_pages is not None:
        pages = min(pages, options.truncate_pages)
    if strategy is PdfStrategy.NATIVE:
        return pages * 2_250
    if strategy is PdfStrategy.TEXT_EXTRACT:
        return pages * 500
    return pages * 2_900


def _is_cost_cliff(
    strategy: PdfStrategy,
    probe: PdfProbe,
    options: PdfOptions,
) -> bool:
    """Flag combinations that should require a confirmation gate.

    Scanned PDFs above ~20 pages dominate panel cost on Opus 4.7
    (D-phase cliff watch: 50 pages by 50 panelists ~= $108 uncached).
    Cost-preview consumers should hard-gate when this returns True.
    """
    return strategy is PdfStrategy.PAGE_AS_IMAGE and probe.page_count >= options.cost_cliff_page_threshold


def decide_from_probe(
    probe: PdfProbe,
    *,
    options: PdfOptions = DEFAULT_OPTIONS,
    model_context: int = 1_000_000,
) -> PdfPlan:
    """Run the decision tree against an externally produced probe.

    Use this when the probe was computed upstream (e.g. cached in a
    panel result) and you only need to re-evaluate the strategy under
    different options or model context.
    """
    return _decide(probe, options=options, model_context=model_context)
