"""Tests for the PDF attachment ingest decision tree (hq-glz6 / hq-31t8).

Real-world PDF fixtures land in the sibling I-phase bead hq-3o1r. These
tests focus on:

* the pure decision tree (every branch is exercised via
  :func:`decide_from_probe` so the logic is covered without needing
  fixture PDFs),
* error paths (encrypted detection, oversize+scanned rejection,
  missing-dependency translation),
* a smoke test of probe / render / extract on a runtime-generated PDF
  so the pypdfium2 wiring stays honest.
"""

from __future__ import annotations

import io

import pytest

from althing.attachments import (
    DEFAULT_OPTIONS,
    PdfEncryptedError,
    PdfError,
    PdfMissingDependencyError,
    PdfOptions,
    PdfOversizeScannedError,
    PdfPlan,
    PdfProbe,
    PdfStrategy,
    SubmissionMode,
    decide_from_probe,
    extract_text_chunks,
    plan_pdf,
    probe_pdf,
    render_pages_as_png,
)

pdfium = pytest.importorskip("pypdfium2", reason="pypdfium2 is the [pdf] extra")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """A 1-page blank letter-size PDF (no text — looks scanned)."""
    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def multipage_blank_pdf_bytes() -> bytes:
    """A 5-page blank letter-size PDF."""
    doc = pdfium.PdfDocument.new()
    for _ in range(5):
        doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _probe(
    *,
    page_count: int = 10,
    file_size: int = 1_000_000,
    avg_chars: float = 1000.0,
    image_coverage: float | None = None,
) -> PdfProbe:
    return PdfProbe(
        page_count=page_count,
        file_size_bytes=file_size,
        avg_chars_per_page=avg_chars,
        image_coverage=image_coverage,
        is_encrypted=False,
    )


# ---------------------------------------------------------------------------
# Decision tree — pure logic, exercised via decide_from_probe
# ---------------------------------------------------------------------------


class TestDecisionTree:
    def test_text_bearing_within_limits_picks_native(self) -> None:
        plan = decide_from_probe(_probe(page_count=10, avg_chars=1500.0))
        assert plan.strategy is PdfStrategy.NATIVE
        assert plan.submission is SubmissionMode.INLINE_BASE64
        assert plan.warnings == ()
        assert not plan.cost_cliff

    def test_native_over_files_api_threshold_picks_files_api(self) -> None:
        plan = decide_from_probe(_probe(page_count=10, file_size=10 * 1024 * 1024, avg_chars=1500.0))
        assert plan.strategy is PdfStrategy.NATIVE
        assert plan.submission is SubmissionMode.FILES_API

    def test_scanned_pdf_picks_page_as_image(self) -> None:
        plan = decide_from_probe(_probe(page_count=5, avg_chars=10.0))
        assert plan.strategy is PdfStrategy.PAGE_AS_IMAGE
        assert plan.submission is SubmissionMode.INLINE_BASE64
        assert not plan.cost_cliff

    def test_scanned_density_threshold_is_32_chars_per_page(self) -> None:
        """Matches traitprint_cloud SCANNED_PDF_MIN_CHARS for cross-rig consistency."""
        # 31 chars/page -> SCANNED
        scanned = decide_from_probe(_probe(page_count=5, avg_chars=31.0))
        assert scanned.strategy is PdfStrategy.PAGE_AS_IMAGE
        # 32 chars/page -> not scanned (exactly at the boundary, falls into mixed band)
        not_scanned = decide_from_probe(_probe(page_count=5, avg_chars=32.0, image_coverage=0.0))
        assert not_scanned.strategy is PdfStrategy.NATIVE

    def test_mixed_band_with_image_coverage_picks_native(self) -> None:
        """32 ≤ avg_chars < 200 + image_coverage ≥ 0.5 → NATIVE (Anthropic safety net)."""
        plan = decide_from_probe(_probe(page_count=10, avg_chars=100.0, image_coverage=0.7))
        assert plan.strategy is PdfStrategy.NATIVE

    def test_mixed_band_without_image_coverage_still_native(self) -> None:
        """Below 200 chars but no image evidence still ends up native (not text-extract)."""
        plan = decide_from_probe(_probe(page_count=10, avg_chars=100.0, image_coverage=0.1))
        assert plan.strategy is PdfStrategy.NATIVE

    def test_oversize_text_bearing_falls_back_to_text_extract(self) -> None:
        plan = decide_from_probe(_probe(page_count=700, file_size=10_000_000, avg_chars=500.0))
        assert plan.strategy is PdfStrategy.TEXT_EXTRACT
        assert plan.submission is None
        assert plan.warnings == ("pdf_too_large_for_native_visual; using_text_extraction",)

    def test_oversize_by_file_size_falls_back_to_text_extract(self) -> None:
        plan = decide_from_probe(_probe(page_count=50, file_size=40 * 1024 * 1024, avg_chars=500.0))
        assert plan.strategy is PdfStrategy.TEXT_EXTRACT

    def test_200k_context_uses_100_page_native_cap(self) -> None:
        """200k-context models cap at 100 pages, so 150 pages is oversize there."""
        probe = _probe(page_count=150, file_size=2_000_000, avg_chars=500.0)
        plan_1m = decide_from_probe(probe, model_context=1_000_000)
        plan_200k = decide_from_probe(probe, model_context=200_000)
        assert plan_1m.strategy is PdfStrategy.NATIVE
        assert plan_200k.strategy is PdfStrategy.TEXT_EXTRACT

    def test_oversize_and_scanned_raises(self) -> None:
        with pytest.raises(PdfOversizeScannedError, match="Scanned PDF exceeds native limits"):
            decide_from_probe(_probe(page_count=700, file_size=10_000_000, avg_chars=10.0))

    def test_truncate_pages_opt_in_keeps_native_with_warning(self) -> None:
        opts = PdfOptions(truncate_pages=50)
        plan = decide_from_probe(_probe(page_count=700, avg_chars=500.0), options=opts)
        assert plan.strategy is PdfStrategy.NATIVE
        assert any("pdf_truncated_to_first_50_pages" in w for w in plan.warnings)

    def test_cost_cliff_flag_set_for_large_scanned(self) -> None:
        """Scanned >= 20 pages = $108+/panel cliff on Opus 4.7 (D-phase §5)."""
        plan = decide_from_probe(_probe(page_count=20, avg_chars=10.0))
        assert plan.cost_cliff is True
        # Just-below threshold: no cliff
        plan = decide_from_probe(_probe(page_count=19, avg_chars=10.0))
        assert plan.cost_cliff is False

    def test_cost_cliff_not_set_for_native(self) -> None:
        """Native PDFs ride the panel cache and don't trigger the cliff."""
        plan = decide_from_probe(_probe(page_count=200, avg_chars=1500.0))
        assert plan.cost_cliff is False

    def test_estimated_tokens_match_design_table(self) -> None:
        """Estimates match D-phase §5: native 2250/pg, text 500/pg, image 2900/pg."""
        native = decide_from_probe(_probe(page_count=10, avg_chars=1500.0))
        assert native.estimated_input_tokens == 22_500
        text = decide_from_probe(_probe(page_count=700, file_size=10_000_000, avg_chars=500.0))
        assert text.estimated_input_tokens == 350_000
        image = decide_from_probe(_probe(page_count=10, avg_chars=10.0))
        assert image.estimated_input_tokens == 29_000

    def test_truncate_pages_caps_token_estimate(self) -> None:
        opts = PdfOptions(truncate_pages=50)
        plan = decide_from_probe(_probe(page_count=700, avg_chars=500.0), options=opts)
        assert plan.estimated_input_tokens == 50 * 2_250


# ---------------------------------------------------------------------------
# Probe / extract / render — real pypdfium2 wiring
# ---------------------------------------------------------------------------


class TestProbe:
    def test_probe_blank_page_reports_zero_text_density(self, blank_pdf_bytes: bytes) -> None:
        probe = probe_pdf(blank_pdf_bytes)
        assert probe.page_count == 1
        assert probe.file_size_bytes == len(blank_pdf_bytes)
        assert probe.avg_chars_per_page == 0.0
        assert probe.is_encrypted is False
        # Density is below mixed_density_low (0 < 32) so image_coverage is not computed.
        assert probe.image_coverage is None

    def test_probe_uses_at_most_probe_pages_samples(self, multipage_blank_pdf_bytes: bytes) -> None:
        probe = probe_pdf(multipage_blank_pdf_bytes, options=PdfOptions(probe_pages=2))
        assert probe.page_count == 5
        # Whether 2 or 5 pages were sampled, the avg is still 0 here, but the call must succeed.
        assert probe.avg_chars_per_page == 0.0


class TestPlanPdf:
    def test_plan_blank_pdf_chooses_page_as_image(self, blank_pdf_bytes: bytes) -> None:
        plan = plan_pdf(blank_pdf_bytes)
        assert plan.strategy is PdfStrategy.PAGE_AS_IMAGE
        assert plan.submission is SubmissionMode.INLINE_BASE64
        # 1-page scanned PDF is below the cliff threshold.
        assert plan.cost_cliff is False
        assert plan.options is DEFAULT_OPTIONS

    def test_plan_carries_options_through(self, blank_pdf_bytes: bytes) -> None:
        opts = PdfOptions(cost_cliff_page_threshold=1)
        plan = plan_pdf(blank_pdf_bytes, options=opts)
        assert plan.options is opts
        assert plan.cost_cliff is True


class TestRender:
    def test_render_emits_one_png_per_page(self, multipage_blank_pdf_bytes: bytes) -> None:
        pngs = render_pages_as_png(multipage_blank_pdf_bytes, options=PdfOptions(render_dpi=72))
        assert len(pngs) == 5
        assert all(p.startswith(b"\x89PNG") for p in pngs)
        # Sanity: 72 DPI letter-size is 612x792 px → not enormous.
        assert all(len(p) < 200_000 for p in pngs)


class TestExtract:
    def test_extract_returns_one_chunk_per_small_pdf(self, blank_pdf_bytes: bytes) -> None:
        chunks = extract_text_chunks(blank_pdf_bytes)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_extract_chunks_at_token_limit(self, multipage_blank_pdf_bytes: bytes) -> None:
        # Even at a tiny chunk_token_limit, blank pages all fit in one chunk
        # because 0 chars never trips the boundary. The chunking is exercised
        # in the cross-rig test fixture suite (hq-3o1r).
        chunks = extract_text_chunks(multipage_blank_pdf_bytes, options=PdfOptions(chunk_token_limit=1))
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_malformed_data_raises_pdf_error_not_pdfium_error(self) -> None:
        with pytest.raises(PdfError) as exc_info:
            probe_pdf(b"%PDF-1.4 not really a PDF")
        # Must be a PdfError but NOT a PdfEncryptedError — bad data is not encryption.
        assert not isinstance(exc_info.value, PdfEncryptedError)

    def test_password_string_in_pdfium_message_classifies_as_encrypted(self) -> None:
        """Encrypted PDFs surface as PdfEncryptedError, not generic PdfError.

        We don't construct an encrypted PDF here (pypdfium2 only reads, not
        writes encrypted PDFs); the heuristic is that PDFium error code 4
        or a "password"/"security" substring in the error message maps to
        :class:`PdfEncryptedError`. The fixture suite (hq-3o1r) covers a
        real encrypted PDF end-to-end.
        """
        from althing.attachments.pdf import _looks_like_password_error

        assert _looks_like_password_error("PDFium: Incorrect password error")
        assert _looks_like_password_error("Unsupported security scheme error")
        assert not _looks_like_password_error("Data format error")

    def test_missing_pypdfium2_message_points_to_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When pypdfium2 is unavailable the error names the install hint."""
        import builtins

        from althing.attachments import pdf as pdf_mod

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pypdfium2":
                raise ImportError("simulated: pypdfium2 not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(PdfMissingDependencyError, match=r"althing\[pdf\]"):
            pdf_mod._import_pdfium()

    def test_missing_pillow_message_points_to_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When Pillow is unavailable the renderer surfaces the same install hint."""
        import builtins

        from althing.attachments import pdf as pdf_mod

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("simulated: PIL not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(PdfMissingDependencyError, match=r"althing\[pdf\]"):
            pdf_mod._import_pillow()


# ---------------------------------------------------------------------------
# Plan dataclass shape
# ---------------------------------------------------------------------------


class TestPlanDataclass:
    def test_plan_is_immutable(self) -> None:
        plan = decide_from_probe(_probe(page_count=10, avg_chars=1500.0))
        assert isinstance(plan, PdfPlan)
        with pytest.raises(Exception):  # frozen dataclass: any mutation raises
            plan.strategy = PdfStrategy.TEXT_EXTRACT  # type: ignore[misc]

    def test_warnings_are_a_tuple(self) -> None:
        """Warnings are tuple, not list — plans hash + are safe to share across panelists."""
        plan = decide_from_probe(_probe(page_count=700, avg_chars=500.0))
        assert isinstance(plan.warnings, tuple)
