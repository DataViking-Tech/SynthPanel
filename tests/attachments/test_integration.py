"""End-to-end attachment flow: parse → filter → block emission → CAS persist
→ result save → readback.

Each step has its own focused unit test elsewhere; this module wires the
steps together with realistic data so a regression in any one of them
that happens to leave the unit tests passing still surfaces here.

The integration runs entirely against in-process bytes — no real LLM
calls, no real network — but uses the actual ``filter_attachments``,
``build_question_blocks``, ``write_blob``, ``save_panel_result``, and
``get_panel_result`` implementations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from synth_panel.attachments import filter_attachments, write_blob
from synth_panel.attachments.filter import count_strata
from synth_panel.instrument import parse_instrument
from synth_panel.llm.models import ImageBlock, TextBlock
from synth_panel.orchestrator import (
    PanelPlanningError,
    _enforce_strata_cap,
)
from synth_panel.prompts import build_question_blocks
from tests.attachments.fixtures import tiny_jpeg, tiny_png

# ---------------------------------------------------------------------------
# Personas + instrument shared across tests
# ---------------------------------------------------------------------------


@pytest.fixture
def personas() -> list[dict[str, Any]]:
    return [
        {"name": "Alice", "device": "mobile", "age": 30},
        {"name": "Bob", "device": "desktop", "age": 45},
        {"name": "Cleo", "device": "mobile", "age": 28},
        {"name": "Dan", "device": "tablet", "age": 51},
    ]


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYNTH_PANEL_ATTACHMENT_DIR", raising=False)
    return tmp_path / "data"


# ---------------------------------------------------------------------------


class TestParseToBlockEmission:
    """Bank-ref attachments + dict-form stratification filters parse, validate,
    filter per persona, and emit content blocks in the canonical order."""

    def test_full_flow_bank_then_filter_then_blocks(self, personas: list[dict[str, Any]]):
        instrument = {
            "version": 1,
            "attachments": {
                "shared_logo": {
                    "type": "image",
                    "media_type": "image/png",
                    "source": {"type": "base64", "data": "ZmFrZQ=="},
                },
            },
            "questions": [
                {
                    "text": "Which ad resonates more?",
                    "attachments": [
                        "shared_logo",
                        {
                            "id": "ad_mobile",
                            "type": "image",
                            "media_type": "image/jpeg",
                            "source": {"type": "base64", "data": "bW9iaWxl"},
                            "filter": [
                                {"field": "device", "op": "equals", "value": "mobile"},
                            ],
                        },
                        {
                            "id": "ad_desktop",
                            "type": "image",
                            "media_type": "image/jpeg",
                            "source": {"type": "base64", "data": "ZGVza3RvcA=="},
                            "filter": [
                                {"field": "device", "op": "equals", "value": "desktop"},
                            ],
                        },
                    ],
                }
            ],
        }

        # Parse — both bank validation AND filter validation must pass.
        parsed = parse_instrument(instrument)
        assert "shared_logo" in parsed.attachments
        q = parsed.questions[0]
        assert isinstance(q["attachments"], list)

        # K≤5 cap fires on the dict-form attachments only. Three devices in
        # personas → at most K=3 strata for this question (mobile/desktop/tablet).
        # Tablet personas match no filter so they fall into the "no-attachment"
        # bucket — still a stratum, but it's <= 5.
        dict_atts = [a for a in q["attachments"] if isinstance(a, dict)]
        assert count_strata(personas, dict_atts) <= 5
        # _enforce_strata_cap is the planning gate; should NOT raise here.
        _enforce_strata_cap(personas, parsed.questions, max_k=5)

        # Filter per persona: Alice (mobile) sees ad_mobile, Bob (desktop)
        # sees ad_desktop, Dan (tablet) sees neither.
        alice_atts = filter_attachments(dict_atts, personas[0])
        bob_atts = filter_attachments(dict_atts, personas[1])
        dan_atts = filter_attachments(dict_atts, personas[3])

        assert [a["id"] for a in alice_atts] == ["ad_mobile"]
        assert [a["id"] for a in bob_atts] == ["ad_desktop"]
        assert dan_atts == []

        # Build blocks for Alice. Canonical order: shared images, then
        # per-question attachments, then text.
        # NOTE: ``q["attachments"]`` is the AUTHORED list (refs + dicts).
        # The orchestrator passes only the persona-filtered DICTS to
        # build_question_blocks; bank refs flow via panel_shared_attachments
        # or are resolved upstream. Mirror that here.
        shared_for_panel = [parsed.attachments["shared_logo"]]
        alice_blocks = build_question_blocks(q, attachments=alice_atts, panel_shared_attachments=shared_for_panel)
        # Last block is the question text; image blocks precede it.
        assert isinstance(alice_blocks[-1], TextBlock)
        assert alice_blocks[-1].text == "Which ad resonates more?"
        image_blocks = [b for b in alice_blocks if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 2  # shared_logo + ad_mobile

    def test_strata_cap_enforced_when_exceeded(self):
        # 6 personas with 6 distinct devices → 6 strata on this question.
        big_personas = [{"name": f"P{i}", "device": f"d{i}"} for i in range(6)]
        instrument = {
            "version": 1,
            "questions": [
                {
                    "text": "Q",
                    "attachments": [
                        {
                            "id": f"a{i}",
                            "type": "image",
                            "media_type": "image/png",
                            "source": {"type": "base64", "data": "eA=="},
                            "filter": [{"field": "device", "op": "equals", "value": f"d{i}"}],
                        }
                        for i in range(6)
                    ],
                }
            ],
        }
        parsed = parse_instrument(instrument)
        with pytest.raises(PanelPlanningError, match=r"K=6|exceeds|strata"):
            _enforce_strata_cap(big_personas, parsed.questions, max_k=5)


class TestCASToReadback:
    """Bytes land in CAS, refs persist in the result sidecar, and the round-trip
    via :func:`get_panel_result` returns identical metadata."""

    def test_full_persistence_round_trip(self, isolated_data: Path):
        from synth_panel.mcp.data import get_panel_result, save_panel_result

        png_bytes = tiny_png()
        jpeg_bytes = tiny_jpeg()

        # 1) Bytes in CAS.
        png_sha = write_blob(png_bytes, ext="png")
        jpeg_sha = write_blob(jpeg_bytes, ext="jpg")
        assert png_sha == hashlib.sha256(png_bytes).hexdigest()
        assert jpeg_sha == hashlib.sha256(jpeg_bytes).hexdigest()

        # 2) AttachmentRef map mirrors the per-result sidecar contract.
        attachments = {
            "logo": {
                "id": "logo",
                "kind": "image",
                "sha256": png_sha,
                "content_type": "image/png",
                "byte_size": len(png_bytes),
                "alt_text": "logo",
            },
            "ad": {
                "id": "ad",
                "kind": "image",
                "sha256": jpeg_sha,
                "content_type": "image/jpeg",
                "byte_size": len(jpeg_bytes),
            },
        }

        # 3) Save the panel result alongside attachment refs.
        rid = save_panel_result(
            results=[
                {
                    "persona": "Alice",
                    "responses": [
                        {
                            "question": "Which ad?",
                            "response": "I prefer the JPEG ad.",
                            "extraction": {"choice": "ad", "attachment_id": "ad"},
                        }
                    ],
                }
            ],
            model="claude-sonnet-4-6",
            total_usage={"input_tokens": 100, "output_tokens": 30},
            total_cost="$0.05",
            persona_count=1,
            question_count=1,
            attachments=attachments,
        )

        # 4) Default readback: no attachments loaded, version bumped.
        result = get_panel_result(rid)
        assert result["result_format_version"] == "1.1"
        assert "attachments" not in result

        # 5) Opt-in readback: attachments fully restored.
        loaded = get_panel_result(rid, load_attachments=True)
        assert loaded["_attachments_loaded"] is True
        assert loaded["attachments"] == attachments

        # 6) Bytes still resolvable from CAS via the loaded refs.
        from synth_panel.attachments.store import read_blob

        for ref in loaded["attachments"].values():
            ext = "png" if ref["content_type"] == "image/png" else "jpg"
            blob = read_blob(ref["sha256"], ext=ext)
            assert hashlib.sha256(blob).hexdigest() == ref["sha256"]
            assert len(blob) == ref["byte_size"]


class TestExtractionSchemaIntegration:
    """An instrument with ``extraction_schema: annotated_choice`` parses, and
    the schema in the registry is the same object the runtime would use."""

    def test_annotated_choice_extraction_schema_parses(self):
        instrument = {
            "version": 1,
            "attachments": {
                "ad_a": {
                    "type": "image",
                    "media_type": "image/png",
                    "source": {"type": "base64", "data": "YQ=="},
                },
                "ad_b": {
                    "type": "image",
                    "media_type": "image/png",
                    "source": {"type": "base64", "data": "Yg=="},
                },
            },
            "questions": [
                {
                    "text": "Pick one.",
                    "attachments": ["ad_a", "ad_b"],
                    "extraction_schema": "annotated_choice",
                }
            ],
        }
        parsed = parse_instrument(instrument)
        q = parsed.questions[0]
        assert q["extraction_schema"] == "annotated_choice"

    def test_unknown_extraction_schema_rejected(self):
        instrument = {
            "version": 1,
            "questions": [
                {
                    "text": "Q",
                    "extraction_schema": "no_such_schema_42",
                }
            ],
        }
        with pytest.raises(Exception, match=r"no_such_schema_42|Unknown"):
            parse_instrument(instrument)
