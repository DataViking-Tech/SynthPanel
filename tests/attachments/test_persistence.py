"""Persistence layer for attachments: CAS round-trip, refs.json sidecar,
``result_format_version`` bump, and ``ANNOTATED_CHOICE_SCHEMA`` shape.

These tests pin the on-disk contract so a future refactor can't silently
break readback or shift the format version. Every test isolates state via
``SYNTH_PANEL_ATTACHMENT_DIR`` / ``SYNTH_PANEL_DATA_DIR`` overrides — no
real ``~/.althing`` is touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from althing.attachments.store import (
    attachments_dir,
    read_blob,
    refs_path,
    write_blob,
)
from althing.structured.schemas import ANNOTATED_CHOICE_SCHEMA, get_schema, is_known_schema
from tests.attachments.fixtures import tiny_jpeg, tiny_pdf_text, tiny_png

# ---------------------------------------------------------------------------
# Shared isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CAS at a tmp dir; return the resolved root."""
    root = tmp_path / "cas"
    monkeypatch.setenv("SYNTH_PANEL_ATTACHMENT_DIR", str(root))
    monkeypatch.delenv("SYNTH_PANEL_DATA_DIR", raising=False)
    return attachments_dir()


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both data and attachment dirs at a tmp dir; return the data root."""
    data = tmp_path / "data"
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(data))
    # save_panel_result resolves results_dir from SYNTH_PANEL_DATA_DIR.
    monkeypatch.delenv("SYNTH_PANEL_ATTACHMENT_DIR", raising=False)
    return data


# ---------------------------------------------------------------------------
# CAS round-trip
# ---------------------------------------------------------------------------


class TestCASRoundTrip:
    def test_write_then_read_returns_identical_bytes(self, isolated_cas: Path):
        payload = tiny_png()
        digest = write_blob(payload, ext="png")
        assert digest == hashlib.sha256(payload).hexdigest()
        assert read_blob(digest, ext="png") == payload

    def test_blob_lands_under_two_char_shard(self, isolated_cas: Path):
        digest = write_blob(tiny_jpeg(), ext="jpg")
        shard = isolated_cas / digest[:2]
        assert shard.is_dir(), f"expected shard dir {shard}"
        blobs = list(shard.iterdir())
        assert len(blobs) == 1
        assert blobs[0].name == f"{digest}.jpg"

    def test_dedup_returns_existing_digest_without_rewrite(self, isolated_cas: Path):
        payload = tiny_pdf_text()
        d1 = write_blob(payload, ext="pdf")
        path = isolated_cas / d1[:2] / f"{d1}.pdf"
        mtime_before = path.stat().st_mtime_ns
        d2 = write_blob(payload, ext="pdf")
        assert d2 == d1
        assert path.stat().st_mtime_ns == mtime_before, "blob was rewritten on duplicate write"

    def test_extension_is_normalized(self, isolated_cas: Path):
        payload = b"hello"
        d_with_dot = write_blob(payload, ext=".bin")
        d_no_dot = write_blob(payload, ext="bin")
        assert d_with_dot == d_no_dot
        # Result file ends with .bin, exactly once.
        assert (isolated_cas / d_with_dot[:2] / f"{d_with_dot}.bin").exists()

    def test_extension_with_path_separators_is_rejected(self, isolated_cas: Path):
        with pytest.raises(ValueError, match="Invalid attachment extension"):
            write_blob(b"x", ext="../etc")

    def test_read_missing_blob_raises(self, isolated_cas: Path):
        # Without ext: store glob-falls-back, then raises "Attachment not found".
        with pytest.raises(FileNotFoundError, match="Attachment not found"):
            read_blob("0" * 64)
        # With ext: passes through to path.read_bytes(); raw FNF is fine.
        with pytest.raises(FileNotFoundError):
            read_blob("0" * 64, ext="png")

    def test_read_without_ext_falls_back_to_glob(self, isolated_cas: Path):
        digest = write_blob(b"payload-bytes", ext="bin")
        # Caller didn't track the extension — read_blob recovers via glob.
        assert read_blob(digest) == b"payload-bytes"

    def test_invalid_short_digest_rejected(self, isolated_cas: Path):
        # _shard_path requires sha256[:2]; a 1-char digest cannot be sharded.
        from althing.attachments.store import _shard_path

        with pytest.raises(ValueError, match="too short"):
            _shard_path(isolated_cas, "a", "")

    def test_non_bytes_input_rejected(self, isolated_cas: Path):
        with pytest.raises(TypeError, match="bytes-like"):
            write_blob("not-bytes", ext="bin")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# refs.json round-trip + result_format_version bump
# ---------------------------------------------------------------------------


class TestRefsJsonAndFormatVersion:
    def test_save_without_attachments_keeps_version_1_0(self, isolated_data: Path):
        from althing.mcp.data import get_panel_result, save_panel_result

        rid = save_panel_result(
            results=[{"persona": "P", "responses": [{"text": "ok"}]}],
            model="claude-sonnet-4-6",
            total_usage={"input_tokens": 5, "output_tokens": 3},
            total_cost="$0.01",
            persona_count=1,
            question_count=1,
        )
        result = get_panel_result(rid)
        assert result["result_format_version"] == "1.0"
        assert result.get("attachments") is None
        # Sidecar dir is NOT created when there are no attachments.
        sidecar = isolated_data / "results" / f"{rid}.attachments"
        assert not sidecar.exists()

    def test_save_with_attachments_bumps_to_1_1_and_writes_refs(self, isolated_data: Path):
        from althing.mcp.data import get_panel_result, save_panel_result

        digest = hashlib.sha256(b"img").hexdigest()
        atts = {
            "ad1": {
                "id": "ad1",
                "kind": "image",
                "sha256": digest,
                "content_type": "image/png",
                "byte_size": 3,
                "alt_text": "tiny ad",
            }
        }
        rid = save_panel_result(
            results=[{"persona": "P", "responses": [{"text": "ok"}]}],
            model="claude-sonnet-4-6",
            total_usage={"input_tokens": 5, "output_tokens": 3},
            total_cost="$0.01",
            persona_count=1,
            question_count=1,
            attachments=atts,
        )
        # Format version bumps when attachments are present.
        result = get_panel_result(rid)
        assert result["result_format_version"] == "1.1"

        # refs.json sidecar is on disk and parseable.
        sidecar = isolated_data / "results" / f"{rid}.attachments" / "refs.json"
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded == atts

        # Default get_panel_result does NOT load attachments (existing
        # consumers stay unchanged).
        assert "attachments" not in result

        # Opt-in load surfaces the same map and a flag.
        result_loaded = get_panel_result(rid, load_attachments=True)
        assert result_loaded["_attachments_loaded"] is True
        assert result_loaded["attachments"] == atts

    def test_get_with_load_attachments_handles_missing_sidecar(self, isolated_data: Path):
        from althing.mcp.data import get_panel_result, save_panel_result

        rid = save_panel_result(
            results=[{"persona": "P", "responses": []}],
            model="m",
            total_usage={},
            total_cost="$0.00",
            persona_count=1,
            question_count=0,
        )
        result = get_panel_result(rid, load_attachments=True)
        # No sidecar -> _attachments_loaded is False, no `attachments` key.
        assert result["_attachments_loaded"] is False
        assert "attachments" not in result

    def test_refs_path_layout_matches_spec(self, tmp_path: Path):
        rp = refs_path(tmp_path, "result-20260509-001")
        assert rp == tmp_path / "result-20260509-001.attachments" / "refs.json"

    def test_list_panel_results_skips_attachments_sidecar(self, isolated_data: Path):
        from althing.mcp.data import list_panel_results, save_panel_result

        rid = save_panel_result(
            results=[{"persona": "P", "responses": []}],
            model="m",
            total_usage={},
            total_cost="$0.00",
            persona_count=1,
            question_count=0,
            attachments={
                "ad": {"id": "ad", "kind": "image", "sha256": "0" * 64, "content_type": "image/png", "byte_size": 1}
            },
        )
        ids = [e["id"] for e in list_panel_results()]
        assert rid in ids
        # No phantom result derived from the sidecar dir.
        assert not any(i.endswith(".attachments") for i in ids)


# ---------------------------------------------------------------------------
# Cross-run dedup (CAS dedups bytes; sidecars do NOT)
# ---------------------------------------------------------------------------


class TestCrossRunDedup:
    def test_same_bytes_two_runs_one_blob(self, isolated_cas: Path):
        png = tiny_png()
        d1 = write_blob(png, ext="png")
        # second "run" — different process logically; same bytes.
        d2 = write_blob(png, ext="png")
        assert d1 == d2
        shard_dir = isolated_cas / d1[:2]
        # Exactly one blob with that digest, regardless of write count.
        matching = [p for p in shard_dir.iterdir() if p.name.startswith(d1)]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# ANNOTATED_CHOICE_SCHEMA
# ---------------------------------------------------------------------------


class TestAnnotatedChoiceSchema:
    def test_schema_registered(self):
        assert is_known_schema("annotated_choice")
        assert get_schema("annotated_choice") is ANNOTATED_CHOICE_SCHEMA

    def test_required_fields_match_pick_one_plus_optional_attachment_id(self):
        # Soft validation: `attachment_id` is OPTIONAL so a model that
        # doesn't cite an attachment still validates the same as pick_one.
        assert ANNOTATED_CHOICE_SCHEMA["required"] == ["choice"]
        assert "attachment_id" in ANNOTATED_CHOICE_SCHEMA["properties"]
        # And the attachment_id, when present, must be a string.
        assert ANNOTATED_CHOICE_SCHEMA["properties"]["attachment_id"]["type"] == "string"

    def test_validates_with_jsonschema_when_available(self):
        jsonschema = pytest.importorskip("jsonschema")

        validator = jsonschema.Draft7Validator(ANNOTATED_CHOICE_SCHEMA)
        # Ok: no attachment_id (matches pick_one shape).
        validator.validate({"choice": "A"})
        # Ok: with attachment_id.
        validator.validate({"choice": "A", "attachment_id": "ad1", "reasoning": "saw ad"})
        # Ok: extra fields (no additionalProperties: false in schema).
        validator.validate({"choice": "A", "extra": 1})

        # Bad: missing required `choice`.
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"attachment_id": "ad1"})
        # Bad: wrong type on attachment_id.
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"choice": "A", "attachment_id": 99})
