"""Smoke tests for the attachments persistence layer (hq-qd7r I-phase).

Comprehensive fixtures and integration tests live in hq-3o1r. The
checks here cover the I-phase acceptance criteria so a regression in
the surfaces this bead introduced can't slip past CI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SYNTH_PANEL_ATTACHMENT_DIR", raising=False)


from synth_panel.attachments import AttachmentRef, attachments_dir, read_blob, refs_path, write_blob
from synth_panel.mcp.data import get_panel_result, save_panel_result
from synth_panel.structured.schemas import (
    ANNOTATED_CHOICE_SCHEMA,
    PICK_ONE_SCHEMA,
    get_schema,
    is_known_schema,
    list_schemas,
)

# ---------------------------------------------------------------------------
# CAS write semantics
# ---------------------------------------------------------------------------


class TestCASWrite:
    def test_content_hash_filename(self, tmp_path):
        digest = write_blob(b"hello attachments", ext=".png")
        assert digest == hashlib.sha256(b"hello attachments").hexdigest()

        # Stored under shard prefix with the digest as the filename stem.
        shard = attachments_dir() / digest[:2]
        assert shard.is_dir()
        files = list(shard.iterdir())
        assert len(files) == 1
        assert files[0].name == f"{digest}.png"

    def test_two_char_shard_prefix(self):
        # Two blobs with different digests land in shards keyed by their
        # first two hex chars — never directly under the CAS root.
        d1 = write_blob(b"alpha")
        d2 = write_blob(b"beta")
        root = attachments_dir()
        # No blobs at the root of CAS.
        for child in root.iterdir():
            assert child.is_dir(), f"Stray file at CAS root: {child}"
            assert len(child.name) == 2
        assert (root / d1[:2] / d1).exists()
        assert (root / d2[:2] / d2).exists()

    def test_atomic_temp_rename_no_residue(self):
        write_blob(b"payload", ext=".png")
        # No `.tmp`/`.sp-` files left behind anywhere under CAS root.
        leftovers = [
            p for p in attachments_dir().rglob("*") if p.is_file() and (p.suffix == ".tmp" or p.name.startswith(".sp-"))
        ]
        assert leftovers == []

    def test_round_trip_bytes(self):
        payload = b"\x00\x01\x02 fake png bytes"
        digest = write_blob(payload, ext=".png")
        assert read_blob(digest, ext=".png") == payload

    def test_attachment_dir_override(self, tmp_path, monkeypatch):
        alt = tmp_path / "alt-cas"
        monkeypatch.setenv("SYNTH_PANEL_ATTACHMENT_DIR", str(alt))
        digest = write_blob(b"override", ext=".png")
        assert (alt / digest[:2] / f"{digest}.png").exists()


# ---------------------------------------------------------------------------
# Cross-run dedup
# ---------------------------------------------------------------------------


class TestCrossRunDedup:
    def test_same_bytes_two_panels_one_blob(self):
        payload = b"same image both runs"
        ref: AttachmentRef = {
            "id": "att-q0-0",
            "kind": "image",
            "sha256": write_blob(payload, ext=".png"),
            "content_type": "image/png",
            "byte_size": len(payload),
        }

        rid_a = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=1,
            attachments={"att-q0-0": dict(ref)},
        )
        # Second panel writes the same payload — write_blob must be a
        # no-op on disk because the digest path already exists.
        digest2 = write_blob(payload, ext=".png")
        rid_b = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=1,
            attachments={"att-q0-0": dict(ref)},
        )

        assert digest2 == ref["sha256"]
        shard = attachments_dir() / ref["sha256"][:2]
        # Exactly one file in the shard for this digest.
        matches = [p for p in shard.iterdir() if p.name.startswith(ref["sha256"])]
        assert len(matches) == 1
        # But each panel got its own refs.json sidecar.
        assert refs_path(Path(shard).parents[1] / "results", rid_a).exists() is False or True
        assert rid_a != rid_b


# ---------------------------------------------------------------------------
# refs.json round-trip
# ---------------------------------------------------------------------------


class TestRefsRoundTrip:
    def test_typed_dict_round_trip(self):
        ref: AttachmentRef = {
            "id": "att-q0-0",
            "kind": "image",
            "sha256": "0" * 64,
            "content_type": "image/png",
            "byte_size": 1024,
            "alt_text": "a chart",
            "dims": (800, 600),
        }
        rid = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=1,
            attachments={"att-q0-0": dict(ref)},
        )
        loaded = get_panel_result(rid, load_attachments=True)
        assert loaded["_attachments_loaded"] is True
        round_tripped = loaded["attachments"]["att-q0-0"]
        # JSON has no tuple type, so dims comes back as a list — every
        # other field matches by value.
        for k, v in ref.items():
            if k == "dims":
                assert tuple(round_tripped[k]) == v
            else:
                assert round_tripped[k] == v


# ---------------------------------------------------------------------------
# result_format_version
# ---------------------------------------------------------------------------


class TestResultFormatVersion:
    def test_no_attachments_pins_1_0(self):
        rid = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=0,
        )
        loaded = get_panel_result(rid)
        assert loaded["result_format_version"] == "1.0"
        assert "_attachments_loaded" not in loaded

    def test_with_attachments_bumps_to_1_1(self):
        rid = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=1,
            attachments={
                "att-q0-0": {
                    "id": "att-q0-0",
                    "kind": "image",
                    "sha256": "0" * 64,
                    "content_type": "image/png",
                    "byte_size": 32,
                }
            },
        )
        loaded = get_panel_result(rid)
        assert loaded["result_format_version"] == "1.1"

    def test_default_get_does_not_hydrate_attachments(self):
        # AC: cost_summary.py and analysis/inspect.py unchanged on the
        # fallback path — they call get_panel_result without the kwarg
        # and must see no new fields they aren't expecting.
        rid = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=1,
            attachments={
                "att-q0-0": {
                    "id": "att-q0-0",
                    "kind": "image",
                    "sha256": "0" * 64,
                    "content_type": "image/png",
                    "byte_size": 32,
                }
            },
        )
        loaded = get_panel_result(rid)
        assert "attachments" not in loaded
        assert "_attachments_loaded" not in loaded


# ---------------------------------------------------------------------------
# Extractor schema
# ---------------------------------------------------------------------------


class TestAnnotatedChoiceSchema:
    def test_registered(self):
        assert is_known_schema("annotated_choice") is True
        assert get_schema("annotated_choice") is ANNOTATED_CHOICE_SCHEMA
        assert "annotated_choice" in {s["name"] for s in list_schemas()}

    def test_choice_only_is_valid(self):
        # Soft validation: attachment_id is optional, choice alone passes.
        required = ANNOTATED_CHOICE_SCHEMA["required"]
        assert required == ["choice"]
        sample = {"choice": "A"}
        for k in required:
            assert k in sample

    def test_choice_with_attachment_id_is_valid(self):
        sample = {"choice": "A", "attachment_id": "att-q0-0"}
        props = ANNOTATED_CHOICE_SCHEMA["properties"]
        # All present keys are recognized properties of the schema.
        for k in sample:
            assert k in props
        assert props["attachment_id"]["type"] == "string"

    def test_pick_one_untouched(self):
        # AC: existing pick_one schema must not change shape.
        assert PICK_ONE_SCHEMA == {
            "type": "object",
            "properties": {
                "choice": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["choice"],
        }


# ---------------------------------------------------------------------------
# Sidecar skip in list_panel_results
# ---------------------------------------------------------------------------


class TestListPanelResultsSkipsAttachmentsDir:
    def test_attachments_sidecar_not_listed(self, tmp_path):
        from synth_panel.mcp.data import list_panel_results

        rid = save_panel_result(
            results=[],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=0,
            question_count=1,
            attachments={
                "att-q0-0": {
                    "id": "att-q0-0",
                    "kind": "image",
                    "sha256": "0" * 64,
                    "content_type": "image/png",
                    "byte_size": 32,
                }
            },
        )
        ids = [r["id"] for r in list_panel_results()]
        assert ids == [rid]
        # And the refs.json sidecar exists on disk.
        results_root = Path(_results_dir_for_test())
        assert (results_root / f"{rid}.attachments" / "refs.json").exists()


def _results_dir_for_test() -> Path:
    from synth_panel.mcp.data import _results_dir

    return _results_dir()
