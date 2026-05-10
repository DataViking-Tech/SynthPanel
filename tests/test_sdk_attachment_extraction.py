"""Tests for ``synth_panel.sdk._extract_attachment_refs`` (hq-hjk8 G7).

Covers the wiring that pulls inline attachment payloads out of panelist
``responses[i]["attachments"]`` and into the CAS sidecar before
``save_panel_result`` writes ``result.json``. Without this layer, a
15-persona x 10-image dogfood panel produced a 79 MB ``result.json``
with all base64 inlined.

Coverage:

* Bytes land in CAS, refs map carries ``AttachmentRef`` records.
* Inline dicts are replaced with ref-id strings in the response stream.
* Bank entries get stable ``att-bank-<id>`` ids; per-question payloads
  use ``att-q<i>-<j>``.
* Same bytes across multiple panelists collapse into one CAS blob and
  one ref-id (cross-persona dedup).
* Same bytes across re-runs reuse the existing CAS blob (cross-run dedup).
* Round-trip via ``get_panel_result(load_attachments=True)`` returns the
  refs map and bytes are still resolvable from CAS.
* URL-source images and ``url`` / ``html`` types pass through unchanged
  (they have no inline blob to extract).
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from synth_panel.attachments.store import attachments_dir, read_blob, refs_path
from synth_panel.instrument import Instrument
from synth_panel.mcp.data import get_panel_result, save_panel_result
from synth_panel.sdk import _extract_attachment_refs
from tests.attachments.fixtures import tiny_jpeg, tiny_pdf_text, tiny_png


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(data))
    monkeypatch.delenv("SYNTH_PANEL_ATTACHMENT_DIR", raising=False)
    return data


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _image_att(payload: bytes, *, media_type: str = "image/png") -> dict:
    return {
        "type": "image",
        "media_type": media_type,
        "source": {"type": "base64", "data": _b64(payload)},
    }


def _document_att(payload: bytes) -> dict:
    return {
        "type": "document",
        "media_type": "application/pdf",
        "source": {"type": "base64", "data": _b64(payload)},
    }


class TestExtractRefs:
    def test_replaces_inline_dicts_with_ref_id_strings(self, isolated_data: Path):
        png = tiny_png()
        flat_results = [
            {
                "persona": "Alice",
                "responses": [
                    {"question": "Q?", "response": "...", "attachments": [_image_att(png)]},
                ],
            }
        ]
        refs = _extract_attachment_refs(None, flat_results)
        assert len(refs) == 1
        new_atts = flat_results[0]["responses"][0]["attachments"]
        assert isinstance(new_atts, list)
        assert all(isinstance(a, str) for a in new_atts)
        rid = new_atts[0]
        assert rid in refs
        # Bytes landed in CAS keyed by sha256.
        sha = hashlib.sha256(png).hexdigest()
        assert refs[rid]["sha256"] == sha
        assert refs[rid]["kind"] == "image"
        assert refs[rid]["content_type"] == "image/png"
        assert refs[rid]["byte_size"] == len(png)
        assert read_blob(sha, ext=".png") == png

    def test_bank_entries_get_stable_att_bank_id(self, isolated_data: Path):
        png = tiny_png()
        bank = {"hero_creative": _image_att(png)}
        instrument = Instrument(version=1, attachments=bank)

        # Empty responses — bank pre-pass alone should register the ref.
        refs = _extract_attachment_refs(instrument, [])
        assert "att-bank-hero_creative" in refs
        assert refs["att-bank-hero_creative"]["sha256"] == hashlib.sha256(png).hexdigest()

    def test_response_dedupe_collapses_to_bank_id(self, isolated_data: Path):
        # When a question's resolved attachment has the same bytes as a bank
        # entry, the response's inline dict should rewrite to the existing
        # ``att-bank-<id>`` id rather than allocating a new ``att-q<i>-<j>``.
        png = tiny_png()
        bank = {"hero": _image_att(png)}
        instrument = Instrument(version=1, attachments=bank)
        flat_results = [
            {
                "persona": "Alice",
                "responses": [
                    {"question": "Q?", "response": "...", "attachments": [_image_att(png)]},
                ],
            }
        ]
        refs = _extract_attachment_refs(instrument, flat_results)
        assert flat_results[0]["responses"][0]["attachments"] == ["att-bank-hero"]
        assert list(refs.keys()) == ["att-bank-hero"]

    def test_per_question_inline_uses_att_q_id(self, isolated_data: Path):
        png = tiny_png()
        jpg = tiny_jpeg()
        flat_results = [
            {
                "persona": "Alice",
                "responses": [
                    {"question": "Q1", "response": "...", "attachments": [_image_att(png)]},
                    {"question": "Q2", "response": "...", "attachments": [_image_att(jpg, media_type="image/jpeg")]},
                ],
            }
        ]
        refs = _extract_attachment_refs(None, flat_results)
        rids = {
            "q0": flat_results[0]["responses"][0]["attachments"][0],
            "q1": flat_results[0]["responses"][1]["attachments"][0],
        }
        assert rids["q0"] == "att-q0-0"
        assert rids["q1"] == "att-q1-0"
        assert refs["att-q0-0"]["content_type"] == "image/png"
        assert refs["att-q1-0"]["content_type"] == "image/jpeg"

    def test_cross_persona_dedup(self, isolated_data: Path):
        # Same image stamped on two personas' responses → one CAS blob,
        # one ref-id, both responses point at the same ref.
        png = tiny_png()
        flat_results = [
            {
                "persona": "Alice",
                "responses": [{"question": "Q", "response": "a", "attachments": [_image_att(png)]}],
            },
            {
                "persona": "Bob",
                "responses": [{"question": "Q", "response": "b", "attachments": [_image_att(png)]}],
            },
        ]
        refs = _extract_attachment_refs(None, flat_results)
        assert len(refs) == 1
        rid = next(iter(refs.keys()))
        assert flat_results[0]["responses"][0]["attachments"] == [rid]
        assert flat_results[1]["responses"][0]["attachments"] == [rid]
        # Exactly one blob in the shard for this digest.
        sha = hashlib.sha256(png).hexdigest()
        shard = attachments_dir() / sha[:2]
        matches = [p for p in shard.iterdir() if p.name.startswith(sha)]
        assert len(matches) == 1

    def test_cross_run_dedup(self, isolated_data: Path):
        # Run 1: register the same payload. Run 2: same payload should
        # return the same digest without rewriting the CAS entry.
        png = tiny_png()
        flat_a = [
            {"persona": "Alice", "responses": [{"question": "Q", "response": "x", "attachments": [_image_att(png)]}]}
        ]
        _extract_attachment_refs(None, flat_a)
        sha = hashlib.sha256(png).hexdigest()
        path = attachments_dir() / sha[:2] / f"{sha}.png"
        mtime_before = path.stat().st_mtime_ns

        flat_b = [
            {"persona": "Bob", "responses": [{"question": "Q", "response": "y", "attachments": [_image_att(png)]}]}
        ]
        _extract_attachment_refs(None, flat_b)
        # Idempotent: blob is not rewritten.
        assert path.stat().st_mtime_ns == mtime_before

    def test_url_source_image_left_inline(self, isolated_data: Path):
        # No bytes to persist — pass through unchanged.
        att = {
            "type": "image",
            "media_type": "image/png",
            "source": {"type": "url", "url": "https://example.com/x.png"},
        }
        flat_results = [{"persona": "Alice", "responses": [{"question": "Q", "attachments": [att]}]}]
        refs = _extract_attachment_refs(None, flat_results)
        assert refs == {}
        # Still inline (dict, not str).
        assert flat_results[0]["responses"][0]["attachments"][0] is att

    def test_url_and_html_types_left_inline(self, isolated_data: Path):
        url_att = {"type": "url", "url": "https://example.com"}
        html_att = {"type": "html", "text": "<p>hello</p>"}
        flat_results = [
            {
                "persona": "Alice",
                "responses": [{"question": "Q", "attachments": [url_att, html_att]}],
            }
        ]
        refs = _extract_attachment_refs(None, flat_results)
        assert refs == {}
        assert flat_results[0]["responses"][0]["attachments"] == [url_att, html_att]

    def test_document_pdf_extracted(self, isolated_data: Path):
        pdf = tiny_pdf_text()
        flat_results = [
            {
                "persona": "Alice",
                "responses": [{"question": "Q", "attachments": [_document_att(pdf)]}],
            }
        ]
        refs = _extract_attachment_refs(None, flat_results)
        assert len(refs) == 1
        rid = next(iter(refs.keys()))
        assert refs[rid]["kind"] == "pdf"
        assert refs[rid]["content_type"] == "application/pdf"
        sha = hashlib.sha256(pdf).hexdigest()
        assert read_blob(sha, ext=".pdf") == pdf
        assert flat_results[0]["responses"][0]["attachments"] == [rid]

    def test_no_attachments_no_refs(self, isolated_data: Path):
        flat_results = [{"persona": "Alice", "responses": [{"question": "Q", "response": "..."}]}]
        refs = _extract_attachment_refs(None, flat_results)
        assert refs == {}
        # Response dict left untouched.
        assert "attachments" not in flat_results[0]["responses"][0]

    def test_string_refs_already_present_pass_through(self, isolated_data: Path):
        # A pre-extracted ref-id should be preserved verbatim — the helper
        # is idempotent on already-rewritten responses.
        flat_results = [
            {"persona": "Alice", "responses": [{"question": "Q", "attachments": ["att-bank-hero"]}]}
        ]
        refs = _extract_attachment_refs(None, flat_results)
        assert refs == {}
        assert flat_results[0]["responses"][0]["attachments"] == ["att-bank-hero"]


class TestSaveResultWiring:
    """End-to-end: extract refs, save, then round-trip via get_panel_result."""

    def test_result_json_carries_only_ref_strings(self, isolated_data: Path):
        png = tiny_png()
        flat_results = [
            {
                "persona": "Alice",
                "responses": [{"question": "Q?", "response": "ok", "attachments": [_image_att(png)]}],
            }
        ]
        refs = _extract_attachment_refs(None, flat_results)
        rid = save_panel_result(
            results=flat_results,
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=1,
            question_count=1,
            attachments=refs or None,
        )

        # result.json: bytes are out, ref-id strings are in, version bumped.
        result_path = isolated_data / "results" / f"{rid}.json"
        on_disk = json.loads(result_path.read_text(encoding="utf-8"))
        assert on_disk["result_format_version"] == "1.1"
        atts = on_disk["results"][0]["responses"][0]["attachments"]
        assert atts == [next(iter(refs.keys()))]
        # No base64 anywhere in the persisted result.
        assert "base64" not in result_path.read_text(encoding="utf-8")

        # Sidecar exists at the documented path.
        sp = refs_path(isolated_data / "results", rid)
        assert sp.exists()

        # Opt-in readback returns the refs map and bytes resolve from CAS.
        loaded = get_panel_result(rid, load_attachments=True)
        assert loaded["_attachments_loaded"] is True
        assert loaded["attachments"] == refs
        for ref in loaded["attachments"].values():
            blob = read_blob(ref["sha256"], ext=".png")
            assert hashlib.sha256(blob).hexdigest() == ref["sha256"]

    def test_result_json_size_bounded_when_many_images(self, isolated_data: Path):
        # Many copies of a 4KB image stamped on three personas. Without
        # extraction, ten copies x three personas = 30 inline base64 blobs
        # in result.json. After extraction the response only carries the
        # ten ref-id strings (the same ten across all three personas, since
        # the payload is identical), and bytes live in CAS.
        # Use distinct payloads so dedup doesn't collapse all of them; the
        # point of the test is the *response* stream gets compact ref ids
        # instead of N copies of base64 per persona.
        png = tiny_png(width=64, height=64) * 1  # ~168 B; small but distinct keys come from index below
        # Build ten distinct large-ish payloads by appending a unique suffix
        # so each one's sha256 differs.
        payloads = [png + f"unique-{i}".encode() * 50 for i in range(10)]
        atts = [_image_att(p) for p in payloads]
        flat_results = [
            {
                "persona": f"P{i}",
                "responses": [{"question": "Q", "response": "ok", "attachments": list(atts)}],
            }
            for i in range(3)
        ]
        pre_size = len(json.dumps(flat_results))

        refs = _extract_attachment_refs(None, flat_results)
        rid = save_panel_result(
            results=flat_results,
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0",
            persona_count=3,
            question_count=1,
            attachments=refs or None,
        )
        result_path = isolated_data / "results" / f"{rid}.json"
        on_disk_size = result_path.stat().st_size

        # Post-extraction file is dramatically smaller than the inlined form:
        # the 30 base64 copies (~150 KB) collapse to ten ref-id strings.
        assert on_disk_size * 10 < pre_size, (
            f"expected on-disk result << inlined form; got {on_disk_size}B vs {pre_size}B inlined"
        )
        # Ten distinct payloads → ten unique CAS blobs and ten ref ids.
        # Each persona's response now lists ten ref-id strings instead of
        # ten inline base64 dicts.
        assert len(refs) == 10
        for rd in flat_results:
            atts_out = rd["responses"][0]["attachments"]
            assert len(atts_out) == 10
            assert all(isinstance(a, str) for a in atts_out)
