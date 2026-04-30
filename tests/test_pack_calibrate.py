"""Tests for ``synthpanel pack calibrate`` (sp-sghl).

Two surfaces under test:

1. ``synth_panel.calibration`` — pure pack-YAML round-trip helpers
   (load, merge by ``(dataset, question)``, splice block back in).
2. ``synth_panel.cli.commands.handle_pack_calibrate`` — CLI wiring,
   dry-run, allowlist gating, in-place rewrite. The actual panel run
   is mocked via ``_run_calibration_panel`` so tests stay hermetic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from synth_panel import calibration as calib_mod
from synth_panel.calibration import (
    CalibrationEntry,
    merge_calibration,
    update_pack_calibration_text,
    write_pack_calibration,
)
from synth_panel.main import main

# ── calibration.py: data + merge logic ────────────────────────────────


def test_calibration_entry_to_yaml_dict_drops_none_alignment_error():
    entry = CalibrationEntry(
        dataset="gss",
        question="HAPPY",
        jsd=0.18,
        n=100,
        samples_per_question=15,
        models=["haiku:0.5", "gemini-flash-lite:0.5"],
        extractor="pick_one:auto-derived",
        panelist_cost_usd=0.6451,
        calibrated_at="2026-04-26T14:23:00Z",
        synthpanel_version="0.11.1",
    )
    d = entry.to_yaml_dict()
    assert "alignment_error" not in d
    assert d["dataset"] == "gss"
    assert d["jsd"] == 0.18
    assert d["models"] == ["haiku:0.5", "gemini-flash-lite:0.5"]
    assert d["methodology_url"].startswith("https://synthpanel.dev")


def test_calibration_entry_keeps_alignment_error_when_set():
    entry = CalibrationEntry(
        dataset="gss",
        question="HAPPY",
        jsd=1.0,
        n=50,
        samples_per_question=15,
        models=[],
        extractor="pick_one:auto-derived",
        panelist_cost_usd=0.0,
        calibrated_at="2026-04-26T14:23:00Z",
        synthpanel_version="0.11.1",
        alignment_error="['x'] vs ['y']",
    )
    d = entry.to_yaml_dict()
    assert d["alignment_error"] == "['x'] vs ['y']"


def test_merge_calibration_appends_when_no_existing_match():
    existing = [
        {"dataset": "gss", "question": "HAPPY", "jsd": 0.9},
    ]
    new = {"dataset": "ntia", "question": "USE", "jsd": 0.2}
    out = merge_calibration(existing, new)
    assert len(out) == 2
    assert out[0]["dataset"] == "gss"
    assert out[1]["dataset"] == "ntia"


def test_merge_calibration_replaces_matching_dataset_question_in_place():
    existing = [
        {"dataset": "gss", "question": "HAPPY", "jsd": 0.9},
        {"dataset": "ntia", "question": "USE", "jsd": 0.2},
    ]
    new = {"dataset": "gss", "question": "HAPPY", "jsd": 0.18}
    out = merge_calibration(existing, new)
    # Order preserved; matching entry replaced; no duplicate added.
    assert len(out) == 2
    assert out[0]["dataset"] == "gss"
    assert out[0]["jsd"] == 0.18
    assert out[1]["dataset"] == "ntia"


def test_merge_calibration_handles_none_existing():
    out = merge_calibration(None, {"dataset": "gss", "question": "HAPPY", "jsd": 0.5})
    assert out == [{"dataset": "gss", "question": "HAPPY", "jsd": 0.5}]


def test_merge_calibration_rejects_non_list_existing():
    with pytest.raises(ValueError, match="must be a list"):
        merge_calibration({"oops": "not-a-list"}, {"dataset": "gss", "question": "HAPPY"})


# ── calibration.py: text round-trip preserves persona content ─────────


PACK_YAML_NO_CALIBRATION = """\
name: General Consumers
description: >
  Broad demographic cross-section.

# Persona definitions (hand-curated).
personas:
  - name: Emily Nakamura
    age: 33
    occupation: Marketing Manager
    background: >
      Lives in a mid-size city.
    personality_traits:
      - convenience-driven
      - brand-aware
"""


PACK_YAML_WITH_CALIBRATION = """\
name: General Consumers
description: >
  Broad demographic cross-section.

personas:
  - name: Emily Nakamura
    age: 33
    occupation: Marketing Manager
    background: >
      Lives in a mid-size city.
    personality_traits:
      - convenience-driven

calibration:
  - dataset: gss
    question: HAPPY
    jsd: 0.42
    n: 20
"""


def test_write_pack_calibration_appends_when_absent():
    entries = [{"dataset": "gss", "question": "HAPPY", "jsd": 0.18, "n": 100}]
    out = write_pack_calibration(PACK_YAML_NO_CALIBRATION, entries)
    # Persona content preserved verbatim.
    assert "Emily Nakamura" in out
    assert "convenience-driven" in out
    assert "# Persona definitions (hand-curated)." in out
    # New calibration block appended.
    assert "calibration:" in out
    assert "dataset: gss" in out
    assert out.endswith("\n")
    # Round-trip parses cleanly.
    parsed = yaml.safe_load(out)
    assert parsed["name"] == "General Consumers"
    assert parsed["calibration"][0]["dataset"] == "gss"
    assert parsed["calibration"][0]["jsd"] == 0.18


def test_write_pack_calibration_replaces_existing_block():
    entries = [{"dataset": "gss", "question": "HAPPY", "jsd": 0.18, "n": 100}]
    out = write_pack_calibration(PACK_YAML_WITH_CALIBRATION, entries)
    # Original Emily persona preserved.
    assert "Emily Nakamura" in out
    parsed = yaml.safe_load(out)
    assert parsed["calibration"] == [{"dataset": "gss", "question": "HAPPY", "jsd": 0.18, "n": 100}]
    # The old jsd: 0.42 is gone.
    assert "0.42" not in out


def test_update_pack_calibration_text_idempotent_for_same_dataset_question():
    raw = PACK_YAML_NO_CALIBRATION
    parsed = yaml.safe_load(raw)
    new_entry = {"dataset": "gss", "question": "HAPPY", "jsd": 0.5}
    after_first = update_pack_calibration_text(raw, parsed, new_entry)
    parsed_after = yaml.safe_load(after_first)
    new_entry_2 = {"dataset": "gss", "question": "HAPPY", "jsd": 0.7}
    after_second = update_pack_calibration_text(after_first, parsed_after, new_entry_2)
    final = yaml.safe_load(after_second)
    assert len(final["calibration"]) == 1
    assert final["calibration"][0]["jsd"] == 0.7


def test_update_pack_calibration_text_appends_distinct_baseline():
    raw = PACK_YAML_WITH_CALIBRATION
    parsed = yaml.safe_load(raw)
    new_entry = {"dataset": "ntia", "question": "USE", "jsd": 0.3}
    out = update_pack_calibration_text(raw, parsed, new_entry)
    final = yaml.safe_load(out)
    # Both old (gss:HAPPY) and new (ntia:USE) entries present.
    datasets = [(e["dataset"], e["question"]) for e in final["calibration"]]
    assert ("gss", "HAPPY") in datasets
    assert ("ntia", "USE") in datasets


# ── CLI: handle_pack_calibrate ────────────────────────────────────────


def _write_pack(tmp_path: Path) -> Path:
    p = tmp_path / "pack.yaml"
    p.write_text(PACK_YAML_NO_CALIBRATION, encoding="utf-8")
    return p


def _mock_panel_run_result() -> dict:
    return {
        "jsd": 0.234567,
        "extractor": "pick_one:auto-derived",
        "models": ["haiku:1.0"],
        "panelist_cost_usd": 0.1234,
        "alignment_error": None,
    }


def test_cli_pack_calibrate_dry_run_does_not_write(tmp_path, capsys):
    pack = _write_pack(tmp_path)
    original = pack.read_text(encoding="utf-8")
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        return_value=_mock_panel_run_result(),
    ):
        rc = main(
            [
                "pack",
                "calibrate",
                str(pack),
                "--against",
                "gss:HAPPY",
                "--n",
                "20",
                "--samples-per-question",
                "5",
                "--dry-run",
            ]
        )
    assert rc == 0
    # File untouched.
    assert pack.read_text(encoding="utf-8") == original
    out = capsys.readouterr().out
    assert "calibration:" in out
    assert "dataset: gss" in out
    assert "jsd: 0.234567" in out


def test_cli_pack_calibrate_writes_in_place(tmp_path):
    pack = _write_pack(tmp_path)
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        return_value=_mock_panel_run_result(),
    ):
        rc = main(
            [
                "pack",
                "calibrate",
                str(pack),
                "--against",
                "gss:HAPPY",
                "--n",
                "20",
                "--samples-per-question",
                "5",
                "--yes",
            ]
        )
    assert rc == 0
    parsed = yaml.safe_load(pack.read_text(encoding="utf-8"))
    assert parsed["name"] == "General Consumers"
    assert "Emily Nakamura" in pack.read_text(encoding="utf-8")
    cal = parsed["calibration"]
    assert len(cal) == 1
    assert cal[0]["dataset"] == "gss"
    assert cal[0]["question"] == "HAPPY"
    assert cal[0]["n"] == 20
    assert cal[0]["samples_per_question"] == 5
    assert cal[0]["extractor"] == "pick_one:auto-derived"
    assert isinstance(cal[0]["jsd"], float)
    assert "calibrated_at" in cal[0]
    assert "synthpanel_version" in cal[0]


def test_cli_pack_calibrate_replaces_prior_entry(tmp_path):
    pack = tmp_path / "pack.yaml"
    pack.write_text(PACK_YAML_WITH_CALIBRATION, encoding="utf-8")
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        return_value=_mock_panel_run_result(),
    ):
        rc = main(
            [
                "pack",
                "calibrate",
                str(pack),
                "--against",
                "gss:HAPPY",
                "--n",
                "20",
                "--samples-per-question",
                "5",
                "--yes",
            ]
        )
    assert rc == 0
    parsed = yaml.safe_load(pack.read_text(encoding="utf-8"))
    cal = parsed["calibration"]
    # Replaced (not appended) — still exactly one entry for gss:HAPPY.
    matching = [e for e in cal if e["dataset"] == "gss" and e["question"] == "HAPPY"]
    assert len(matching) == 1
    assert matching[0]["jsd"] != 0.42  # the old value is gone


def test_cli_pack_calibrate_writes_to_output_path(tmp_path):
    pack = _write_pack(tmp_path)
    out = tmp_path / "out.yaml"
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        return_value=_mock_panel_run_result(),
    ):
        rc = main(
            [
                "pack",
                "calibrate",
                str(pack),
                "--against",
                "gss:HAPPY",
                "--n",
                "20",
                "--samples-per-question",
                "5",
                "--output",
                str(out),
                "--yes",
            ]
        )
    assert rc == 0
    # Source untouched.
    assert "calibration:" not in pack.read_text(encoding="utf-8")
    # New file has the calibration block.
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert parsed["calibration"][0]["dataset"] == "gss"


def test_cli_pack_calibrate_rejects_bad_against_format(tmp_path, capsys):
    pack = _write_pack(tmp_path)
    rc = main(
        [
            "pack",
            "calibrate",
            str(pack),
            "--against",
            "gss",  # missing :QUESTION
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "DATASET:QUESTION" in err


def test_cli_pack_calibrate_rejects_gated_dataset(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SYNTHBENCH_ALLOW_GATED", raising=False)
    pack = _write_pack(tmp_path)
    rc = main(
        [
            "pack",
            "calibrate",
            str(pack),
            "--against",
            "wvs:Q1",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "inline-publishable" in err


def test_cli_pack_calibrate_missing_pack_yaml(tmp_path, capsys):
    rc = main(
        [
            "pack",
            "calibrate",
            str(tmp_path / "nope.yaml"),
            "--against",
            "gss:HAPPY",
        ]
    )
    assert rc == 2
    assert "file not found" in capsys.readouterr().err


def test_cli_pack_calibrate_rejects_empty_yaml(tmp_path, capsys):
    pack = tmp_path / "empty.yaml"
    pack.write_text("", encoding="utf-8")
    rc = main(
        [
            "pack",
            "calibrate",
            str(pack),
            "--against",
            "gss:HAPPY",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "mapping" in err or "personas" in err


def test_cli_pack_calibrate_rejects_zero_personas(tmp_path, capsys):
    pack = tmp_path / "no_personas.yaml"
    pack.write_text("name: empty pack\npersonas: []\n", encoding="utf-8")
    rc = main(
        [
            "pack",
            "calibrate",
            str(pack),
            "--against",
            "gss:HAPPY",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "no personas" in err


def test_cli_pack_calibrate_rejects_malformed_yaml(tmp_path, capsys):
    pack = tmp_path / "bad.yaml"
    pack.write_text("personas: [\n  - name: alice\n  - missing closing", encoding="utf-8")
    rc = main(
        [
            "pack",
            "calibrate",
            str(pack),
            "--against",
            "gss:HAPPY",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "failed to parse" in err.lower() or "parse" in err.lower()


def test_cli_pack_calibrate_unexpected_error_clean_message(tmp_path, capsys):
    pack = _write_pack(tmp_path)
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        side_effect=KeyError("missing-field"),
    ):
        rc = main(
            [
                "pack",
                "calibrate",
                str(pack),
                "--against",
                "gss:HAPPY",
                "--n",
                "20",
                "--samples-per-question",
                "5",
                "--yes",
            ]
        )
    assert rc == 1
    err = capsys.readouterr().err
    assert "unexpected failure" in err
    assert "KeyError" in err


def test_cli_pack_calibrate_debug_reraises_unexpected(tmp_path):
    pack = _write_pack(tmp_path)
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        side_effect=KeyError("missing-field"),
    ):
        with pytest.raises(KeyError):
            main(
                [
                    "pack",
                    "calibrate",
                    str(pack),
                    "--against",
                    "gss:HAPPY",
                    "--n",
                    "20",
                    "--samples-per-question",
                    "5",
                    "--yes",
                    "--debug",
                ]
            )


def test_cli_pack_calibrate_propagates_panel_failure(tmp_path, capsys):
    pack = _write_pack(tmp_path)
    with patch(
        "synth_panel.cli.commands._run_calibration_panel",
        side_effect=RuntimeError("panel run for calibration failed (exit 2): boom"),
    ):
        rc = main(
            [
                "pack",
                "calibrate",
                str(pack),
                "--against",
                "gss:HAPPY",
                "--n",
                "20",
                "--samples-per-question",
                "5",
                "--yes",
            ]
        )
    assert rc == 1
    assert "boom" in capsys.readouterr().err


def test_now_iso_utc_format():
    s = calib_mod.now_iso_utc()
    # RFC3339 UTC, second precision, trailing Z.
    assert s.endswith("Z")
    assert len(s) == len("2026-04-26T14:23:00Z")
