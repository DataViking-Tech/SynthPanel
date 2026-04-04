"""Tests for synth_panel.mcp.data persistence layer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point SYNTH_PANEL_DATA_DIR at a temp directory for every test."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))


# --- import after env is set so _data_dir() picks it up ---
from synth_panel.mcp.data import (
    get_panel_result,
    get_persona_pack,
    list_panel_results,
    list_persona_packs,
    save_panel_result,
    save_persona_pack,
)


# ---------------------------------------------------------------------------
# Persona packs
# ---------------------------------------------------------------------------

class TestPersonaPacks:
    def test_list_empty(self):
        assert list_persona_packs() == []

    def test_save_and_list(self):
        personas = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = save_persona_pack("Test Pack", personas)
        assert result["name"] == "Test Pack"
        assert result["persona_count"] == 2
        assert "id" in result

        packs = list_persona_packs()
        assert len(packs) == 1
        assert packs[0]["name"] == "Test Pack"
        assert packs[0]["persona_count"] == 2

    def test_save_with_custom_id(self):
        result = save_persona_pack("Custom", [{"name": "Eve"}], pack_id="my-pack")
        assert result["id"] == "my-pack"

    def test_get_pack(self):
        save_persona_pack("Get Test", [{"name": "Charlie"}], pack_id="get-test")
        pack = get_persona_pack("get-test")
        assert pack["name"] == "Get Test"
        assert pack["id"] == "get-test"
        assert len(pack["personas"]) == 1

    def test_get_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            get_persona_pack("does-not-exist")

    def test_overwrite_pack(self):
        save_persona_pack("V1", [{"name": "A"}], pack_id="ow")
        save_persona_pack("V2", [{"name": "A"}, {"name": "B"}], pack_id="ow")
        pack = get_persona_pack("ow")
        assert pack["name"] == "V2"
        assert len(pack["personas"]) == 2


# ---------------------------------------------------------------------------
# Panel results
# ---------------------------------------------------------------------------

class TestPanelResults:
    def test_list_empty(self):
        assert list_panel_results() == []

    def test_save_and_list(self):
        rid = save_panel_result(
            results=[{"persona": "Alice", "responses": []}],
            model="haiku",
            total_usage={"input_tokens": 100, "output_tokens": 50},
            total_cost="$0.001",
            persona_count=1,
            question_count=2,
        )
        assert rid.startswith("result-")

        results = list_panel_results()
        assert len(results) == 1
        assert results[0]["model"] == "haiku"
        assert results[0]["persona_count"] == 1

    def test_get_result(self):
        rid = save_panel_result(
            results=[{"persona": "Bob", "responses": [{"q": "hi", "a": "hello"}]}],
            model="sonnet",
            total_usage={"input_tokens": 200, "output_tokens": 100},
            total_cost="$0.01",
            persona_count=1,
            question_count=1,
        )
        result = get_panel_result(rid)
        assert result["model"] == "sonnet"
        assert result["id"] == rid
        assert len(result["results"]) == 1

    def test_get_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            get_panel_result("does-not-exist")
