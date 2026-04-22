# Pack Lifecycle and Delivery — Research Summary

## 1. Persona Pack Lifecycle — Authoring to Consumption

Four authoring paths:

1. **Manual YAML authoring** — create `name` + `personas: [{name, age, occupation, background, personality_traits}]`. Example format: `src/synth_panel/packs/developer.yaml:1-83`.
2. **`pack import`** (`cli/commands.py:2114-2144`): `synthpanel pack import file.yaml [--name] [--id]` → saved to `$SYNTH_PANEL_DATA_DIR/persona_packs/<pack_id>.yaml` (default `~/.synthpanel/persona_packs/`).
3. **`pack generate`** (`cli/commands.py:2188-2286`): LLM-driven synthesis via structured output. Fields enforced: `name`, `age`, `occupation`, `background`, `personality_traits`. Validated via `validate_persona_pack()` (data.py:204-241).
4. **SDK `save_persona_pack(name, personas, pack_id=None)`** (data.py:244-260). Auto-generates `pack-{8-char hex}` if no ID.

**Validation (data.py:204-241):**
- `personas` must be list, non-empty
- Each persona must be dict with required `name` (non-empty string)
- Optional: `age`, `occupation`, `background`, `personality_traits`
- `personality_traits` normalized to lowercase list (accepts CSV string or list)
- Raises `PackValidationError` on violation

**Usage in panel run:** user passes `--personas pack-id` or `--personas path.yaml`. Handler `_load_personas()` (`commands.py:461-478`) resolves as file path first, then pack ID (user-saved first, bundled fallback).

## 2. Instrument Pack Lifecycle

1. **Manual YAML authoring** — top-level manifest (`name`, `version` required) + instrument (`version: 1|2|3`, `rounds: [...]`). Example: `src/synth_panel/packs/instruments/product-feedback.yaml:1-47`.
2. **`instruments install`** (`cli/commands.py:2322-2372`): validates via `parse_instrument()` before saving to `$SYNTH_PANEL_DATA_DIR/packs/instruments/<name>.yaml`.
3. **SDK `save_instrument_pack(name, content)`** (data.py:330-348).

**Key difference vs personas:** instruments require explicit pre-install validation (line 2356-2363); personas only validate field presence/types. No `instruments import/export/generate` subcommands exist.

## 3. Bundled Pack Delivery

**Mechanism:** Packaged data in the wheel via `pyproject.toml` `[tool.setuptools.package-data]`:
```toml
"synth_panel.packs" = ["*.yaml"]
"synth_panel.packs.instruments" = ["*.yaml"]
```

**Discovery:** `_bundled_packs()` / `_bundled_instrument_packs()` (`mcp/data.py:83-124`) scans via `importlib.resources.files("synth_panel.packs")`. Directory scan for `.yaml` files; silently skips unparseable (line 98: `except Exception: continue`). Pack ID = filename stem.

**Inventory:** 9 persona packs (developer, enterprise-buyer, general-consumer, healthcare-patient, startup-founder, job-seekers, recruiters-talent, product-research, ai-eval-buyers) + 8 instrument packs.

**Update mechanism:** pip upgrade synthpanel → new packs appear on next `_bundled_packs()` call.

## 4. Namespacing and Collision Resolution

**Hierarchy (`list_persona_packs()` data.py:132-175):**
- User-saved packs listed first (take precedence on ID collision)
- Bundled packs listed only if not shadowed

**Collision behavior** (`test_mcp_data.py:137-147` `test_user_pack_overrides_bundled`):
- `get_persona_pack("developer")` checks `~/.synthpanel/persona_packs/developer.yaml` first
- Falls back to bundled only if file doesn't exist
- **No warning or error.** User override silently wins.

**No formal namespace for mayor's extra packs.** sp-g270 related but not implemented as a proper namespace.

## 5. User-Saved Pack Persistence

**Default location:** `~/.synthpanel/` via `_data_dir()` (`mcp/data.py:38-54`):
```python
Path(os.environ.get("SYNTH_PANEL_DATA_DIR", "~/.synthpanel")).expanduser()
```

**Structure:**
```
~/.synthpanel/
  persona_packs/<pack_id>.yaml
  packs/instruments/<instrument_id>.yaml
  results/<result_id>.json
```

**Override:** `SYNTH_PANEL_DATA_DIR` env var.

**Permission model:** No explicit mode setting; inherits umask. Linux typical: 644 user-writable. No documented privacy guarantees or world-read prevention.

**Test isolation (`test_mcp_data.py:12-15`):**
```python
@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
```
Autouse fixture applies to all tests in module.

## 6. Pack Subcommands

| Command | Args | Output | Handler |
|---|---|---|---|
| `pack list` | — | text or JSON list | 2096-2111 |
| `pack import` | `file [--name] [--id]` | confirmation string | 2114-2144 |
| `pack export` | `pack_id [-o out.yaml]` | YAML to stdout or file | 2147-2174 |
| `pack show` | `pack_id` | YAML to stdout | 2177-2185 |
| `pack generate` | `--product X --audience Y --count N` | confirmation string | 2188-2286 |

Parser: `parser.py:507-606`. No equivalent `instruments import/export/generate` — only `instruments install` is a write operation.

## 7. Schema & Versioning

**Persona pack schema** — implicit, code-enforced (no JSON Schema file):
```yaml
name: string          # pack-level, required
description: string   # optional
personas:
  - name: string      # REQUIRED per persona
    age: int          # optional
    occupation: string
    background: string
    personality_traits: string | list[string]  # normalized to lowercase list
```

**Instrument pack schema:**
```yaml
name: string      # required
version: int      # required (schema version 1/2/3)
description, author: optional
instrument:
  version: 1|2|3
  rounds: [...]   # v2+ only
  questions: [...] # v1 flat
```

Validation via `parse_instrument()` (not shown in this audit); called before save at `commands.py:2360`.

**Version tracking:**
- Persona packs: **no `version:` field supported.** No per-pack version. Overwriting same ID replaces content silently.
- Instrument packs: **required `version:` at top level.** Indicates instrument *schema* version (not pack version). All bundled instruments are `version: 1`. No interaction with synthpanel's own version.
- **No migration tooling.** Breaking schema changes would require manual user YAML updates.

**`_MANIFEST_FIELDS` shared pattern** (`data.py:61-69`):
```python
_MANIFEST_FIELDS = ("name", "version", "description", "author")
# Returns: {"id": pack_id, "name": ..., "version": ..., "description": ..., "author": ...}
```
Used by `list_instrument_packs()` for metadata display. Personas don't use `version`; defaults to empty string.

## Seams Observed

1. **Personas lack version field.** Unlike instruments. Overwriting pack ID = silent replace, no history/rollback.
2. **No schema registry.** Schemas inferred from Python validation functions, not published JSON Schema or OpenAPI.
3. **`SYNTH_PANEL_DATA_DIR` is the only configuration point.** No `~/.synthpanel/config.yaml`, no XDG alternative.
4. **Bundled pack immutability.** No in-place replacement mechanism; pip upgrade required.
5. **File-based collision resolution.** ID-based filename matching; no registry or metadata check for conflicts.
6. **LLM generation schema hardcoded.** `pack generate` schema defined inline in handler, not referenceable as a pack-level constraint.
7. **No published pack index.** No registry server, no PyPI-like discovery of third-party packs.
8. **Instruments and personas have separate mechanics.** Instruments have manifest `version` + pre-install validation; personas don't. Install paths diverge.

## Citations

mcp/data.py:1-533; cli/commands.py:2096-2372; cli/parser.py:507-606; tests/test_mcp_data.py:12-225; tests/test_cli.py:2087-2303; src/synth_panel/packs/*.yaml (9 bundled); packs/instruments/*.yaml (8 bundled); pyproject.toml.
