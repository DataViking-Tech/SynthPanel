# Pack Authoring Friction — Evidence Inventory

Systematic audit of 19 pack files (9 bundled + 10 custom) totaling 172 personas and 2,149 LOC.

## 1. Corpus Distribution

**Bundled (n=9):** 84 personas across 1,138 LOC. Range 63–248 LOC; 4–20 personas.
- Largest: product-research.yaml (248 LOC, 20 personas)
- Smallest: startup-founder.yaml (63 LOC, 4 personas)

**Custom (n=10):** 88 personas across 1,011 LOC. Range 24–183 LOC; 1–20 personas.
- Largest: extra_50_synthpanel.yaml (183 LOC, 20 personas)
- Smallest: contrarian_synthbench.yaml (29 LOC, 1 persona)

**Combined:** 2,149 LOC, 172 personas. Median pack ~100 LOC, ~8 personas.

## 2. Duplication and Drift

**Exact duplications (byte-identical):**

| Persona | Bundled Pack | Custom Pack | Notes |
|---|---|---|---|
| Abdul Rahman | job-seekers.yaml:130 | extra_50_traitprint.yaml:130 | OSS maintainer, age 32 |
| Ahmad Qureshi | ai-eval-buyers.yaml:105 | extra_50_synthbench.yaml:75 | ML research engineer |
| Aisha Nwosu | job-seekers.yaml:22 | extra_50_traitprint.yaml:13 | apprentice, age 19 |
| Alejandro Ruiz | ai-eval-buyers.yaml:165 | extra_50_synthbench.yaml:120 | ops engineer |
| Andreas Weber | product-research.yaml:191 | extra_50_synthpanel.yaml:140 | growth marketer |
| Beatrice Laurent | product-research.yaml:106 | extra_50_synthpanel.yaml:164 | VP product, healthcare SaaS |
| Carla Mendes | ai-eval-buyers.yaml:10 | extra_50_synthbench.yaml:10 | ML engineer |
| (~40 total personas appear in 2+ packs) | | | |

**Pattern:** `extra_50_*` packs are **wholesale copies** of specific bundled packs, not augmentations. No drift detected — when names match, content is byte-identical.

**No same-archetype-different-spelling found.**

## 3. YAML Structural Patterns

**Universal (all 19 packs):**
```yaml
name: <string>
description: <string>
personas:
  - name: <string>
    age: <integer>
    occupation: <string>
    background: <string>  # multiline ">" folded
    personality_traits: <list[string]>
```

**Uniformity:** 100% — all 172 personas have exactly these 5 fields. No missing optional fields, no extra undeclared keys.

**Stylistic variance in `personality_traits`:**
- Bundled packs: atomic 2-3 word strings. Example `["PM buyer", "discovery-constrained", "speed-seeker"]` (product-research.yaml:17-20)
- Custom ICP packs: compound complex phrases. Example `["methodologically trained, values validity over speed"]` (icp_synthpanel.yaml:17-22)
- Contrarian pack: longest traits, heavily qualified. Example `"thinks 'zero-config sampling mode' incentivizes bad research"` (contrarian_synthpanel.yaml:28)

**Schema enforcement:** `mcp/data.py:204-241` — no published schema file; inline Python validation only.

## 4. Validation Error Paths

**Required fields:** `name` (non-empty string) per persona. `personas` must be list, non-empty.

**Optional fields (not validated beyond presence):** `age`, `occupation`, `background`, `personality_traits`.

**`personality_traits` normalization:** string (CSV) → list of lowercase strings (data.py:232-233: `traits = [str(t).strip().lower() for t in traits if str(t).strip()]`). **Silently downcases compound statements.** "Methodologically Trained" → "methodologically trained". No user guidance; icp packs use downcased traits anyway.

**User-facing messages:**

| Scenario | Message | File:line |
|---|---|---|
| Missing `name` | `persona at index {i} is missing required field 'name'` | mcp/data.py:223 |
| `personas` not list | `personas must be a list` | :214 |
| Empty personas | `personas list must not be empty` | :216 |
| Persona not dict | `persona at index {i} must be a dict` | :221 |
| Invalid YAML | (yaml.safe_load generic error) | commands.py:2119 |
| File not found | `Error loading file: {exc}` | :2121 |
| `personality_traits` wrong type | `personality_traits must be a list or comma-separated string` | data.py:235 |

**Path traversal protection:** `_validate_pack_id()` (data.py:32-35) rejects IDs with `/` or `..`.

**SDK validation** (`sdk.py:293-300`): mirrors CLI; enforces `name` + count ≤ `MAX_PERSONAS = 200`.

## 5. Extra Packs — Relationship to Bundled

| Extra Pack | Bundled Equivalent | Overlap |
|---|---|---|
| extra_50_synthpanel.yaml | product-research.yaml | 20/20 byte-identical |
| extra_50_synthbench.yaml | ai-eval-buyers.yaml | 20/20 byte-identical |
| extra_50_traitprint.yaml | job-seekers.yaml | 15/15 byte-identical |
| extra_100_topup.yaml | (none directly) | 10 mixed from ICP + contrarian archetypes |

**Net-new content in `extra_50_*`:** Zero. These exist solely to scale bundled personas to n=50/100 for audits. Naming implies augmentation; reality is duplication.

**Contrarian + ICP packs (different pattern):** genuine authored content. Single stress-test persona (Dr. Anya Volkov) in `contrarian_*`; five custom ICP personas in `icp_*`.

## 6. Informal Sharing Channels

**None documented.**

Searches across `docs/`, READMEs, CONTRIB guides — no mention of "share a persona pack", "pack distribution", "community packs".

Existing docs (`docs/ensemble.md`, `docs/convergence.md`) show `--personas file.yaml` with implicit local authorship. `docs/agent-integration-landscape.md` references "registry presence" only in MCP/Composio context, not pack registries.

**Inferred informal sharing pathway** (from code, not explicit): user authors local YAML → `pack import` → saves to `~/.synthpanel/persona_packs/` → `pack export` to stdout → email/Slack/PR to share.

## 7. Schema Definition

**Implicit, code-enforced. No declarative schema file.**

Loci:
- Persona schema: `mcp/data.py:204-241` `validate_persona_pack()`
- Pack manifest: `mcp/data.py:57-69` `_extract_manifest()`. Expected keys per `_MANIFEST_FIELDS = ("name", "version", "description", "author")`. Graceful degradation to empty string if missing.
- Instrument schema: `mcp/data.py:306-327` (similar inline, not scoped here)

**No JSON Schema, GraphQL, or Pydantic model.** No `.schema.yaml` or docstring type hints for the YAML structure.

**Validation triggers:**
| Path | Location | When |
|---|---|---|
| CLI import | commands.py:2114-2144 | `pack import <file>` |
| SDK runtime | sdk.py:313-324 | `run_panel(personas=..., pack_id=...)` |
| MCP save | data.py:244-260 | `save_persona_pack()` |
| MCP load | data.py:178-197 | `get_persona_pack(pack_id)` — **no validation; assumes valid** |

## Friction-Hotspot Inventory

1. **Duplication without net-new content.** `extra_50_*` packs exist as copies, not extensions. 40+ personas appear in 2+ packs.
2. **Implicit schema with no user-facing documentation.** No README section defining required/optional fields. Authoring friction surfaces only as terse runtime errors.
3. **`personality_traits` format ambiguity.** Atomic vs compound stylistic choice undocumented; both pass validation. Inconsistency hurts downstream analysis.
4. **No version/upgrade semantics for personas.** Schema evolution has no migration path. Breaking changes would silently fail validation.
5. **No naming-convention or namespace-collision detection.** Silent shadowing. `data.py:162` checks `if pack_id in seen` without warning.
6. **No informal sharing mechanism or registry.** Custom packs (ICP, contrarian) live in mayor/results/ but have no upstream-facing distribution.
7. **`extra_100_topup.yaml` has unclear provenance.** Description doesn't clarify source. Pack name suggests generic but content is domain-specific.
8. **Validation at import/save only, not at load.** Manual edits to user-saved packs that break schema surface only at panel run, not at load.
9. **No test coverage for cross-pack integrity.** Tests import/export individual packs; no check for duplication, collision, or drift. No lint command to detect issues before shipping.
10. **`personality_traits` silently downcases.** May obscure capitalization-sensitive meaning. Rule undocumented.

## Summary

Authoring friction exists primarily around **duplication without extension**, **implicit schema**, **no versioning path**, and **no discovery/registry mechanism**. The 19-pack corpus shows 172 personas, but at least 40 are duplicates across packs, indicating scale-out via copy-paste is the current workflow.
