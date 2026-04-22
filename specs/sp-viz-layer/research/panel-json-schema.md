# Panel-Result JSON Schema — Research Summary

**Scope:** `src/synth_panel/` + tests + docs, cross-validated against real artifacts in `/Users/openclaw/gastown-dev/mayor/results/synthpanel-self-audit/`.

## 1. Schema Definition: Locus and Versioning

**Where defined:** Inferred from code, not a formal schema artifact. Three loci:
- `src/synth_panel/sdk.py:224–286` — `PanelResult`, `PollResult`, `PromptResult` dataclasses. `to_dict()` at 163–165 serializes to plain dict.
- `src/synth_panel/synthesis.py:64–97` — `_SYNTHESIS_SCHEMA` JSON Schema for the synthesis object.
- `src/synth_panel/mcp/data.py:482–530` — `save_panel_result()` documents which top-level fields are saved; optional fields are conditionally included (521–530).

**Versioning:** No explicit `schema_version` field. CHANGELOG references output-shape changes via ticket IDs (sp-avmm, sp-g270, sp-kkzz) but does not bump any schema version.

**Back-compat strategy:** Optional fields default in dataclasses (e.g., `cost_is_estimated: bool = False` at sdk.py:279). Old results lacking them deserialize cleanly. Example: `synthesis_error` added in sp-avmm (sdk.py:285); `terminal_round` present only on multi-round runs.

**Canonical contract (prose):** `docs/mcp.md:99–134` — JSON example showing `rounds`, `path`, `terminal_round`, `synthesis`, `total_cost`, `total_usage`, `results`.

## 2. Distinct Sections and Shapes

### Top-level scalars & metadata

| Field | Type | Always | Notes |
|---|---|---|---|
| `result_id` | string | ✓ | `result-YYYYMMDD-HHMMSS-<hex>` |
| `message` | string | ✓ | CLI-emitted human status |
| `model` | string | ✓ | Panelist model alias |
| `persona_count`, `question_count` | int | ✓ | |
| `total_cost` | string | ✓ | USD formatted (`"$0.1234"`) |
| `panelist_cost` | string | ◐ | Ensemble runs (sp-atvc) |
| `total_usage` | dict | ✓ | Token bucket (see below) |
| `panelist_usage` | dict | ◐ | Multi-question CLI runs (sp-027) |
| `run_invalid` | bool | ◐ | Default False (sp-bjt4) |
| `cost_is_estimated` | bool | ◐ | Default False (sp-nn8k) |

### Rounds (per-round results)

```python
rounds: list[{
    "name": str,
    "results": list[PanelistResult],
    "synthesis": dict | None,
    "usage": dict | None,  # map-reduce only
}]
```

### Panelist result (per-persona)

```python
rounds[i]["results"][j] = {
    "persona": str,
    "responses": list[{"question": str, "response": str, "extraction": dict | None}],
    "usage": dict,
    "cost": str,  # pre-formatted USD
    "error": str | None,
    "model": str | None,  # ensemble only
}
```

### Synthesis

Serialized from `SynthesisResult` (synthesis.py:104–147). Keys: `summary`, `themes`, `agreements`, `disagreements`, `surprises`, `recommendation`, `usage`, `cost`, `model`, `prompt_version`, plus strategy-dependent `strategy`, `per_question_synthesis` (dict[str,str] keyed by question index — strings for JSON round-trip), `map_cost_breakdown`.

### Path (multi-round routing)

```python
path: list[{"round": str, "branch": str, "next": str | None}]
```

Empty list for v1/v2 single-round. `orchestrator.py:815` renders branch descriptions.

### Metadata (provenance & audit)

```python
metadata: dict | None = {
    "version": {"synthpanel": str, "python": str},
    "models": {"panelist": str, "synthesis": str | None},
    "generation_params": {"temperature": float|None, "top_p": float|None, "max_tokens": int},
    "config_hash": str,  # sp-ui40
    "cost": {"total_tokens": int, "total_cost_usd": float, "per_model": dict},
    "timing": {"total_seconds": float, "per_panelist_avg": float},
    "template_vars_fingerprint": dict[str, str],
}
```

Pre-metadata results have `metadata: None`. `per_model` bucketing is ensemble-specific (sp-atvc).

### Ensemble-specific

```python
cost_breakdown: {"by_model": dict[str, str], "total": str}  # sp-gl9
model_assignment: dict[str, str]  # persona → model
per_model_results: [...]  # ensemble runs
```

### Warnings & failure stats

`warnings` (list[str]), `failure_stats` (errored_pairs/failure_rate/failed_panelists), `synthesis_error` (dict added in sp-avmm), `missing_input_stats`.

## 3. Canonical vs Optional vs Strategy-Dependent

**Always canonical (all versions):** `result_id`, `model`, `persona_count`, `question_count`, `total_cost`, `total_usage`, `rounds` (≥1), `path` (may be empty), `synthesis` (may be None).

**Optional (context-dependent):** `metadata`, `terminal_round`, `results` (back-compat flat shape), `panelist_cost`, `panelist_usage`, `per_model_results`, `cost_breakdown`, `model_assignment`, `warnings`, `failure_stats`, `run_invalid`, `synthesis_error`, `missing_input_stats`, `convergence` (sp-yaru, opt-in).

**Strategy-dependent (synthesis.py:141–147):**
- `synthesis.strategy` — omitted for single, `"map-reduce"` otherwise
- `synthesis.per_question_synthesis` — None for single, dict for map-reduce (string keys)
- `synthesis.map_cost_breakdown` — None/list[dict]
- `rounds[i].usage` — map-reduce only

## 4. Schema Stability

**Unstable with back-compat guardrails.** Recent changes:
| Ticket | v | Change |
|---|---|---|
| sp-avmm | 0.9.8 | Added `synthesis_error`, `run_invalid` |
| sp-kkzz | 0.9.8 | Added `strategy`, `per_question_synthesis`, `map_cost_breakdown` |
| sp-atvc | 0.9.8 | Ensemble `per_model` bucket; added `cost_breakdown`, `model_assignment` |
| sp-gl9 | 0.9.8 | `per_model_results` / `cost_breakdown` on non-ensemble runs |
| sp-g270 | 0.9.8 | `personas_merge_warnings` array |
| sp-027 | 0.9.7 | `panelist_usage` on multi-question CLI |
| sp-ui40 | 0.9.7 | `template_vars_fingerprint`; config_hash includes resolved vars |

No deprecations documented. No removals announced. Consumers use presence checks.

## 5. Observed Panel-Result JSON Sizes

| Source | n | Size |
|---|---|---|
| round3/synthpanel__landing-page-comprehension | 6 | 51 KB |
| ensemble_30/synthpanel__product-feedback | 30 | 1.2 MB |
| ensemble_50/synthpanel__product-feedback | 50 | 2.0 MB |
| ensemble_100/synthpanel__product-feedback | 100 | 3.9 MB |
| ensemble_100_v099_ctx2/synthpanel__product-feedback | 100 | 6.7 MB |

Linear: ~40 KB/persona baseline. Crosses 1 MB at n≈25. 10 MB extrapolates to n≈250. Largest observed: 6.7 MB.

## Seams observed

1. **`analysis/inspect.py`** — `InspectReport` dataclass (line 75) walks schema for per-persona/per-model/synthesis-status summaries. No CLI subcommand exposes it beyond `panel inspect`.
2. **Per-response `extraction` field** (`mcp/data.py:507`) — optional structured-output dict persisted but no downstream consumer documented.
3. **`metadata.cost.per_model`** — multi-model cost breakdown structured, not surfaced in CLI summary.
4. **Session persistence** (`persistence.py`, `mcp/data.py:385–403`) — `{result_id}.sessions/{persona_name}.json` enables replay/extend but no browsing API.
5. **Synthesis strategy metadata** (`synthesis.py:141–147`) — map-reduce intermediate summaries serialized but not rendered.
6. **`metadata.template_vars_fingerprint`** — available for cross-run diffing; no diff utility exposed.
7. **Failure stats granularity** (`failure_stats.errored_personas`) — per-persona error lists captured but not surfaced.

## Citations

sdk.py:163–286; synthesis.py:64–147; mcp/data.py:482–530; analysis/inspect.py:75–97; docs/mcp.md:99–134; CHANGELOG.md lines 12–73.
