# SynthBench ↔ SynthPanel Data Flow — Research Summary

## 1. SynthBench's Ingestion of Synthpanel Output

### Provider contract

`SynthPanelProvider` (`src/synthbench/providers/synthpanel.py:287-400`) implements two execution paths:

- **Direct API path** (`_HAS_SYNTH_PANEL_API=True`): imports `synth_panel.llm.client.LLMClient`, uses `ThreadPoolExecutor` for concurrency. Zero subprocess overhead (~1s/call saved).
- **CLI fallback**: `synthpanel` binary via subprocess when API import fails.

Config:
- `model` (str): alias or canonical name
- `temperature` (float | None)
- `profile` (str | None)
- `prompt_template` (str | None)

### Data consumed

From `synthpanel panel run --output-format json`:
```
data["rounds"][0]["results"][*].responses[i].response  → raw text
data["rounds"][0]["results"][*].usage                  → token metadata
data["rounds"][0]["results"][*].error                  → panelist-level error flag
data.model / data.total_cost / data.panelist_cost     → metadata
```

Provider extracts: (1) raw `.response` text per panelist × question; (2) selected option via letter parser (`_parse_letter()`) or substring match; (3) refusal count when parse fails; (4) aggregated token usage.

**Key subset:** provider reads only `responses[].response` field per panelist per question. Metadata fields (cost/usage) are optional, used for attribution/debugging.

### Idempotency

**Re-runnable, no cache.** Each `get_distribution()` or `batch_get_distribution()` call re-invokes synthpanel. No checkpointing. By design.

## 2. Question Representation & Distribution Keying

### `Question` dataclass (`src/synthbench/datasets/base.py:34-50`)

```python
@dataclass
class Question:
    key: str                                    # Stable question identifier
    text: str                                   # Full survey question text
    options: list[str]                          # Answer choices as strings
    human_distribution: dict[str, float]        # Option → probability mass
    survey: str = ""                            # Dataset + wave
    topic: str = ""                             # Topical label
```

### Keying

**Human distribution keyed by option string, not ordinal index or enum token.**

```json
{"Very satisfied": 0.35, "Somewhat satisfied": 0.42, "Neutral": 0.15, "Dissatisfied": 0.08}
```

Distributions normalized in `__post_init__` if sum deviates from 1.0 by >1%. `options` list defines canonical ordering; probabilities aligned by string match. No separate enum/token layer — option text **is** the key.

## 3. Question Matching

**Explicit by dataset structure.** Not text similarity, not manual alignment.

Flow:
1. `Dataset.load(n=n)` returns `list[Question]` with pre-set `key`, `text`, `options`, `human_distribution`
2. Provider called with `question.text` + `question.options` directly
3. Provider returns `Distribution` aligned to `question.options` ordering

**No fuzzy-matching, no registry of synthpanel-output-question-name → SynthBench-key.** Integration assumes: SynthBench orchestrates (picks dataset, loads questions, calls provider); SynthPanel is invoked *only* by SynthBench's provider; `Question.key` is internal to dataset adapter, not exposed to synthpanel.

## 4. Metric Computation

### JSD (`src/synthbench/metrics/distributional.py:6-34`)

```python
def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    jsd = jensenshannon(p_vec, q_vec, base=2) ** 2
    return float(jsd)
```

### Input contract

- `p`, `q` = dicts mapping option string → probability
- **Disjoint supports handled** — missing keys padded with 0.0
- Both normalized internally

### Output

- Single `float` in [0.0, 1.0]
- 0 = identical, 1 = maximally divergent
- Returned per question in `QuestionResult.jsd` (runner.py:180-200)

### Run-level

`BenchmarkResult` (runner.py):
```python
BenchmarkResult(
    provider_name: str,
    dataset_name: str,
    questions: list[QuestionResult],
    config: dict,
    elapsed_seconds: float,
)
```

Each `QuestionResult`:
```python
jsd: float
kendall_tau: float
parity: float  # composite metric
human_distribution: dict[str, float]
model_distribution: dict[str, float]
n_samples: int
model_refusal_rate: float
```

## 5. Cross-Package Dependency Posture

**One-way: SynthBench → synthpanel.** Provider calls synthpanel; synthpanel does NOT import SynthBench in core paths.

**Optional extras** (`synthpanel/pyproject.toml:46-48`):
```toml
[project.optional-dependencies]
convergence = ["synthbench>=0.1"]
```

**Lazy import** (`src/synth_panel/convergence.py:554-608`): `load_synthbench_baseline(spec: str)` attempts `import synthbench` inside the function; raises `SynthbenchUnavailableError` with install instructions if missing. Calls synthbench's exported loader (tries `load_convergence_baseline`, `load_baseline`, `convergence_baseline` attribute). Returns baseline payload dict (human distribution + convergence metadata).

**CLI flag** (`--convergence-baseline "gss:happiness"`): opt-in, only triggered if user explicitly requests.

**No circular dependency.** Synthpanel core CLI works without synthbench. SynthBench includes synthpanel as a provider. Synthpanel imports synthbench lazily, at CLI time not import time.

## 6. Nine Datasets & Redistribution Tiers

All in `src/synthbench/datasets/__init__.py:DATASETS`:

| Dataset | Policy | License URL | Citation |
|---|---|---|---|
| **gss** | `full` | gss.norc.org | NORC General Social Survey |
| **ntia** | `full` | ntia.gov (public domain, 17 USC 105) | US Gov Internet Use Survey |
| **wvs** | `gated` | worldvaluessurvey.org | WVS Wave 7 |
| **michigan** | `gated` | sca.isr.umich.edu | UMich Survey of Consumers |
| **pewtech** | `gated` | pewresearch.org | Pew Research Internet & Tech |
| **eurobarometer** | `gated` | gesis.org | EC / GESIS Consumer Modules |
| **opinionsqa** | `gated` | codalab.org | (no explicit license URL) |
| **globalopinionqa** | `gated` | CC BY-NC-SA 4.0 | Durmus et al. 2023, Anthropic |
| **subpop** | `gated` | huggingface.co/datasets/jjssuh/subpop | Suh et al., ACL 2025 |

### Tier definitions (`base.py:72-100`)

- **`full`**: per-question distributions publishable on static site (public domain or explicit permissive license)
- **`gated`**: per-question artifacts ship to authenticated R2 bucket; public sees sign-in gate
- **`aggregates_only`** (default): only aggregate scores public; no per-question
- **`citation_only`**: only metadata (text, options); metrics suppressed

## 7. SynthBench CLI Surface

### `synthbench run` (cli.py:41-279)

```bash
synthbench run --provider raw-anthropic --model haiku --dataset opinionsqa --n 100 --samples 30 --output results/
synthbench run --provider synthpanel --model sonnet --dataset gss --submit --wait
```

Key calibration flags:
- `--provider {raw-anthropic,raw-openai,raw-gemini,openrouter,ollama,synthpanel,http}`
- `--model`, `--dataset`, `--samples`, `--suite {smoke,core,full}`, `--topic {political,consumer,neutral}`
- `--baselines-dir PATH`
- `--demographics` (AGE, POLIDEOLOGY, ...)
- `--full-evaluation`, `--submit`, `--wait`

**Output:** JSON per-question JSD / Kendall's τ / parity / model refusal rate / token usage.

**No dedicated `synthbench report` subcommand.** Results land in `--output` directory as JSON.

## 8. Inline-Consumption Integration Points (synthpanel side)

### Current convergence output structure (sp-0h9x, sp-yaru)

```json
{
  "rounds": [...],
  "model": "...",
  "total_cost": "...",
  "metadata": {...},
  "synthesis": {...},
  "convergence": {
    "final_n": 500,
    "epsilon": 0.02,
    "auto_stopped": false,
    "overall_converged_at": 250,
    "tracked_questions": ["q1", "q2"],
    "per_question": {
      "q1": {"converged_at": 200, "curve": [{"n": 50, "jsd": 0.15}, ...], "support_size": 4}
    },
    "human_baseline": {...}   // spliced from SynthBench if --convergence-baseline set
  },
  "per_model_results": [...],
  "cost_breakdown": {...},
  "warnings": [...]
}
```

### Convergence flow (sp-yaru)

1. **Question filtering** (`convergence.py:163-176`): `identify_tracked_questions()` drops free-text, keeps bounded (Likert / yes-no / pick-one / enum / ranking). Returns `[(index, key, question_dict)]`.
2. **Live recording** (`convergence.py:373-399`): each completed panelist feeds into `convergence_tracker.record()`. Extracts categorical value via `extract_category()`. Updates running Counter. Every K (default 20), computes rolling JSD. Auto-stop when JSD < epsilon for M consecutive checks.
3. **Post-run report** (`convergence.py:411-445`): `build_report(baseline=...)` returns dict with per-question curves, converged-at thresholds, overall flag. Baseline payload spliced verbatim under `human_baseline`.

### Backward compat

`convergence` is sibling to existing keys. Parsers ignoring unknown keys unaffected. No breaking changes.

## Seams Observed

Natural integration points for inline calibration:

1. **Per-question JSD attachment** — synthpanel already loads SynthBench baseline via `--convergence-baseline`. Could emit `per_question[key].human_jsd` during run. No data-model changes needed; wire additional metric computation in `_run_check_locked()`.
2. **Baseline question registry** — SynthBench datasets have stable keys (`"gss:spkath"`). Synthpanel instruments could tag questions with synthbench keys. `load_synthbench_baseline()` could load a full mapping payload, enabling per-question human distributions without manual alignment.
3. **Output schema versioning** — `convergence` is opt-in; could become default when inline calibration is. `per_question_metrics` sibling could house JSD + refusal + token attribution per question without enlarging `convergence`.
4. **CLI integration** — `synthpanel panel run --calibrate-against {gss,opinionsqa}` shorthand for `--convergence-baseline gss:*`. `synthpanel panel inspect --calibration` rendering.
5. **MCP attachment** — `compare_against_synthbench(result_id, dataset, question_key)` tool in MCP server.

Only current seam is `load_synthbench_baseline()`; no reverse direction needed beyond the optional `convergence` extra already declared.
