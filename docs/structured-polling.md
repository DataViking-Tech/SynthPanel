# Structured Polling — the Agent Default

Free-text panel output is great for human researchers. For agents it's
a liability: every `result["rounds"][0]["results"][i]["response"]`
needs prose parsing, every prose parser is a regex you'll regret, and
every regex breaks the moment a model rephrases.

**Althing has had schema-enforced polling since 0.9.8.** This page
documents the path so agents stop asking panelists for "JSON in prose"
and start getting parsed, typed answers back directly. No prose
parsing. No regex. No post-hoc extraction step unless you want one.

## TL;DR

For agent workflows, **always pass a bounded `response_schema`** when
running a poll. The four supported schema types map to the polling
patterns you'll actually need:

| Pattern                                     | `response_schema.type`              |
|---------------------------------------------|-------------------------------------|
| Forced-choice (pick one of N)               | `enum`                              |
| Likert / 1–N rating                         | `scale`                             |
| Confidence / familiarity (1–N)              | `scale`                             |
| Objections, risks, themes from a fixed list | `tagged_themes` (with `multi: true`) |
| Ranking (whole list)                        | free text + `--extract-schema ranking` |
| Free text + post-hoc structured pull        | free text + `--extract-schema <name-or-json>` |

When `response_schema` is set, each panelist's `response` field is a
parsed **dict** (not free text) and `"structured": true` is attached
so downstream consumers can branch on it.

The bundled JSON Schemas and matching Pydantic models live at:

- `src/althing/structured/schemas.py` — wire-format JSON Schemas
  (`ranking`, `likert`, `yes_no`, `pick_one`, `annotated_choice`).
- `src/althing/structured/models.py` — Pydantic mirrors with
  tighter constraints (e.g. `Likert.rating` is constrained to 1..5).

Pass either a bundled name or an inline JSON Schema dict to
`--extract-schema` (CLI) or `extract_schema=` (MCP).

## The five patterns

Every example below is the *whole* invocation. Pick the surface that
matches your runtime — CLI for shell agents, MCP for editor agents,
Python SDK for in-process agents. All three drive the same engine.

### 1. Forced-choice selection — `enum`

> "Which positioning wins: A, B, C, or D?"

Bounded categorical. The model is constrained to emit exactly one of
the listed options at generation time via tool-use forcing;
out-of-vocabulary outputs trigger the 3-strike schema-drift retry path
(see [docs/response-contract.md](response-contract.md)).

**Instrument YAML:**

```yaml
# positioning.yaml
instrument:
  version: 1
  questions:
    - text: >
        Which positioning best matches how you'd describe this product to a
        peer?
        A) The fastest synthetic-panel runtime
        B) The most calibrated synthetic panel for product decisions
        C) The MCP server every agent already speaks
        D) The cheapest way to pre-screen survey instruments
      response_schema:
        type: enum
        options: ["A", "B", "C", "D"]
```

**CLI:**

```bash
# Export the bundled pack to a file, then run.
althing pack export product-research > /tmp/product-research.yaml
althing panel run \
  --personas /tmp/product-research.yaml \
  --instrument positioning.yaml \
  --output-format json --save
```

**MCP tool call:**

```jsonc
{
  "tool": "run_panel",
  "arguments": {
    "pack_id": "product-research",
    "instrument": {
      "version": 1,
      "questions": [{
        "text": "Which positioning best matches...",
        "response_schema": { "type": "enum", "options": ["A", "B", "C", "D"] }
      }]
    },
    "decision_being_informed": "picking the launch positioning"
  }
}
```

**Python SDK:**

```python
from althing import run_panel

result = run_panel(
    pack_id="product-research",
    instrument={
        "version": 1,
        "questions": [{
            "text": "Which positioning best matches...",
            "response_schema": {"type": "enum", "options": ["A", "B", "C", "D"]},
        }],
    },
)
```

**What the agent reads.** Each panelist's response is a dict with
`"structured": true`. The dict's payload key varies per model (the
JSON Schema generator names it via the schema's intent) — use the
helper below or `althing.analysis.subgroup_cli._flatten_structured_value`
to unwrap to the scalar:

```python
def pick(resp: dict) -> str:
    """Unwrap an enum response to its chosen option."""
    if not resp.get("structured"):
        raise ValueError("response is not structured")
    val = resp["response"]
    if isinstance(val, str):
        return val
    for key in ("option", "answer", "value", "choice", "response"):
        if isinstance(val.get(key), str):
            return val[key]
    raise ValueError(f"could not unwrap enum response: {val!r}")

votes = [pick(r) for round_ in result["rounds"] for r in round_["results"]
         if not r.get("error")]
# {"A": 3, "B": 11, "C": 4, "D": 2}
counts = {opt: votes.count(opt) for opt in ["A", "B", "C", "D"]}
```

### 2. Likert / scale rating — `scale`

> "On a 1–5 scale, how likely are you to recommend this to a colleague?"

```yaml
# nps.yaml
instrument:
  version: 1
  questions:
    - text: "On a scale of 1-5, how likely are you to recommend this to a colleague?"
      response_schema:
        type: scale
        min: 1
        max: 5
```

```bash
althing pack export ai-eval-buyers > /tmp/ai-eval-buyers.yaml
althing panel run --personas /tmp/ai-eval-buyers.yaml --instrument nps.yaml \
  --output-format json --save
```

The agent unwraps each `response` dict (`score`/`rating`/`value` key)
and computes the mean in one line. Or — for the canonical aggregate —
run `althing analyze <result-id> --output json`, which calls
`althing.analysis.distribution.distribution_for_question()` and
returns:

```jsonc
{
  "Q0": {
    "type": "scale",
    "n": 35, "n_valid": 35, "n_invalid": 0,
    "frequencies": {"1": 0, "2": 1, "3": 5, "4": 16, "5": 13},
    "mean": 4.17, "median": 4, "stdev": 0.74,
    "min": 2, "max": 5
  }
}
```

**Pydantic alternative.** If you want a typed model, pass
`extract_schema="likert"` to either the CLI (`--extract-schema likert`)
or MCP (`extract_schema="likert"`). Each panelist responds in free
text, then a second LLM call extracts `{"rating": int, "reasoning":
str}` and stores it under an `extraction` key alongside the raw text.
The `Likert` Pydantic model enforces `1 <= rating <= 5` at parse time;
violations land as a `pydantic.ValidationError` with the field path,
not as silent garbage.

### 3. Objections / risks — `tagged_themes`

> "Which risks would you raise about this proposal? (Pick all that apply.)"

`tagged_themes` is the structured equivalent of "give me your
objections." You define the taxonomy; the model picks the subset that
applies; the engine returns the picked list. `multi: true` allows
multi-select.

```yaml
# risks.yaml
instrument:
  version: 1
  questions:
    - text: >
        Which of these risks would you raise about adopting a synthetic-panel
        runtime to inform product decisions?
      response_schema:
        type: tagged_themes
        multi: true
        taxonomy:
          - hallucination
          - lacks-real-user-grounding
          - vendor-lock-in
          - cost-unpredictable
          - compliance-or-privacy
          - team-skill-gap
          - integration-effort
          - none
```

Each panelist returns a list (`["hallucination", "cost-unpredictable"]`).
`althing analyze` aggregates per tag, zero-filling unmentioned
taxonomy entries plus a bucket for off-taxonomy `other` themes so the
agent can rank risks by mentions without parsing prose.

### 4. Confidence / familiarity — `scale`

Same shape as Likert; conventionally 1–5 or 1–7. Pair it with the main
question to gate downstream logic on certainty.

```yaml
# confidence.yaml
instrument:
  version: 1
  questions:
    - text: "How familiar are you with MCP servers? (1=never heard of it, 5=I run one)"
      response_schema:
        type: scale
        min: 1
        max: 5
    - text: "Would you trust an MCP-based synthetic panel to pre-screen a survey?"
      response_schema:
        type: enum
        options: [yes, no, depends]
```

An agent can split panelists by self-rated familiarity *before*
reading the trust answer — the per-panelist `response` records are
indexed by panelist so the join is a dict lookup, not a regex.

### 5. Ranking (whole list) — `--extract-schema ranking`

Ranking is not a `response_schema` type because the answer is a list,
not a scalar. Use post-hoc extraction instead. The bundled `ranking`
schema produces `{"ranked": [{"name", "rank", "reasoning"}]}` per
panelist, validated by the `Ranking` Pydantic model.

```yaml
# ranking.yaml
instrument:
  version: 1
  questions:
    - text: >
        Rank these features from most to least valuable for your team
        in the next 90 days:
        market_pull, demoability, trust_burden, cost_transparency, ensemble_support.
        Return as a ranking with brief reasoning per item.
      response_schema:
        type: text
```

```bash
althing panel run \
  --personas /tmp/product-research.yaml \
  --instrument ranking.yaml \
  --extract-schema ranking \
  --output-format json --save
```

Each panelist response then carries an `extraction` field:

```jsonc
{
  "response": "1. market_pull — solves...\n2. demoability — ...",
  "extraction": {
    "ranked": [
      { "name": "market_pull", "rank": 1, "reasoning": "solves the cold-start..." },
      { "name": "demoability", "rank": 2, "reasoning": "..." }
    ]
  },
  "extraction_is_fallback": false
}
```

For aggregating into a panel-wide ranking, use Borda count or mean
rank across the per-panelist `extraction.ranked` lists.

## A realistic 35-persona prioritization poll

The acceptance criterion for this feature is "examples include a
realistic 30–40 persona prioritization poll." Here is one — runnable
end-to-end with the bundled packs.

The bundled persona packs ship 15–20 personas each. For 35 personas,
merge `product-research` (20) with `ai-eval-buyers` (15) at the CLI:

```bash
althing pack export product-research > /tmp/product-research.yaml
althing pack export ai-eval-buyers   > /tmp/ai-eval-buyers.yaml

althing panel run \
  --personas       /tmp/product-research.yaml \
  --personas-merge /tmp/ai-eval-buyers.yaml \
  --instrument     prioritize.yaml \
  --model haiku \
  --output-format json --save
```

`prioritize.yaml` runs three bounded questions in one pass — pick-one,
confidence, multi-select risks — so the agent gets the whole signal
from a single panel run:

```yaml
# prioritize.yaml
instrument:
  version: 1
  questions:
    - text: >
        Which of these positioning lines best matches how you'd describe
        this product to a peer?
        A) The fastest synthetic-panel runtime
        B) The most calibrated synthetic panel for product decisions
        C) The MCP server every agent already speaks
        D) The cheapest way to pre-screen survey instruments
        E) The drop-in MCP tool for evidence-based product decisions
      response_schema:
        type: enum
        options: ["A", "B", "C", "D", "E"]

    - text: "How confident are you in that pick? (1=guessing, 5=certain)"
      response_schema:
        type: scale
        min: 1
        max: 5

    - text: "Which risks would you flag about that positioning?"
      response_schema:
        type: tagged_themes
        multi: true
        taxonomy:
          - vague
          - overpromises
          - buzzwordy
          - narrow-audience
          - category-confusion
          - credible
          - none
```

Cost on `haiku` for 35 personas × 3 bounded questions: typically
**$0.05–$0.10**. Compare to a real panel of equivalent size: $5,000
and three weeks. The point isn't that synthetic replaces real — it's
that the cost curve makes structured prioritization a *first* step,
not a *last* step.

### Computing the headline summary

Two equivalent paths. Pick whichever fits your agent runtime.

**(a) `althing analyze` — let the package do it for you.**

```bash
althing analyze <result-id> --output json > analysis.json
```

`analysis.json` contains a `distribution_by_question` block with per-
question frequencies, mean/median/stdev for scales, and zero-filled
taxonomy counts for `tagged_themes`. Closed-shape, agent-readable, no
prose parsing.

**(b) Compute from the panel result directly.** When the agent already
has the panel result in memory (e.g. from an MCP `run_panel` call),
import the same routines the CLI uses:

```python
from althing.analysis import distribution_for_question

# Q0 (enum) — pick-one
q0_responses = [
    _unwrap_structured(r, "Q0", "enum")
    for round_ in result["rounds"]
    for r in round_["results"]
    if not r.get("error")
]
q0_dist = distribution_for_question(q0_responses, {"type": "enum",
                                                    "options": ["A","B","C","D","E"]})
# {"type":"enum","frequencies":{"A":3,"B":17,"C":7,"D":4,"E":4,"other":0},
#  "n":35,"n_valid":35,"n_invalid":0}

winner = max(q0_dist["frequencies"], key=q0_dist["frequencies"].get)
```

`_unwrap_structured(response_dict, question_id, schema_type)` peels
the per-panelist record down to the scalar/list the distribution
function expects — see the helper in
`src/althing/analysis/subgroup_cli.py`.

### Stable JSON output for agents

The `distribution_by_question` block (from `althing analyze`) and
the per-panelist `response` field (when `response_schema` is set) are
the schema-stable parts of the payload. They are:

- **Closed-shape**: one key per option (or per scale integer, or per
  taxonomy tag) plus type-specific summary stats.
- **Deterministic**: counts are computed from validated panelist
  responses, not from prose.
- **Versioned**: the v1.0.0 envelope owns the panel-result shape — see
  [docs/response-contract.md](response-contract.md) for the full
  reference.

Agents can compute summaries directly:

```python
# Pick-one / enum: winner + margin
def winner(freq: dict[str, int]) -> dict:
    sorted_counts = sorted(freq.items(), key=lambda kv: -kv[1])
    top, runner_up = sorted_counts[0], sorted_counts[1] if len(sorted_counts) > 1 else (None, 0)
    return {"choice": top[0], "votes": top[1], "margin": top[1] - runner_up[1]}

# Scale: mean is already in the distribution payload
def scale_summary(dist: dict) -> dict:
    return {"mean": dist["mean"], "mode": max(dist["frequencies"],
                                              key=dist["frequencies"].get),
            "n": dist["n"]}

# Tagged-themes (multi): rank tags by mentions
def tag_ranking(dist: dict) -> list[tuple[str, int]]:
    return sorted(dist["frequencies"].items(), key=lambda kv: -kv[1])
```

## CLI shortcut — there isn't one (yet)

There is no `althing poll run --choices A,B,C` wrapper command
today. The equivalent is a tiny instrument YAML — usually one block —
plus the standard `panel run`:

```bash
# (a) In-band structured output (recommended; zero extra LLM calls).
cat > /tmp/positioning.yaml <<'YAML'
instrument:
  version: 1
  questions:
    - text: "Which positioning wins? A) ... B) ... C) ... D) ..."
      response_schema:
        type: enum
        options: ["A", "B", "C", "D"]
YAML
althing panel run \
  --personas /tmp/product-research.yaml \
  --instrument /tmp/positioning.yaml \
  --output-format json --save

# (b) Free-text question + post-hoc extraction to a bundled schema.
cat > /tmp/positioning-text.yaml <<'YAML'
instrument:
  version: 1
  questions:
    - text: "Which positioning wins? A, B, C, or D? Briefly say why."
      response_schema:
        type: text
YAML
althing panel run \
  --personas /tmp/product-research.yaml \
  --instrument /tmp/positioning-text.yaml \
  --extract-schema pick_one \
  --output-format json --save
```

Option (a) does structured-output forcing in-band. Option (b) costs
one extra extraction call per panelist but keeps the free-text answer
in the same record under `response`, with the parsed pick under
`extraction.choice`.

From the Python SDK, the equivalent of (a) is `run_panel(instrument={
"version": 1, "questions": [{"text": ..., "response_schema": ...}]})`.
`quick_poll(question, ...)` is the one-question shortcut but does
**not** accept `response_schema` today — for structured output via
the SDK, use `run_panel` with a one-question `instrument` dict.

## MCP integration tips

- **Use `run_quick_poll` with `response_schema`** for one-shot bounded
  questions across a small panel. It accepts a JSON Schema dict
  directly and is the lowest-latency surface. Note that
  `response_schema` is **not supported in MCP sampling mode** (raw
  text is returned instead) — BYOK is required for structured output.
- **Use `run_panel` with `instrument`** for multi-question structured
  surveys. The `instrument` dict's per-question `response_schema`
  entries are honored exactly the same as the CLI's YAML.
- **Use `extract_schema=<name>`** to opt into one of the five bundled
  Pydantic-backed schemas (`ranking`, `likert`, `yes_no`, `pick_one`,
  `annotated_choice`) without authoring a JSON Schema yourself.
- **`ALTHING_DRIFT_DEGRADE=1`** in the MCP `env` block returns a
  degraded `panel_verdict` with `flags: [{"code": "schema_drift", ...}]`
  instead of a hard error when the 3-strike retry budget is exhausted.
  Recommended for agent workflows where partial signal beats no
  signal — see [docs/mcp.md#host-integration-flags](mcp.md#host-integration-flags).

## Why this is the agent default

1. **Schema beats prose.** A `frequencies` dict survives model
   refactors; a regex over "I think B is best because..." does not.
2. **Validation runs both sides.** The structured-output engine
   forces conforming output at generation time (tool-use forcing) AND
   validates the result on egress (the response gate, AC-9). A
   malformed panelist response triggers retry, not a silent parse
   error downstream.
3. **Cost is bounded.** Token budgets for bounded responses are
   tighter than free text — the engine sets sensible `max_tokens`
   defaults per `response_schema.type`.
4. **Distributions compose.** Multi-question instruments produce one
   distribution per question. Joining them is dict ops, not NLP.

If you're writing an agent that calls `run_panel` or `run_quick_poll`
and you find yourself reaching for a regex over
`result["rounds"][0]["results"][i]["response"]`, **stop and add a
`response_schema`**. That's almost always the right fix.

## References

- [docs/response-contract.md](response-contract.md) — the v1.0.0
  envelope and validation surface.
- [docs/cookbook/instrument-regression-testing.md](cookbook/instrument-regression-testing.md) —
  using bounded schemas to drive a CI regression check on instrument
  drift.
- [docs/convergence.md](convergence.md) — auto-stop based on
  bounded-question Jensen-Shannon divergence (only works with
  `enum`/`scale`/`yes_no`).
- `src/althing/structured/schemas.py` — bundled JSON Schemas.
- `src/althing/structured/models.py` — Pydantic mirrors.
- `src/althing/instrument.py` — `response_schema` validator.
- `src/althing/analysis/distribution.py` — deterministic per-
  question distributions used by `althing analyze`.
