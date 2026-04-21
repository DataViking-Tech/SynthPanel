# Multi-Model: `--models`, ensemble, and blend

The `--models` flag has **two distinct meanings** depending on its shape.
This page documents which shape does what, how weights map to persona
assignments, and the edge cases operators have tripped on.

## Two shapes of `--models`

### 1. Weighted per-persona assignment (has `:`)

```bash
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml \
  --models 'haiku:0.5,gemini-2.5-flash:0.5'
```

Splits your panel across models in the given ratio. With 6 personas and
`haiku:0.5,gemini:0.5`, you get 3 personas on haiku, 3 on gemini. Each
persona answers once, on the model they were assigned.

Useful for:

- Running a cross-provider panel cheaply (haiku + flash instead of all
  sonnet).
- A/B comparing provider behavior within a single panel run.

### 2. Ensemble (no `:`)

```bash
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml \
  --models 'haiku,sonnet,gemini-2.5-flash'
```

Runs the **full panel once per model**. With 6 personas and 3 models,
every persona answers on every model — 18 sessions total. Use with
`--blend` to weight-average the resulting distributions.

Useful for:

- Measuring cross-model agreement/disagreement on the same panel.
- Building weighted ensembles (`--blend`) for decision support.

The two shapes are mutually exclusive with `--model` but are chosen
implicitly by whether any entry in the spec contains `:`.

## Weighted assignment: the algorithm

Given a spec like `a:0.5,b:0.5` and N personas, the algorithm is:

1. **YAML overrides first.** Any persona whose YAML has a `model:` field
   keeps that model and is removed from the assignment pool. This always
   wins over `--models`.
2. **Normalize.** Sum the weights, divide each by the total. Absolute
   magnitudes don't matter — `a:2,b:3`, `a:0.4,b:0.6`, and `a:40,b:60`
   all produce the same split.
3. **Split proportionally.** For each model *except the last*, allocate
   `round(weight_normalized * pool_size)` personas. The last model
   absorbs whatever's left. This guarantees totals always sum exactly to
   the pool size, with no rounding gap.
4. **Fill in order.** Personas are consumed from the pool in the order
   they appear in the YAML (minus YAML-override ones). Model order
   matches the `--models` spec.

The algorithm is **fully deterministic**: same personas + same spec →
same assignment, every run. No RNG, no seed.

## Edge cases

### Non-even division

With 7 personas and three equal models (`a:0.33,b:0.33,c:0.33`):

```
round(0.33 * 7) = 2  → a gets 2
round(0.33 * 7) = 2  → b gets 2
remainder             → c gets 3
```

**Swapping model order changes who gets the extra.** If you care about
balance, put the model you want to over-sample last.

### Weights that don't sum to 1.0

Weights are normalized, so `a:2,b:3` works and means "40% / 60%". But a
sum far from 1.0 often means a typo (`0.3,0.3,0.3` intended as
`0.33,0.33,0.33`). The CLI prints a warning on sums outside
`1.0 ± 0.02`:

```
Warning: --models weights sum to 0.900, not 1.0. Weights will be
normalized, preserving the ratio.
```

The run still proceeds — this is a soft warning, not a hard error.

### Zero and negative weights

Rejected at parse time with an error. A model with weight 0 couldn't
receive any personas anyway, and negative weights don't have a sensible
meaning.

### YAML `model:` override

If a persona has a `model:` field in its YAML, that model is used for
that persona regardless of `--models`. Example:

```yaml
personas:
  - name: Alice
    model: claude-opus-4-7        # always Opus, ignores --models
  - name: Bob
    # no model field → assigned by --models
```

This is also the only way to set per-persona models via the MCP server
(`persona_models` argument on `run_panel`).

### Weights vs. ensemble — how the CLI tells them apart

A spec with **any** `:` is weighted. A spec with **no** `:` is
ensemble. Mixing shapes (`haiku,sonnet:0.5`) is a parse error — there's
no coherent meaning.

## Seeing the assignment

The CLI prints the resolved persona → model map to stderr before any
LLM calls:

```
Model assignment:
  Maya Chen        → haiku
  Derek Washington → gpt-mini
  ...
Totals: gemini-flash=2, gpt-mini=2, haiku=2
```

If the split looks wrong (rounding on a small panel, a YAML override
you forgot about), `^C` and re-run.

The same map is available in JSON output under `model_assignment`:

```json
{
  "model_assignment": {
    "Maya Chen": "haiku",
    "Derek Washington": "gpt-mini"
  }
}
```

## MCP equivalent

The MCP `run_panel` tool exposes this behavior through two separate
parameters:

- `persona_models: dict[str, str]` — pre-computed persona→model map.
  The MCP client is expected to do the weighted split itself if it
  wants one; MCP accepts the final map directly, which avoids rounding
  ambiguity across clients.
- `models: list[str]` — ensemble mode only. Every persona answers every
  model.

There is no weighted-spec parser on the MCP surface — it's a CLI
convenience, not a protocol concern.
