# Reproducibility — what synthpanel can and can't promise

LLM-driven research lives or dies on whether you can rerun a panel and get
the same answers. This page lays out the two reproducibility tools in
synthpanel, what each one actually guarantees, and where the joints are.

## TL;DR

| Goal | Use | Determinism |
|---|---|---|
| Replay a *previously executed* run exactly | `panel run --resume <run-id>` | Strong: cached responses are served verbatim |
| Constrain a *new* run for repeatability | `panel run --seed N` | Best-effort, provider-dependent |

These solve different problems. `--resume` lets you pick up where a crashed
or interrupted run left off, with byte-identical answers for every
already-completed panelist. `--seed` lets you start a fresh run that you
*want* to be repeatable — useful for instrument validation, paper
supplementary materials, or debugging an unexpected synthesis output.

## `--seed` — provider-aware sampling

```bash
synthpanel panel run --seed 42 --personas p.yaml --instrument s.yaml
```

The seed is forwarded to the provider's sampling parameter on every
panelist call (and every synthesis call) for the duration of the run.

### Provider support

| Provider | `--seed` behavior |
|---|---|
| OpenAI / OpenRouter | Forwarded as `seed` on the chat-completions request |
| Gemini (Google) | Forwarded as `seed` on the OpenAI-compatible endpoint |
| xAI (Grok) | Forwarded as `seed` |
| Anthropic (Claude) | **Not supported.** synthpanel logs a single warning per provider per run and proceeds without determinism |
| Local / unknown OpenAI-compatible | Forwarded; whether it's honored depends on the runtime |

### Why Claude isn't reproducible via `--seed`

Anthropic's Messages API has no `seed` parameter at the time of writing.
Setting one would mislead users into thinking determinism holds. Instead,
synthpanel:

1. Logs **one** warning per provider per run, naming the provider:
   ```
   --seed=42 is not supported by Anthropic models; runs against this
   provider will not be deterministic. Use temperature=0 for closer-to-
   deterministic output.
   ```
2. Drops the seed from the Anthropic request body (so the API doesn't
   reject the call).
3. Keeps the seed on every other provider's request as normal.

For Claude, the closest you can get to reproducibility is
`--temperature 0`, but expect drift when Anthropic ships model updates.

### Mixed-provider runs

`--models claude-sonnet-4-6:0.5,gpt-4o-mini:0.5` (or per-persona model
overrides) splits the panel across providers. With `--seed N`:

- The OpenAI-routed panelists get `seed=N`.
- The Anthropic-routed panelists do not.
- You see exactly **one** "seed not supported" warning for Anthropic,
  and exactly zero for OpenAI.

The warning is one-shot per (LLMClient instance, provider) so a 1000-
panelist mixed run does not produce 500 duplicate warnings.

### What gets recorded

When `--seed` is set, the value lands in two places in the output JSON:

- `metadata.parameters.seed` — surfaced for downstream auditors.
- The checkpoint's `config` fingerprint — so `--resume` of a checkpointed
  run rejects mismatched seeds with a `CheckpointDriftError` rather than
  silently splicing two different seeded slices together.

### What `--seed` does **not** promise

- **Cross-version stability.** A new model release on the provider side
  will change outputs even with the same seed. The seed only constrains
  sampling *given a fixed model*.
- **Cross-provider stability.** Two providers with the same seed will
  emit different text. Seeds are local to one provider's RNG.
- **Determinism on every single token.** Many providers describe seeded
  sampling as best-effort; backend load balancing, batching, and
  numerical drift can still nudge outputs. Treat `--seed` as "very
  similar most of the time," not "byte-identical always."

## `--resume` — replay an existing run

```bash
synthpanel panel run --resume <run-id>
```

`--resume` operates on a checkpoint written by a prior `panel run` (with
`--checkpoint-dir`, or via the default checkpoint root). Every panelist
that completed before the crash / SIGINT is reloaded from disk; only the
remaining panelists are dispatched to the provider.

This **is** byte-deterministic for the already-completed slice — the
responses are served from the checkpoint, not regenerated. Combined with
`--seed N` on the resumed slice, you get the closest synthpanel offers
to a fully deterministic run on supporting providers.

### Drift detection

`--resume` rejects a run when the saved fingerprint disagrees with the
current invocation on any of:

- persona names / count / identities
- question texts / count
- model (or `persona_models` map)
- temperature, top_p, **seed**
- response/extract schemas
- prompt template variables

Pass `--allow-drift` to override (with a loud warning that the
resumed run is statistically inconsistent).

## Best practices

- **Always print or capture the seed** if you intend to publish or audit
  the run. It is in the JSON output but easy to miss in text mode.
- **Pin the model** (`--model claude-sonnet-4-6` rather than `--model sonnet`).
  Aliases resolve to whatever the latest version is, which defeats the
  purpose of seeding.
- **Pair with `--temperature 0`** on Anthropic if you need
  near-deterministic Claude output. It's not a true seed, but it removes
  most sampling variance.
- **Use `--resume` for replay, `--seed` for repeatability.** They're
  complementary, not interchangeable.
