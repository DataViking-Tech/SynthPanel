---
name: synthpanel
description: Director skill for SynthPanel — decide when to reach for synthetic panels (focus groups, name tests, concept tests, pricing probes, mass-appeal polls) and route to the right sub-skill, with anti-overclaiming guardrails baked in.
version: 1.0.0
author: SynthPanel maintainers
license: MIT
allowed-tools:
  - mcp__synth_panel__run_prompt
  - mcp__synth_panel__run_quick_poll
  - mcp__synth_panel__run_panel
  - mcp__synth_panel__extend_panel
  - mcp__synth_panel__list_persona_packs
  - mcp__synth_panel__get_persona_pack
  - mcp__synth_panel__save_persona_pack
  - mcp__synth_panel__list_instrument_packs
  - mcp__synth_panel__get_instrument_pack
  - mcp__synth_panel__save_instrument_pack
  - mcp__synth_panel__list_panel_results
  - mcp__synth_panel__get_panel_result
metadata:
  hermes:
    tags:
      - research
      - synthetic-respondents
      - focus-group
      - panels
      - mcp
      - market-research
    related_skills:
      - focus-group
      - name-test
      - concept-test
      - survey-prescreen
      - pricing-probe
    config:
      - key: synthpanel.preferred_interface
        description: How to call SynthPanel when both are available (mcp, cli).
        default: mcp
        prompt: Prefer the synth_panel MCP server, or the synthpanel CLI?
      - key: synthpanel.default_panel_size
        description: Persona count for routine polls and concept tests.
        default: "5"
        prompt: Default persona count (3–6 is normal; raise for mass-appeal polls)
      - key: synthpanel.default_model_alias
        description: LLM alias for routine runs (sonnet, haiku, gemini, grok, ...).
        default: haiku
        prompt: Default LLM alias for routine synthetic runs
required_environment_variables:
  - name: ANTHROPIC_API_KEY
    prompt: Anthropic API key (default provider for SynthPanel)
    help: Needed for Claude-family models. Skip only if exclusively using another provider.
    required_for:
      - default-claude-runs
  - name: OPENAI_API_KEY
    prompt: OpenAI API key (optional; only if you ask SynthPanel to use OpenAI models)
    help: Required only when --model selects an OpenAI or OpenAI-compatible model.
    required_for:
      - openai-runs
  - name: GOOGLE_API_KEY
    prompt: Google API key (optional; only for Gemini models)
    help: Required only when --model selects a gemini-* model.
    required_for:
      - gemini-runs
  - name: XAI_API_KEY
    prompt: xAI API key (optional; only for Grok models)
    help: Required only when --model selects a grok-* model.
    required_for:
      - grok-runs
---

# SynthPanel — synthetic respondent research

## When to use

Reach for this skill the moment a user asks for any of the following — these
are the canonical triggers:

- "run a synthetic panel" / "run a focus group" / "synthetic focus group"
- "test positioning" / "test this value prop" / "concept test"
- "compare product names" / "name test" / "which name is better"
- "pricing probe" / "what would people pay" / "price sensitivity"
- "poll 30 personas" / "poll the panel" / "mass-appeal poll"
- "prioritization poll" / "rank these options across personas"
- "pre-screen this survey" / "stress-test these questions"

It is also the right skill when a user wants directional, fast, **synthetic**
input *before* spending budget on real participants — e.g. "I have a hunch,
should I bother running this past anyone?" or "what would land vs. what would
fall flat?"

**Do not** use this skill when the user needs decisions backed by real human
data (regulatory, statistically powered, claims-grade). Say so explicitly and
recommend a real panel instead — synthetic output is preflight, not validation.

## Quick reference

SynthPanel exposes two interfaces. They are equivalent; prefer the MCP path
when configured (`synthpanel.preferred_interface = mcp`).

### MCP tools (preferred when `synth_panel` MCP server is registered)

| Tool | Purpose |
| --- | --- |
| `mcp__synth_panel__run_prompt` | One-shot prompt against one persona (smoke test). |
| `mcp__synth_panel__run_quick_poll` | One-question poll across N personas; returns vote tallies + synthesis. |
| `mcp__synth_panel__run_panel` | Full panel with persona pack + instrument pack (v1, v2, v3 branching). |
| `mcp__synth_panel__extend_panel` | Append one ad-hoc round on top of a saved result. Not v3 re-entry. |
| `mcp__synth_panel__list_persona_packs` / `get_persona_pack` / `save_persona_pack` | Manage persona libraries. |
| `mcp__synth_panel__list_instrument_packs` / `get_instrument_pack` / `save_instrument_pack` | Manage survey instruments (v1/v2/v3). |
| `mcp__synth_panel__list_panel_results` / `get_panel_result` | Retrieve saved runs by ID. |

### CLI (fallback or scripted use)

```bash
synthpanel prompt "Say hello"                                # single prompt
synthpanel panel run \                                       # full panel
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml
synthpanel panel run --personas examples/personas.yaml \     # branching v3
  --instrument pricing-discovery
synthpanel instruments list                                  # discover bundled packs
synthpanel mcp-serve                                         # start MCP server (stdio)
synthpanel mcp install                                       # register MCP server in the active editor
```

Provider env vars map: `ANTHROPIC_API_KEY` (default), `OPENAI_API_KEY`,
`GOOGLE_API_KEY`/`GEMINI_API_KEY`, `XAI_API_KEY`. Pick the model with
`--model sonnet|haiku|gpt-4o|gemini|grok-3|...`.

### Sub-skill routing

For task-shaped work, delegate to the matching sub-skill rather than
re-deriving the workflow here. They live alongside this skill:

| Trigger | Sub-skill |
| --- | --- |
| Open-ended exploration with personas | `focus-group` |
| 1–3 candidate names compared head-to-head | `name-test` |
| "Does this value-prop land?" / problem validation | `concept-test` |
| "Pre-screen this survey before fielding it" | `survey-prescreen` |
| Pricing sensitivity / willingness-to-pay | `pricing-probe` |
| One-question poll, no follow-ups | `run_quick_poll` MCP tool directly |

## Procedure

1. **Confirm the question is synthetic-appropriate.** If the user needs
   audited, real-human evidence, stop and say so. Otherwise frame what you
   are about to do as a synthetic preflight.

2. **Pick the interface.** Check whether the `synth_panel` MCP server is
   registered (tools prefixed `mcp__synth_panel__*` exist). If yes, use MCP.
   If not, fall back to the CLI. Tell the user which one you used.

3. **Pick the sub-skill** from the routing table above. If none of the
   sub-skills fits, run an ad-hoc `run_quick_poll` (single question,
   `panel_size` personas) or a `run_panel` with an inline instrument.

4. **Dry-run first when feasible.** Use a small persona pack (3–5) and the
   default fast model (`haiku` / `gemini-flash` tier) to validate the
   instrument before scaling up. Confirm the questions read clearly and the
   schemas parse before spending on a 30-persona pass.

5. **Force structured output where it matters.** Use the structured /
   schema-backed response path (Pydantic-style schemas via tool-use forcing)
   for any question whose downstream use requires aggregation: vote counts,
   weighted scores, segment splits, ranked lists. Free-form text is fine for
   discovery but useless for tallies.

6. **Save results and capture the result ID.** Both `run_panel` and
   `run_quick_poll` persist runs; surface the result ID in your reply so the
   user can `get_panel_result` later or `extend_panel` for one more round.

7. **Synthesize with discipline.** Report:
   - Vote tallies (counts and percentages) for closed-form items.
   - Weighted scores or rank order for prioritization tasks.
   - Segment splits when personas span clearly distinct groups.
   - Top objections, surfaced verbatim from at least one persona.
   - A confidence note that calls out sample size and synthetic provenance.

8. **Apply skeptical / anti-sycophancy prompting.** When designing the
   instrument:
   - Ask "what would make you *not* buy / use / recommend this?" alongside
     positive framings.
   - Force at least one persona-disagreement probe ("which of you would push
     back on the loudest answer here?").
   - Use negative-control questions for sanity (e.g. an obviously bad
     concept) when calibrating a panel for the first time.
   - Prefer instruments with `route_when` branches that catch lukewarm
     responses and probe the friction, rather than letting personas drift
     into agreement.

9. **Frame the output as synthetic preflight, not market validation.** Every
   reply that contains panel results should include a one-line caveat: *this
   is synthetic input from N AI personas using <model>; treat as a hypothesis
   generator, not as evidence about real users.*

## Pitfalls

- **Overclaiming.** Synthetic personas are not respondents. Never report
  results as "users said …" — they didn't. Say "synthetic personas modeled
  on <demographic> tended to …" and include sample size.
- **Sycophancy drift.** Without skeptical prompting and disagreement probes,
  panels of LLM-derived personas converge on whatever the question implies.
  Bake in pushback questions; check that at least one persona disagrees on
  every important item.
- **Persona homogeneity.** Three personas that all look like the model's
  default voice are worse than one well-shaped persona. Audit persona packs
  before reuse — diverse `personality_traits` and explicit `background`
  text matter.
- **Treating `extend_panel` as branching.** `extend_panel` appends exactly
  one ad-hoc round on a saved result; it does **not** re-enter a v3
  instrument's `route_when` DAG. If you want adaptive branching, run a fresh
  `run_panel` against a v3 instrument instead.
- **Cost surprises.** Large panels with capable models add up fast. Default
  to the configured fast model alias (`synthpanel.default_model_alias`,
  haiku by default); upgrade only when the dry-run signal warrants it.
- **Theme-matching gotcha on v3 instruments.** `route_when` predicates
  compare against the *exact* theme strings the synthesizer emits. Prefix v3
  instruments with a comment block listing canonical theme tags so routes
  don't silently fall through to `else`.

## Verification

- The user sees an explicit interface note ("used the `synth_panel` MCP
  server" or "used the `synthpanel` CLI").
- The reply includes structured numbers — counts, percentages, weighted
  scores, or rank order — for any aggregation question.
- A saved result ID is surfaced (e.g. `panel_result_id: 01h…`) so the run
  is recoverable.
- A synthetic-preflight caveat appears in the same reply as the results.
- At least one pushback / counter-argument appears in the synthesis, not
  just consensus.
- For pricing or naming work, an explicit recommendation against acting on
  synthetic-only signal for high-stakes decisions appears.

## See also

- [`focus-group`](../focus-group/SKILL.md) — open-ended qualitative panels.
- [`name-test`](../name-test/SKILL.md) — head-to-head naming.
- [`concept-test`](../concept-test/SKILL.md) — value-prop validation.
- [`survey-prescreen`](../survey-prescreen/SKILL.md) — instrument pre-flight.
- [`pricing-probe`](../pricing-probe/SKILL.md) — pricing sensitivity.
- [Agent Skills Discovery index](/.well-known/agent-skills/index.json) on
  synthpanel.dev — the published index that lists all SynthPanel skills.
- [`docs/agent-skills.md`](../../docs/agent-skills.md) — install paths
  (Claude Code plugin, manual copy, other hosts).
- [`docs/mcp.md`](../../docs/mcp.md) — MCP server reference, sampling mode.
