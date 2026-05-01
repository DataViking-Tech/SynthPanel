# Composio Marketplace Submission Runbook

**Parent bead:** sp-2cw.3 · **Status:** Connector code committed; human-only submit steps below
**Last updated:** 2026-04-16

This is a checklist for listing SynthPanel in the Composio public tool
catalog. The connector code itself is already committed — see
[`src/synth_panel/integrations/composio.py`](../src/synth_panel/integrations/composio.py)
and [`examples/integrations/composio_langchain.py`](../examples/integrations/composio_langchain.py).
The remaining steps require GitHub PR authorship, Discord outreach, and
Composio-side review, so they must be completed by a maintainer.

## Why this connector exists

A single Composio listing puts SynthPanel in the native tool catalogs of
every major Composio-supported framework — LangChain, CrewAI, Semantic
Kernel, AutoGen, OpenAI Agents, and Google ADK — simultaneously. That's
roughly the same reach as shipping six framework-specific wrapper
packages, for one connector.

See [`docs/agent-integration-landscape.md`](agent-integration-landscape.md)
section C.3 for the full strategic rationale.

## What's already shipped

| Artifact | Path | Purpose |
|----------|------|---------|
| Toolkit factory | [`src/synth_panel/integrations/composio.py`](../src/synth_panel/integrations/composio.py) | Builds a Composio experimental `Toolkit` exposing 5 SynthPanel actions |
| LangChain example | [`examples/integrations/composio_langchain.py`](../examples/integrations/composio_langchain.py) | End-to-end LangChain agent using SynthPanel via Composio |
| CrewAI example | [`examples/integrations/composio_crewai.py`](../examples/integrations/composio_crewai.py) | End-to-end CrewAI agent using SynthPanel via Composio |
| Unit tests | [`tests/test_integrations_composio.py`](../tests/test_integrations_composio.py) | 11 tests — shape, delegation, import-guards |

The five registered actions are:

| Action slug | Wraps | Purpose |
|-------------|-------|---------|
| `SYNTHPANEL_QUICK_POLL` | `synth_panel.quick_poll` | One question → synthesized panel findings |
| `SYNTHPANEL_RUN_PANEL` | `synth_panel.run_panel` | Full panel run against instrument packs |
| `SYNTHPANEL_LIST_PERSONAS` | `synth_panel.list_personas` | Discover installed persona packs |
| `SYNTHPANEL_LIST_INSTRUMENTS` | `synth_panel.list_instruments` | Discover installed instrument packs |
| `SYNTHPANEL_GET_PANEL_RESULT` | `synth_panel.get_panel_result` | Load a saved panel result by id |

> Composio namespaces local toolkit actions as `LOCAL_<toolkit>_<action>`,
> so the agent-visible slugs are e.g. `LOCAL_SYNTHPANEL_QUICK_POLL`.

## Canonical pitch (use verbatim in forms)

**One-liner (≤120 chars):**

> Run synthetic focus groups against AI persona panels — quick_poll, multi-round instruments, saved results.

**Standard description (≤300 chars):**

> SynthPanel exposes five research actions to your agent: quick_poll
> (one question → synthesized panel findings), run_panel (multi-round
> instruments with v3 branching), and persona/instrument/result
> discovery. Runs locally via the SynthPanel SDK. Any LLM.

**Category / tags:**

- Research · Market research · Survey · Synthetic respondents
- Python · Local · Agent · LLM

**Auth model:** No Composio-hosted auth. The connector runs in-process;
SynthPanel reads the caller's `ANTHROPIC_API_KEY` (or OpenAI, Gemini,
xAI) from the environment. Document this in the Composio listing's
"auth" section.

**Links:**

- Site: <https://synthpanel.dev>
- Repo: <https://github.com/DataViking-Tech/SynthPanel>
- PyPI: <https://pypi.org/project/synthpanel/>
- Docs: <https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md>

## Submission paths (two possible; try in order)

Composio's marketplace has two entry points for third-party tools, and
we don't yet know which one the Composio team prefers for a
locally-executing toolkit like ours. Start with path A; fall back to
path B if asked.

### Path A — PR against `ComposioHQ/composio` (preferred)

1. Fork <https://github.com/ComposioHQ/composio>.
2. Read [`CONTRIBUTING.md`](https://github.com/ComposioHQ/composio/blob/next/CONTRIBUTING.md)
   end to end. Note the directory conventions — for *providers* they use
   `packages/providers/`, but SynthPanel is a **tool**, not a provider,
   so the target directory may differ. Open the PR as a draft and ask
   Composio maintainers where it belongs.
3. Add a connector manifest pointing at the published PyPI package
   (`synthpanel`) and the toolkit factory
   (`synth_panel.integrations.composio:synthpanel_toolkit`). Exact
   manifest shape is Composio-defined; copy the nearest existing
   example in their repo for a locally-executing Python tool.
4. Include a changeset via `pnpm changeset`.
5. Link the examples in [`examples/integrations/`](../examples/integrations/)
   as proof of working LangChain + CrewAI end-to-end runs.
6. Record the PR URL below.

**Recorded PR:** `__ TBD — fill after opening __`

### Path B — Discord + "Request a Toolkit" form

If path A is the wrong venue, the Composio team routes third-party
toolkits through Discord first:

1. Join <https://discord.gg/composio>.
2. Post in `#integrations-request` (or the channel linked from their
   contribution docs) with:
   - The canonical pitch above
   - Links to the committed connector code and examples
   - A note that the toolkit runs locally and requires no Composio-side
     auth
3. They typically reply with a Notion/Airtable form for marketplace
   listing. Fill it; paste the canonical pitch verbatim.
4. Record the resulting listing URL below.

**Recorded listing URL:** `__ TBD — fill after approval __`
**Composio toolkit slug:** `SYNTHPANEL`

## Verification (do once the listing is live)

1. Fresh venv, install only `composio composio_langchain langchain
   langchain-anthropic synthpanel`.
2. Export `ANTHROPIC_API_KEY` and `COMPOSIO_API_KEY`.
3. Run [`composio_langchain.py`](../examples/integrations/composio_langchain.py);
   confirm the agent successfully calls `quick_poll` and returns a
   synthesis recommendation.
4. Repeat with [`composio_crewai.py`](../examples/integrations/composio_crewai.py).
5. In the Composio web UI, confirm `SYNTHPANEL` appears in the catalog
   and the action descriptions are legible.

## Post-launch

- Add a link to the live Composio listing from
  [`docs/agent-integration-landscape.md`](agent-integration-landscape.md)
  (section C.3) and the repo `README.md` integrations table.
- Cross-post in the SynthPanel announcements (blog + social).
- Close bead `sp-2cw.3` with the recorded URLs in the close reason.

## Troubleshooting upstream Composio drift

The integration binds to Composio's experimental Toolkit API surface
(`composio_client.experimental.Toolkit` + the `@toolkit.tool()`
decorator). That surface is still labelled experimental upstream, so we
pin the dependency tightly in [`pyproject.toml`](../pyproject.toml)
(currently `composio>=0.5,<0.6`). When Composio cuts a new major or
minor release, the contract this integration relies on may change
without an in-band signal — users will only notice when the toolkit
fails to construct or actions don't appear in their agent's catalog.

Use this checklist when triaging a suspected upstream-shape drift:

| Symptom | Likely cause | Where to check |
|---------|--------------|----------------|
| `RuntimeError: experimental.Toolkit` at startup | Composio renamed or moved the experimental Toolkit factory | [Composio changelog](https://github.com/ComposioHQ/composio/releases) — search for "experimental" |
| `TypeError` from `experimental.Toolkit(...)` | Constructor signature changed (new required kwarg, removed kwarg) | Composio release notes for the version installed in your venv |
| Tool schemas missing or malformed in the agent catalog | `@toolkit.tool()` decorator semantics changed | Composio docs for "custom toolkits" |
| `tests/test_integrations_composio.py::TestUpstreamShapeCanary` fails after a Composio bump | The end-to-end translation contract has drifted | Re-read the failing assertion; update the stub in the test file to mirror the new shape, then update [`composio.py`](../src/synth_panel/integrations/composio.py) |
| Pydantic input validation errors at agent call time | Composio is passing a different argument shape to tool functions | Composio's tool-router source for the version installed |

If you confirm an upstream change is intentional and behaves correctly:

1. Update the matching stub in
   [`tests/test_integrations_composio.py`](../tests/test_integrations_composio.py)
   (the `_StubToolkit` / `_StubExperimental` classes) so the canary
   passes against the new contract.
2. Bump the upper bound in
   [`pyproject.toml`](../pyproject.toml) (`[project.optional-dependencies].composio`).
3. Note the verified version in this doc and re-run the integration
   tests in
   [`examples/integrations/`](../examples/integrations/) against the
   new release.

If you cannot reproduce the failure with the pinned version listed in
`pyproject.toml`, ask your user which Composio version they have
installed (`pip show composio`) — they may have overridden the pin in
their own environment.

## Why the connector code lives inside SynthPanel

The alternative would be a standalone `composio-synthpanel` package.
We chose inline for three reasons:

1. The factory is ~300 lines — the version-skew risk of a separate
   package outweighs its isolation benefit.
2. `composio` is a soft import; users who don't use Composio don't pay
   any install cost (the adapter is guarded and gracefully raises a
   `ComposioNotInstalledError` with the right `pip install` command).
3. The SDK surface it wraps (`quick_poll`, `run_panel`, …) lives in the
   same repo, so both move together through the normal release cycle.

If Composio prefers third-party toolkits live in Composio-managed
repositories, we can lift the module verbatim into a
`composio-synthpanel` package at that time; nothing in the current
design prevents the move.
