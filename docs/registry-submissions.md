# Registry Submission Runbook

**Parent beads:** sp-ege.7 (initial prep), sp-fiv (Smithery + Glama sweep)
**Status:** Prep artifacts committed; human-only submit steps below
**Last updated:** 2026-04-19

This is a checklist for submitting SynthPanel to major MCP registries. The
code-doable prep work is already done (`glama.json`, `server.json`); the
remaining steps require interactive auth or web forms and must be completed
by a maintainer.

## Canonical pitch (use verbatim)

**One-liner (≤120 chars):**

> Open-source synthetic focus groups. Any LLM. Your terminal or your agent's tool call.

**Standard description (≤300 chars):**

> Run synthetic focus groups using AI personas. 12 MCP tools for single
> prompts, full panel runs, and v3 branching (adaptive) instruments across
> any LLM provider (Claude, OpenAI, Gemini, xAI). MIT licensed, pip-install.

**Long description (for forms with room):**

> SynthPanel is a lightweight, LLM-agnostic research harness for running
> synthetic focus groups using AI personas. Define personas in YAML, define
> your research instrument in YAML, and run against any LLM — from your
> terminal, from a pipeline, or from an AI agent's MCP tool call. Includes
> a 12-tool MCP server for Claude Code / Cursor / Windsurf. MIT licensed.
> Benchmark score: SPS 0.90 on SynthBench. pip install synthpanel[mcp].

**Category / tags (where supported):**

- Research · Market research · Survey · Synthetic respondents
- MCP · model-context-protocol · Agent · Claude · Cursor · Windsurf
- Python · PyPI · LLM · AI personas

**Links:**

- Site: <https://synthpanel.dev>
- Repo: <https://github.com/DataViking-Tech/SynthPanel>
- PyPI: <https://pypi.org/project/synthpanel/>
- MCP docs: <https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md>
- Benchmark: <https://synthbench.org>

---

## 1. Glama.ai — `glama.ai/mcp/servers`

**Status:** Prep committed · Claim flow required

Glama auto-indexes GitHub repos with the `mcp` topic; SynthPanel's repo
already has the topic set, so the listing should appear on its own. To
claim ownership (required to edit the listing, configure a Docker image,
see usage reports), we've committed `glama.json` to the repo root with
both org admins listed as maintainers.

**Human steps (one-time, ~5 min):**

1. Sign in at <https://glama.ai> with GitHub (as `the-data-viking` or
   `openclaw-dv`). Either admin works since both are in `glama.json`.
2. Search for `synthpanel` in the registry. If the server appears:
   visit its page → **Claim server** → follow the GitHub OAuth flow.
3. If the server does **not** appear yet, use the **Add Server** button
   at <https://glama.ai/mcp/servers> and paste
   `https://github.com/DataViking-Tech/SynthPanel`. Glama will index
   within minutes and the claim flow becomes available.
4. Once claimed, update the server metadata:
   - **Title:** SynthPanel
   - **Tagline:** (use the one-liner above)
   - **Description:** (use the standard description above)
5. Record the live URL below.

**Recorded URL:** `__ TBD — fill after claim __`

**Reference:** <https://glama.ai/blog/2025-07-08-what-is-glamajson>

---

## 2. Smithery — `smithery.ai`

**Status:** No repo-committed prep applicable · CLI publish required

Smithery is the other major MCP registry. Unlike Glama, Smithery does **not**
auto-index GitHub repos by topic, and there is no equivalent of `glama.json`
that we can commit to claim ownership from the repo side. Submissions happen
either through the `smithery` CLI (authenticated via Smithery API key) or
through Smithery's web UI.

A `smithery.yaml` manifest is useful only for Smithery's hosted/bundled
deployment modes (JS module upload, MCPB bundle, container build). SynthPanel
ships as a PyPI package invoked via `uvx`, so the appropriate submission path
is **URL-based** — point Smithery at the GitHub repo; it will scan and index
the server metadata (tools, README, install instructions).

**Human steps (one-time, ~10 min):**

1. Sign in at <https://smithery.ai> with GitHub (as `the-data-viking` or
   `openclaw-dv`).
2. Generate a Smithery API key from the account settings page and export it:
   ```bash
   export SMITHERY_API_KEY=sk_...
   ```
3. Install and run the Smithery CLI to publish the repo URL:
   ```bash
   npx @smithery/cli mcp publish \
     https://github.com/DataViking-Tech/SynthPanel \
     -n synthpanel
   ```
   The CLI registers SynthPanel under the qualified name `synthpanel` and
   scans the linked repo for metadata. If the CLI path fails, fall back
   to the web submission form at <https://smithery.ai/new>.
4. Edit the listing via the Smithery dashboard to set:
   - **Title:** SynthPanel
   - **Tagline:** (use the one-liner above)
   - **Description:** (use the standard description above)
   - **Tags:** research, survey, personas, llm, mcp, python
5. Record the live URL below.

**Recorded URL:** `__ TBD — fill after publish __`

**If Smithery requires a deployable bundle (MCPB / container) rather than a
URL-only listing:** this escalates from a ~10 min submission into a packaging
project. File a separate bead (`Package SynthPanel as MCPB bundle for
Smithery hosted deploy`) before attempting — do not improvise the packaging
inside this submission sweep.

**Reference:** <https://smithery.ai/docs/build/publish>

---

## 3. Official MCP Registry — `registry.modelcontextprotocol.io`

**Status:** Prep committed · Publish required (upstream of PulseMCP)

This is the **canonical upstream registry** run by Anthropic. PulseMCP
auto-ingests from it daily, so a single publish here yields both
listings. The `server.json` manifest is already committed at the repo
root and describes:

- PyPI package `synthpanel`, version 0.9.1
- `runtimeHint: uvx` → clients run `uvx synthpanel[mcp] synthpanel mcp-serve`
- stdio transport
- All four optional API-key env vars

**Human steps (one-time, ~10 min):**

1. Install the publisher CLI. Via Homebrew:
   ```bash
   brew install mcp-publisher
   ```
   Or download a release binary from
   <https://github.com/modelcontextprotocol/registry/releases>.
2. From the repo root, authenticate with GitHub (namespace ownership
   proof for `io.github.DataViking-Tech/*`):
   ```bash
   mcp-publisher login github
   ```
3. Publish:
   ```bash
   mcp-publisher publish
   ```
   The CLI reads `./server.json` automatically.
4. Verify at <https://registry.modelcontextprotocol.io/v0/servers?search=synthpanel>.
5. Record the listing URL below.

**Recorded URL:** `__ TBD — fill after publish __`

**Re-publishing for future versions:** bump the `version` field in both
`server.json` and `pyproject.toml` (keep them in lockstep), publish to
PyPI, then `mcp-publisher publish` again.

**Reference:** <https://github.com/modelcontextprotocol/registry>

---

## 4. PulseMCP — `pulsemcp.com/submit`

**Status:** Downstream of Official MCP Registry — should auto-populate

PulseMCP ingests entries from the Official MCP Registry daily and
re-processes weekly. If step 2 above succeeds, a PulseMCP listing
should appear within 7 days with no additional action.

**Human steps (only if the auto-ingest doesn't pick it up):**

1. Wait 7–10 days after the Official MCP Registry publish.
2. Search <https://www.pulsemcp.com/servers> for `synthpanel`.
3. If missing, submit manually at <https://www.pulsemcp.com/submit>:
   - **Type:** Server
   - **URL:** `https://github.com/DataViking-Tech/SynthPanel`
4. For metadata adjustments, email `[email protected]`.
5. Record the URL below.

**Recorded URL:** `__ TBD — verify 7–10 days after Official Registry publish __`

---

## 5. MCPHub — `mcphub.com`

**Status:** Submission mechanism not publicly documented · Best-effort

MCPHub (mcphub.com) is a curated search/discovery site. Unlike Glama
and PulseMCP, their submission process is not advertised on the
landing page. The site was unreachable during research (timed out on
the crawl), so verification of the submission path requires a
manual visit.

**Human steps (exploratory, ~10 min):**

1. Visit <https://mcphub.com> and look for a "Submit Server" or
   "Add Server" link in the nav or footer.
2. If a submission form exists, fill it using the canonical pitch above.
3. If no form exists, email or DM the maintainers (look for contact
   info in the site footer or GitHub org).
4. Record the URL or close as "no submission path available".

**Recorded URL:** `__ TBD — investigate __`

**Alternative:** there is a separate project `mcphub.io` (note: `.io`)
that appears to be a different curator; worth checking both domains.

---

## 6. Klavis AI — NOT a registry (out of scope)

**Status:** Deferred · Out of scope for a "submission sweep"

Despite its 300+ "MCP Servers" claim, Klavis is **not a registry** in
the sense of a listing catalog. Klavis's
[`MCP_SERVER_GUIDE.md`](https://github.com/Klavis-AI/klavis/blob/main/MCP_SERVER_GUIDE.md)
requires servers to be integrated **into their monorepo** at
`mcp_servers/<name>/` with:

- Full server source code (they host it on their infra, not us)
- Screenshots or video proving tool functionality
- Per-tool documentation and prompt words
- Ongoing maintenance inside their repo

This is a fork-and-integrate engineering project, not a one-form
submission.

(Historical note: the original sp-ege.7 bead grouped Smithery with Klavis
under "skip unless Python hosting works." sp-fiv revisited Smithery and
confirmed a URL-based publish path exists — see Section 2 above. Klavis
remains out of scope.)

**Recommendation:** skip for this submission sweep. If Klavis listing
becomes strategically important, file a separate bead scoped as
"Evaluate + implement Klavis integration for SynthPanel" (estimated:
days, not hours).

---

## Post-submission: LLM retrieval test

The bead's acceptance criterion **"aggregate cross-registry presence
confirmed via LLM retrieval test"** should be run **4–6 weeks** after
submissions go live, to allow indexing + propagation. Test queries:

- `"is there an open-source MCP server for synthetic focus groups?"`
- `"mcp tool for synthetic survey respondents"`
- `"python mcp server for user research"`

Run these on ChatGPT, Claude, Perplexity, and Gemini. Score
SynthPanel's surface: cited / mentioned / missing. File as a separate
bead (the test itself is deliverable-producing work).

---

## Checklist summary

| Registry | Prep | Human step | Recorded URL |
|---|---|---|---|
| Glama.ai | ✓ `glama.json` | Claim flow (GitHub OAuth) | TBD |
| Smithery | — (URL-based, no repo manifest) | `smithery mcp publish <url>` or web form | TBD |
| Official MCP Registry | ✓ `server.json` | `mcp-publisher publish` | TBD |
| PulseMCP | (downstream auto-ingest) | Verify after 7d | TBD |
| MCPHub | — | Investigate submit path | TBD |
| Klavis AI | — | Deferred (separate bead) | N/A |

When all URLs are recorded, update this doc and close sp-ege.7 + sp-fiv.
