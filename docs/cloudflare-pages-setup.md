# Cloudflare Pages — synthpanel.dev

Operational runbook for the static landing site served at <https://synthpanel.dev>.

## Source

- Files live in [`site/`](../site/) at the repo root.
- Pure static HTML; Tailwind is loaded from `cdn.tailwindcss.com`. **No build
  step is required for HTML.**
- `site/_headers` ships the security header set (CSP, HSTS, frame deny, etc.)
  via the `/*` rule, plus RFC 8288 `Link` headers for agent discovery
  (`api-catalog`, `service-doc`) on the homepage `/` rule. A scoped block
  attaches JSON content-type + wildcard CORS to
  `/.well-known/mcp/server-card.json` (AR-7 / sy-02p) so agent discovery
  scanners can fetch the card cross-origin. Cloudflare Pages applies all of
  these automatically.
- `site/_worker.js` is a Pages **Advanced Mode** worker that implements
  `Accept: text/markdown` content negotiation (see "Markdown for Agents"
  below). Markdown renditions are pre-built from each HTML page by
  `scripts/render_site_markdown.py` and committed alongside the source.
- `site/.well-known/mcp/server-card.json` is the MCP Server Card
  (SEP-2127). The version inside the card is drift-guarded against
  `synth_panel.__version__` by `tests/test_well_known_server_card.py` —
  bump both together at release time.

## Cloudflare Pages project (one-time setup)

Done once by a maintainer with Cloudflare dashboard access. The polecat
workflow cannot perform these steps because they require interactive auth /
domain ownership confirmation.

1. **Create the project** (Pages → Create → Connect to Git)
   - Project name: `synthpanel`
   - Production branch: `main`
   - Repository: `DataViking-Tech/SynthPanel`
2. **Build configuration**
   - Framework preset: `None`
   - Build command: *(leave empty — pure static)*
   - Build output directory: `site`
   - Root directory: *(leave empty)*
3. **Environment variables**: none required.
4. **Custom domain**
   - Pages project → Custom domains → `synthpanel.dev`
   - Cloudflare auto-creates the proxied CNAME on the synthpanel.dev zone.
   - Confirm orange-cloud (proxied) and SSL/TLS mode = Full (strict).
5. **Verification**
   ```bash
   curl -sI https://synthpanel.dev/ | head -5
   #   HTTP/2 200
   #   content-type: text/html; charset=utf-8
   #   content-security-policy: default-src 'self'; ...
   ```

After the initial setup, every push to `main` redeploys automatically.

## Local preview

```bash
# from repo root
python3 -m http.server --directory site 8080
open http://localhost:8080/
```

There is no test suite for the site — visual smoke check only.

## Editing checklist

When you change `site/index.html`:

- Keep the install command (`pip install synthpanel`) discoverable above the
  fold. Discoverability of the install command is the page's main job.
- Keep the SynthBench cross-link present. The bead authoring this site
  (sp-p3g) made the cross-link an explicit acceptance criterion.
- If you bump Tailwind to the production build, drop the
  `cdn.tailwindcss.com` entry from `site/_headers` CSP `script-src`.

When you change **any** `site/**/*.html`:

- Re-run `python scripts/render_site_markdown.py` to regenerate the
  agent-facing `.md` rendition. CI's `test_committed_markdown_matches_fresh_render`
  fails the build if you forget.

## Markdown for Agents (content negotiation)

The site honors `Accept: text/markdown` per Cloudflare's
[Markdown for Agents reference](https://developers.cloudflare.com/fundamentals/reference/markdown-for-agents/)
and the
[`agent-skills/markdown-negotiation`](https://isitagentready.com/.well-known/agent-skills/markdown-negotiation/SKILL.md)
skill. AI agents that prefer structured markdown over rendered HTML
get a clean, token-efficient rendition; browsers continue to receive
HTML untouched.

**Verification:**

```bash
# HTML by default
curl -sI https://synthpanel.dev/ | grep -i content-type
#   content-type: text/html; charset=utf-8

# Markdown when explicitly requested
curl -sI -H 'Accept: text/markdown' https://synthpanel.dev/
#   HTTP/2 200
#   content-type: text/markdown; charset=utf-8
#   vary: Accept
#   x-markdown-tokens: 1827

# 406 when no rendition exists for the path
curl -sI -H 'Accept: text/markdown' https://synthpanel.dev/og-image.png
#   HTTP/2 406
```

**Implementation:**

- `site/_worker.js` is the Pages worker. It runs first for every
  request, and only intercepts when the `Accept` header explicitly
  prefers `text/markdown`. All other traffic falls through to static
  asset serving (HTML, images, sitemap, robots.txt).
- Markdown renditions live next to the HTML they mirror —
  `site/index.html` ⇄ `site/index.md`,
  `site/mcp/index.html` ⇄ `site/mcp/index.md`, etc.
- The renditions are produced by `scripts/render_site_markdown.py`, a
  stdlib-only Python script that walks `site/**/*.html` and emits a
  structured markdown rendition (headings, lists, code blocks, tables,
  links). The script is idempotent; rerun whenever HTML changes.
- The `x-markdown-tokens` response header is an approximate token
  count (chars / 4) for budgeting purposes; agents that need exact
  numbers should run their own tokenizer over the body.

## Why a separate site (vs. README only)

GitHub renders the README well for developers but isn't a domain users can
land on after seeing the project mentioned. `synthpanel.dev` exists so
that "synthpanel" the brand has a stable address that long-outlives any
single repo host.
