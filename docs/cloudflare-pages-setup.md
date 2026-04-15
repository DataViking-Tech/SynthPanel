# Cloudflare Pages — synthpanel.dev

Operational runbook for the static landing site served at <https://synthpanel.dev>.

## Source

- Files live in [`site/`](../site/) at the repo root.
- Pure static HTML; Tailwind is loaded from `cdn.tailwindcss.com`. **No build
  step is required.**
- `site/_headers` ships the security header set (CSP, HSTS, frame deny, etc.).
  Cloudflare Pages applies these automatically for any path matched by the
  `/*` rule.

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

## Why a separate site (vs. README only)

GitHub renders the README well for developers but isn't a domain users can
land on after seeing the project mentioned. `synthpanel.dev` exists so
that "synthpanel" the brand has a stable address that long-outlives any
single repo host.
