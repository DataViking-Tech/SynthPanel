# Site Audit: althing.dev vs CLI Surface

**Date:** 2026-04-30  
**CLI version audited:** 0.12.0  
**Auditor:** GH-318

## Framework

Static HTML — no Astro/Next.js. `site/index.html.j2` is a Jinja2 template
rendered by `scripts/render_site.py`. All other pages are hand-authored HTML.
Tailwind CSS is loaded from CDN (no build step). No existing CI check for
site/CLI sync.

## Pages

| URL | File | Status |
|-----|------|--------|
| `/` | `site/index.html.j2` → `site/index.html` | ⚠️ Outdated — missing `login` in Quick Start; no mention of `pack calibrate` |
| `/mcp` | `site/mcp/index.html` | ✅ Up-to-date — all 12 MCP tools documented |
| `/recommended-models` | *(missing)* | ❌ Missing — promised in 0.12.0 CHANGELOG, `docs/recommended-models.md` exists but no site page |
| `/docs/calibration` | *(missing)* | ❌ Missing — referenced in `calibration:` pack YAML output (`methodology_url`), `docs/calibration.md` exists |
| `/blog/althing-vs-commercial-alternatives.html` | `site/blog/…` | ✅ Up-to-date |

## CLI commands vs site coverage

### Documented on site

- [x] `althing prompt` — Quick Start snippet on `/`
- [x] `althing panel run` — Quick Start snippet on `/`
- [x] `althing report` — Quick Start snippet on `/`
- [x] `althing mcp-serve` — MCP section on `/` and full docs on `/mcp`

### Missing or undocumented

- [ ] `althing login` / `logout` / `whoami` — not mentioned anywhere (added in 0.9.x)
- [ ] `althing pack calibrate` — new in 0.12.0, not on site
- [ ] `althing pack generate` / `search` / `import` / `export` / `show` / `list` — not on site
- [ ] `althing instruments list` / `install` / `show` / `graph` — not on site
- [ ] `althing analyze` — not on site
- [ ] `althing cost summary` — not on site
- [ ] `althing panel inspect` — not on site
- [ ] `althing panel synthesize` — not on site

### panel run flags not documented

- [ ] `--models` (weighted per-persona or ensemble)
- [ ] `--blend` (distribution blending)
- [ ] `--variants N` (LLM-perturbed persona variants)
- [ ] `--convergence-check-every` / `--auto-stop` / `--convergence-*`
- [ ] `--calibrate-against` (inline calibration vs SynthBench)
- [ ] `--best-model-for` (SynthBench leaderboard model selection)
- [ ] `--submit-to-synthbench`
- [ ] `--synthesis-strategy` / `--synthesis-auto-escalate`
- [ ] `--personas-merge` / `--personas-merge-on-collision`
- [ ] `--extract-schema`
- [ ] `--rate-limit-rps`
- [ ] `--checkpoint-dir` / `--checkpoint-every` / `--resume` (standalone)

## Gaps fixed in this PR (GH-318)

- `site/index.html.j2`: Quick Start step 2 now shows `althing login` as
  an alternative to the env-var approach
- `site/index.html.j2`: Added `pack calibrate` to the "further features" note
  below the Quick Start
- `site/index.html` regenerated via `python scripts/render_site.py`

## Follow-up issues filed

See linked beads. Priority order:

1. **Missing `/recommended-models` page** — promised in 0.12.0 CHANGELOG;
   source content ready in `docs/recommended-models.md`
2. **Missing `/docs/calibration` page** — `methodology_url` in pack YAML
   points to it; source content in `docs/calibration.md`
3. **CLI reference pages** — undocumented subcommands above (`pack`,
   `instruments`, `analyze`, `cost`, `panel inspect`, `panel synthesize`)
4. **Advanced `panel run` flag docs** — ensemble, convergence, calibration
   flags are significant features with no site coverage
5. **CI sync check** — add a GitHub Actions job that runs
   `althing --help` and diffs against known coverage so drift is caught
   at the PR level
