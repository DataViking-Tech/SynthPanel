# Research Map — sp-pack-registry

Synthesis of three research dimensions for the D-phase gate. Ticket now revealed: **sp-5roa — community persona + instrument pack registry.**

## What the codebase currently provides

**Pack delivery is wheel-packaged + directory-scanned.** 9 bundled persona packs + 8 instrument packs ship as `*.yaml` resources under `synth_panel.packs` / `synth_panel.packs.instruments`. Discovery via `importlib.resources.files()` + filename-stem pack ID. Update mechanism: `pip install --upgrade synthpanel`.

**User-saved packs live at `~/.synthpanel/persona_packs/`** (overridable by `SYNTH_PANEL_DATA_DIR`). User-saved shadows bundled by pack ID (no warning). 5 pack subcommands: `list`, `import`, `export`, `show`, `generate`. Instruments have `install` (with pre-install validation) but no `import/export/generate` equivalents yet.

**Persona schema is implicit, code-enforced:** `validate_persona_pack()` at `mcp/data.py:204-241` requires `personas` list non-empty + each persona dict with `name`. Optional fields (`age`, `occupation`, `background`, `personality_traits`) unvalidated beyond type. **No `version:` field for persona packs.** Instruments require `version:` in manifest (schema version 1/2/3, all bundled are v1, unrelated to synthpanel version). No migration tooling for either.

**Actual corpus shows authoring-friction patterns.** Audit of 19 pack files (9 bundled + 10 custom) = 172 personas, 2,149 LOC:
- **~40 personas duplicated byte-identically** across packs (e.g., Abdul Rahman appears in both `job-seekers.yaml` and `extra_50_traitprint.yaml`)
- `extra_50_*` packs are wholesale copies of bundled equivalents with zero net-new content — authored as scale-up workarounds, not extensions
- Only genuine authored content in `contrarian_*` (single stress-test persona) and `icp_*` (5 product-specific personas)
- Stylistic drift in `personality_traits`: bundled packs use atomic 2-3 word strings; custom ICP packs use compound complex phrases; both pass validation
- `personality_traits` silently lowercases via `str(t).strip().lower()` — undocumented normalization rule

**Validation error path is minimal.** Generic messages: `"personas must be a list"`, `"persona at index {i} is missing required field 'name'"`. No user-facing schema documentation. Schema discovery = reading Python source.

**No informal sharing channels documented.** No README section on pack distribution, no community-channel references, no blog posts on authoring. Inferred pathway: local YAML → `pack import` → email/slack/PR the exported YAML.

## How comparable ecosystems solve this

**Three dominant patterns across 11 audited tools:**

1. **Decentralized Git-URL + manifest** (pre-commit, cookiecutter): zero infrastructure, repositories are namespaces, maximum discovery friction, author owns everything. Abandoned repos persist indefinitely. No moderation. Scales for small specialized ecosystems.

2. **Centralized curated registry + distributed hosting** (HACS for Home Assistant, conda-forge): manifest-based registration via PR to a `default` list or staged-recipes repo. Curates community contributions. Explicit submission + review process. Transparent community governance scales better than opaque single-maintainer (dbt Hub's undocumented submission is documented as a failure mode). Version compatibility flags prevent broken installs.

3. **PyPI namespace + entry-point auto-discovery** (pytest plugins, Sphinx extensions): leverages existing infra, zero user config, but inherits PyPI's flat-namespace vulnerabilities (typosquatting, character-normalization collisions). Entry-point name conflicts resolve non-deterministically. Abandoned packages clutter indefinitely.

**Failure patterns shared across all centralized registries:**
- Opaque submission/governance creates author friction and maintainer burnout
- Abandoned content persistence without cleanup mechanism
- Moderation scaling requires documented review criteria + appeals
- Security review is typically delegated or omitted

**Cross-cutting governance lessons:** document submission/review/maintenance explicitly; transparency builds trust even with strict processes. Avoid flat namespaces for plugin namespacing. Pair entry-point auto-discovery with explicit allow-listing or plugin-load logging.

## Observable seams (current extensibility)

1. **`_bundled_packs()` directory scan** (`mcp/data.py:83-102`) — any `*.yaml` in `synth_panel.packs` is discovered. A community-pack directory on disk would plug in via the same mechanism if packaged as a namespace package or via a separate entry point.
2. **`$SYNTH_PANEL_DATA_DIR` override** — single env-var configuration. A registry install-path could reuse this or extend with a subdirectory (`~/.synthpanel/registry-packs/`).
3. **Pack ID shadowing with no warning** — `data.py:162-173` check silently. Adding a collision warning is a focused change that also benefits sp-g270's existing work on merge-level collisions.
4. **`pack import`** — already handles file → saved-pack flow. Extending to accept URLs (`pack import https://...` or `pack import gh:user/repo`) is an additive flag on an existing command.
5. **`pack generate`** — already produces a pack via LLM with structured output. A "submit this to a registry" action is adjacent.
6. **Instrument manifest has `version`, persona pack doesn't** — asymmetry worth resolving before any registry would need to track versions.

## Design-space tensions (for D-phase discussion)

**Centralization tension.** Decentralized git-URL (cookiecutter style) is zero-infra but fragments discovery; centralized registry is moderatable but becomes a governance surface. Precedent in Gas Town favors minimal centralization, but a registry's value is primarily discoverability.

**Where the registry lives** — synthpanel.dev static site vs GitHub Pages vs a dedicated repo (`synthpanel/registry` or `synthpanel-packs/index`) vs a third-party tap pattern. Each has different implications for author submission friction, moderation surface, and cross-machine reproducibility.

**Versioning semantics.** Persona packs currently have no version field; adding one is a schema change. Options: (a) ignore versioning, let authors re-submit (pre-commit style), (b) add optional `version:` field, (c) adopt the instrument manifest pattern wholesale for consistency. Registry needs *some* version signal for users to pin or upgrade.

**Calibration score integration** — research (sp-inline-calibration) overlaps here. If packs carry a SynthBench calibration fingerprint (hash + score), the registry page becomes a quality-signal surface. This is potentially the strongest differentiator vs peer registries but depends on calibration tooling landing first.

**Submission/review tension.** Community-driven transparent governance (conda-forge pattern) is high-trust but requires maintainer time. PR-to-default-list (HACS pattern) reduces moderation burden but requires a clear manifest schema + repository structure convention. Publish-and-pray (PyPI-style) has lowest author friction but highest supply-chain risk.

**Security surface.** Persona packs are YAML — safe to load via `yaml.safe_load`. But packs executed as panel personas carry *behavioral* content (prompt content shaping what LLMs emit) that can be adversarial without being executable code. Community packs could be used to inject prompt-injection attacks into downstream consumers' panels. Moderation considerations extend beyond executable-code safety.

**Branding tension.** A community marketplace with DataViking/synthpanel attribution raises brand exposure. HACS-style centralization puts maintenance of the manifest registry on synthpanel maintainers; pure decentralization (cookiecutter-style) fragments but lets the community self-identify.

**Sharing-the-authoring-corpus tension.** The audit shows ~40 duplicate personas across existing packs — evidence that scale-out via copy-paste is the current authoring workflow. A registry could reduce duplication if the discovery surface made reuse easier than copying. But if the registry is too far from the current `pack import` workflow, duplication continues.

## What the research does not answer

- Whether there is observable community demand (Slack, Discord, GitHub issues) for a shared pack registry today, or whether this remains a synthetic-panelist ask
- How much friction is acceptable for authors submitting to a registry vs authors sharing via gist/email
- Whether calibration scores should gate registry inclusion or be informational
- What the moderation team capacity looks like relative to expected submission volume
- Whether the registry is for personas, instruments, or both (they have different authoring friction profiles)

## Files

- `research/pack-lifecycle-and-delivery.md`
- `research/python-ecosystem-patterns.md`
- `research/authoring-friction-evidence.md`

## Ready for D-phase gate

No design direction proposed. Observable friction, observable seams, observable tensions. Human brain-surgery on direction is the next gate.
