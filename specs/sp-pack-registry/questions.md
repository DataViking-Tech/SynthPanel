# Q-phase: Persona + instrument pack distribution

*Ticket hidden per QRSPI discipline. These questions drive objective code-mapping only; do not invent design direction from them.*

## Problem (shared with R polecats)

Users author persona packs and instruments as YAML files today. Map how packs currently enter a user's environment (authoring, sharing, installing, validating, upgrading), what lifecycle invariants already exist, and what comparable content-distribution patterns exist in the Python ecosystem and adjacent tools.

## Research dimensions

### 1. Pack lifecycle today — authoring to consumption

- How does a user create a new persona pack today? What are the steps from blank editor to runnable-in-`panel run`?
- How does a user create a new instrument? Any differences in lifecycle vs personas?
- How does a user pass a pack to a teammate or to a second machine? Copy the YAML? Include it in a repo? Email?
- Are there any existing `pack` subcommands for discovery (list, show, search, browse)? What do they accept as input?
- Are there any `pack import/export/install/uninstall/save` commands? Where do saved packs live on disk?

### 2. Bundled pack delivery mechanism

- How do the 9 bundled packs reach a user's machine? Package data in the wheel? Downloaded? Inlined?
- How does the code discover bundled packs at runtime (directory scan, explicit registry, entry points)?
- What's the update mechanism today — upgrade `synthpanel` and new packs appear?
- How are bundled packs namespaced vs user-authored packs vs mayor's `extra_*` packs? What's the collision-resolution behavior (sp-g270 is partially related)?

### 3. Pack validation and versioning

- Is there a schema for persona packs? For instruments? Where is it defined? What does it enforce?
- How are pack versions tracked today? Is there a `version:` field? How does it interact with `synthpanel`'s own version?
- How are breaking changes to persona / instrument schemas handled in the codebase? Any migration tooling?
- If two packs have the same name, what wins? What's the deterministic order?

### 4. User-saved pack persistence

- What user-owned storage does `synthpanel` use (`~/.synthpanel/*`, XDG_DATA_HOME, other)?
- How is the location selected, overridden, and documented?
- What's the permission / privacy model — are saved packs world-readable, user-only, other?
- How do tests avoid collision with real user state (env-var overrides, tempdir fixtures)?

### 5. Adjacent Python-ecosystem distribution patterns

- How do comparable tools distribute user-authored content: `pre-commit` hooks, `cookiecutter` templates, `dbt` packages, `jinja` templates, `jupyter-contrib-extensions`, `homeassistant` integrations, `opensearch-dashboards` plugins?
- Which use a git-URL install model (`pip install git+…`), a registry server (PyPI, npm, custom), a GitHub-topic-based discovery model, a static-index file, or other?
- What is `pip install synthpanel-pack-foo` style namespace distribution like for those tools? Does anyone do it?
- What's the submission / review / moderation model in each? Any failure patterns documented?

### 6. SynthBench cross-link

- SynthBench computes calibration metrics on synthpanel output against real human datasets. Is there already any per-pack signal flowing between the two projects?
- Could a pack carry a "calibration fingerprint" (hash + score) computed from a SynthBench run? What would that require from both sides?
- Is there any existing pack-level quality signal in the repo (test coverage, example runs, provenance metadata)?
- How does SynthBench reference persona / instrument packs today — by name, by hash, by path?

### 7. Authoring friction evidence from the existing corpus

- Audit mayor/ and the bundled packs for duplication, drift, re-invention patterns: same persona archetype spelled differently across packs? Same instrument question under different keys?
- What's the rough LOC / persona-count / file-count distribution of existing authored packs?
- What's the most common authoring error path (missing field, invalid YAML, name collision, wrong top-level shape)? What error messages do users see?
- Where do people paste pack YAML today (gist, slack, PR, local file) when sharing informally?

## Deliverable

`specs/sp-pack-registry/research/<dim>.md` per dimension, written objectively. Research phase synthesized into `research-map.md` for the human D-phase gate.
