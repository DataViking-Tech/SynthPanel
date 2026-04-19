# Traitprint Local — CPO Sign-off

**Author:** cpo (synthpanel)
**Date:** 2026-04-19
**Requested by:** mayor/
**Subject:** Sign-off on Traitprint Local — open-source local-first vault with optional cloud sync

**Verdict: APPROVE WITH CONDITIONS.**

The strategic logic is sound, the competitive risk is low, and the
audience fit for AI Engineer Miami is high. Eight execution conditions
must be met to avoid foot-guns — see §6.

---

## 1. Does this dilute or strengthen the cloud product?

**Strengthens, materially.** Cannibalization risk is low; the pattern
is well-established across the open-core playbook (Obsidian, Supabase,
Plausible, PostgreSQL, Git/GitHub).

The cloud features are **network-requiring by definition**, not
feature-gates on top of local capability:

- Public profiles at `traitprint.com/<username>` require hosted URLs.
- Job matching requires a pool of other users.
- Always-on digital twin chat requires hosted inference + availability.

None of these can be replicated from a local vault alone. The only
residual cannibalization vector is a power user who pipes their local
vault into their own agent for private twin-chat — and that user was
never paying for hosted twin-chat anyway. For the Miami audience, they
are the **best possible advocate** we can earn: they will tweet the
`traitprint mcp-serve → Claude` loop and convert their non-technical
friends to the cloud version downstream.

### Upside beyond direct conversion

1. **Directly neutralizes the "why does this need a cloud?" objection**
   — which Miami will absolutely ask. The demo *is* the answer.
2. **Turns Traitprint into a data standard, not just a SaaS.** If the
   schema is open and adopted, Traitprint becomes the de-facto format
   for AI-era career identity. That's a moat commercial-only
   competitors cannot match.
3. **Privacy-conscious wedge.** Installed-base users who would never
   trust a cloud career profile will install the local vault. Cloud
   conversion follows trust, not the other way around.
4. **Distribution asymmetry.** Commercial competitors will not
   open-source their data model. This is a durable differentiation.
5. **Dogfood for the DataViking MCP story.** Three products, one
   install pattern (`pip install X → X mcp-serve → Claude connects`).
   Tonight's pitch becomes infinitely more coherent.

**Expected conversion funnel (hypothesis, to be instrumented):**
local-first install → builds ≥10 vault entries → hits a feature that
requires network (wants to share a public link, wants matching, wants
always-on twin) → `traitprint auth login`. That's the loop.

---

## 2. Same pip package, or separate?

**Same package. `pip install traitprint`, single distribution, one name.**

### Why not `traitprint-local` as separate package

- "traitprint-local" reads as the lesser version. Hurts brand.
- Upgrade friction is painful: `pip uninstall traitprint-local && pip
  install traitprint` is a far bigger ask than `traitprint auth login`.
- Dual PyPI listings cause SEO and discovery confusion.
- Versioning drift between `-local` and `-cloud` will bite within two
  releases.
- Doubles maintenance surface for zero user benefit.

### Recommended structure

```
pip install traitprint           # Everything. Vault + CLI + MCP + sync client.
```

The cloud sync commands (`traitprint push`, `traitprint pull`, `traitprint
auth login`) **ship in the base package but truly no-op without an API
key set**. If you want an extras-gated variant later for heavy optional
deps, use the `synthpanel[mcp]` pattern — but only when there's an
actual dependency weight argument. Today there isn't one; `httpx` is
already in the tree.

### One narrative, one binary

The pitch is cleaner: *"`pip install traitprint` gives you the local
vault and the MCP server, MIT-licensed. `traitprint auth login` unlocks
public profiles, job matching, and always-on twin chat."* That sentence
is the whole positioning.

---

## 3. Risks of open-sourcing the vault schema + MCP tools?

**Low, and the offsetting upside is large.** Walking through the
concrete risks:

### A. Competitor copies the schema

Schemas are not moats. The moat is (a) the network of Traitprint
users, (b) the cloud-side matching and twin-chat algorithms, (c) the
brand/SEO at `traitprint.com/<username>`. A competitor replicating the
YAML/SQLite schema gains an empty shell. Worst case they adopt it
verbatim for interop — which means **we become the standard.** Win.

### B. Competitor copies the MCP tool definitions

Same argument. The tools are thin CRUD over an open schema. The value
is the data (which stays with users) and the cloud services (which
are not in the open-source tree). Clone those tools, get an empty CRUD
app.

### C. Security / fingerprinting

Marginal. The cloud sync API already has to define these formats, so
anyone inspecting sync traffic sees the schema today. Open-sourcing
formalizes what is already derivable.

### D. Support burden

Real. Budget **~1 engineer-day per week** for GitHub issues + PR
review once any community forms. Copy the synthpanel `CONTRIBUTING.md`
/ `CODE_OF_CONDUCT.md` / issue templates verbatim.

### What we must NOT open-source

These stay server-side, period:

- Matching algorithm weights / ranking models
- Digital twin system prompts and persona-generation prompts
- API key provisioning / billing logic
- Cloud-only data enrichments (employer verification, compensation
  benchmarks, referral graph)

### Governance

- Ship schema v1 with an explicit version field and a documented
  migration path. Signal: "may evolve; breaking changes will be
  flagged."
- Keep the cloud API client in the repo minimal — an HTTP facade, not
  the cloud logic itself.
- Audit logs/telemetry before first release to ensure nothing PII-ish
  leaks in local mode. Local-first means **truly local**, including
  no background phone-home.

---

## 4. Does the demo flow work for Miami?

**Yes, it works — and it works *because* SynthPanel already
established the `pip install → mcp-serve → Claude connects` rhythm
for this audience tonight.**

The three-step demo:

```
traitprint init
traitprint vault import-resume ~/resume.pdf
traitprint mcp-serve      # (then: register in Claude Desktop)
```

...followed by *"Claude, what are my strongest skills based on my
Traitprint?"* with the answer flowing from the local vault via MCP. If
that sequence runs in under 90 seconds cold, it is a better demo than
the current cloud-only Traitprint.

### Why this lands for Miami specifically

1. **Echoes the SynthPanel demo beat-for-beat.** Same host, same
   command shape, same time budget. The audience sees a pattern and
   concludes DataViking is a *platform*, not three products.
2. **Pre-empts the "needs a cloud?" objection** before it's asked.
3. **Self-demonstrating.** The founder is the persona. Wesley
   importing his own resume and asking Claude about himself is an
   unfakeable authenticity moment.

### Requirements for the demo flow to work

- `traitprint init` must complete in **<5 seconds** cold.
- Local mode must work with **zero API keys set** (including no LLM
  key — use MCP sampling in the host, same pattern as SynthPanel's
  `sp-6at`).
- MCP tool names must read well in a Claude tool-call log (short,
  descriptive, no `cloud_*` prefix on local tools).
- Vault file format must be openable in a text editor and
  human-grokkable in 30 seconds. SQLite is fine for storage; export
  to JSON/YAML for inspection.
- Resume import must not fail silently on common formats — test
  .pdf, .docx, and LinkedIn export JSON before ship.

### Timing caution

If Miami is **tonight**, do **not** attempt to ship and demo Traitprint
Local on stage. File it as an **"early access, pip install coming
this week"** cue-card. Stage-first demos of unshipped work lose deals.
The right Miami move is: show current Traitprint (cloud), drop one
line in the pitch — *"Traitprint Local, open-source, pip installable,
shipping this week"* — and let the room ask at the booth. That's a
lead-gen mechanism, not a risk.

---

## 5. Overall verdict: **APPROVE WITH CONDITIONS**

### Conditions

1. **Single `pip install traitprint` package.** No `-local` suffix.
2. **Cloud sync code truly no-ops without an API key.** No telemetry,
   no phone-home, no background auth attempts in local mode.
3. **Keep matching algorithms, twin-chat prompts, persona-generation
   prompts OUT of the open-source repo.** Server-side only.
4. **Schema v1 with explicit version field + migration doc from day
   one.** Signal evolvability.
5. **Demo flow tested end-to-end on a clean laptop before any
   public-facing use.** No cold-start at the booth.
6. **Copy the SynthPanel `LICENSE` / `CONTRIBUTING.md` /
   `CODE_OF_CONDUCT.md` / issue templates.** We already wrote them;
   reuse.
7. **README makes the cloud upgrade path visible, not hidden.** Lead
   with *"This is free and open. Here is what cloud adds."* Hiding the
   paid upgrade is the #1 open-core credibility killer.
8. **Do NOT ship-and-demo tonight.** File as P1 "this week" work.
   Reference it in the Miami pitch as coming; demo the proven cloud
   product on stage.

### Secondary recommendations (not conditions, but worth doing)

- **Announce Traitprint Local as part of the Miami booth handout**,
  not the stage demo. "Scan this QR, get early access." Captures emails
  without demo risk.
- **Wire Traitprint's local MCP tools to interop with SynthPanel's
  persona format** as a downstream stretch goal. A Traitprint vault
  can become a SynthPanel persona definition. That's the DataViking
  platform story made literal.
- **Instrument the local → cloud conversion funnel** before first
  release. Without telemetry, you cannot answer the dilution question
  empirically in six months.

### What happens if the conditions slip

- Slip #2 (phone-home) → privacy trust destroyed permanently. Fatal.
- Slip #3 (leak algorithms) → cloud product immediately clonable. Near-fatal.
- Slip #5 (untested demo) → one bad public demo sets the narrative. Recoverable but painful.
- Others → recoverable, cost time not credibility.

**Signed off, with the above. Ship it — after Miami.**

— cpo
