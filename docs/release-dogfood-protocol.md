# Release Dogfood Protocol

A cross-town convention for surfacing friction on each significant
Althing release before it hits external users.

## Why this exists

Althing's primary surface is *other agents calling it from other
codebases*. The single-author dogfood that backs most pre-release QA
catches the failure classes that author already knows to test. It does
not catch:

- Orchestrator-dispatch asymmetries (e.g. v1 single-round vs.
  multi-round paths handling attachments differently)
- Transport-layer regressions on non-default providers (e.g. OpenRouter
  Anthropic-passthrough vs. native Anthropic)
- Output-schema asymmetries between modes (e.g. ensemble emitting no
  synthesis while single-model and weighted-split do)
- Cost-telemetry drift against provider-reported values

Three independent dogfood sweeps on v1.0.4 → v1.0.5 (yggdrasil, midgard,
jotunheim) each surfaced a *different* class of friction. None of the
three would have caught the others' findings unprompted. The cadence
below locks that property in.

## Cadence

A friction sweep runs against each of:

- **Major release** (`n.0.0`)
- **Minor release** (`n.m.0`)
- **First patch on a major** (`n.0.1`) — catches regressions in the
  initial fix wave
- **Release candidates** (`n.m.0-rc`) — optional, recommended for
  releases that touch dispatch/transport layers

Waves arrive within **24h of PyPI publish**. After 24h, friction is
filed as a normal bead without the sweep tag.

## Roles

- **Receiving mayor** — the town that owns Althing for this release
  cycle (currently `jotunheim`). Bundles incoming wave reports into
  beads in the Althing rig DB (the canonical store). Drops
  sister-bug callouts when an incoming wave overlaps with a separately
  filed local issue (e.g. yggdrasil's v1 bank-ref silent drop being a
  sibling of jotunheim's MCP v3 multi-round termination — same
  architectural smell, different surface).

- **Sweeping mayors** — every other town with a Althing rig runs
  their own dogfood independently, against their own product surfaces
  and use cases. Independence is the point. **Do not coordinate on
  what to test before sweeping** — the divergence is the signal.

## Wave structure

Each wave is a single mail to the receiving mayor with subject prefix
`[FRICTION wave N]`. The wave includes:

1. **Friction items**, numbered. Each item has:
   - **Repro** — minimal YAML / shell command that reproduces.
   - **Observed vs. expected** — what happened, what should happen.
   - **Root-cause hypothesis** — when traceable. Cite source files +
     line numbers if you read the code. ("Read the source" is fair
     game in dogfood — the polecat picking up the bead benefits.)
   - **Recommendation** — when the fix is decideable. Multiple options
     are welcome; the polecat picks.

2. **Positive verdict-equivalents.** Things you tested that worked,
   especially when the working behavior was non-obvious from docs.
   This prevents the "all bad news" framing and gives polecats a list
   of regression-test surfaces to protect.

3. **Still ahead.** What you haven't tested yet but plan to. Lets the
   receiving mayor see in-flight scope and queue follow-ups.

The receiving mayor responds with a single mail acknowledging triage:

- **Beads filed**, by priority. Each bead links back to the wave-mail
  ID (cross-town message references survive in the Dolt audit trail).
- **Sister-bug callouts** when an incoming item overlaps with a
  separately-filed local bead. Naming the smell ("orchestrator
  silently degrades when a feature isn't wired into the right dispatch
  path") is more valuable than naming the bug.

## Bead conventions

When filing a bead off a wave report:

- Label with `cross-town-reported` so the rig DB can surface them in
  release-readiness queries.
- Include the **verbatim repro** from the wave mail in the description.
  Do not summarize. The polecat needs the exact YAML/command.
- When the wave proposed multiple fix options, enumerate them in the
  description. The polecat is empowered to pick.
- Link back to the source wave-mail in the Reporter footer using a
  `Source-wave:` line. The `(item N of M)` clause is optional but
  useful when one wave bundles multiple beads — it lets a polecat
  re-read the original wave context without paginating through every
  derived bead's description.

Example footer:

```
## Reporter

yggdrasil mayor (althing v1.0.4 dogfood, 2026-05-10)
Source-wave: hq-wisp-l4a (item 3 of 4)
```

## Integration with the release process

A release is **dogfood-ready** when:

- Synthpanel CI green on the release tag
- Receiving mayor has dispatched the friction-sweep heads-up to peer
  mayors
- (Optional) A 24h window for waves to arrive

A release is **dogfood-complete** when:

- All received waves have been triaged into beads
- P1 friction items have been slung (in flight or shipped)
- P2 items have been at minimum acknowledged in a release note follow-up

The release entry in `CHANGELOG.md` includes a footer:

> Friction-sweep contributors (this release): `<town list>`. Bead IDs:
> `<comma list>`. Sister-bug pattern callouts: `<one-line summaries>`.

## What "friction" includes

- Behavioral bugs (silent drops, wrong outputs, crashes)
- API ergonomics (surprising required arguments, no-op flags, error
  messages that point at the wrong layer)
- Documentation gaps that cost time to author past (e.g. attachments
  in this product — no example YAML in repo before v1.0.5's sweep
  caught it)
- Cost / observability drift (estimator-vs-actual divergence, missing
  telemetry on a code path)
- Output-schema asymmetries between configuration modes
- Anything the dogfooder *had to read source code to figure out* —
  that's a doc gap or a UX gap by definition

## What "friction" does not include

- Feature requests (file as a normal bead, not as friction)
- Personal preference rebases ("I'd prefer if X were named Y")
- Bugs you can't reproduce in 1 minute — they get a normal bead with
  whatever evidence you have, not a friction-sweep entry

## Appendix: cross-rig pattern notes

When a sweep cycle surfaces a *meta-finding* — e.g. multiple
independent friction items share an underlying architectural smell —
the receiving mayor includes a `## Cross-rig pattern notes` section in
their triage reply naming the pattern. These notes are the
highest-value output of the cadence; they're what no individual
dogfooder can produce alone. Anchor them in the release notes when
they're load-bearing.

Example (v1.0.4 → v1.0.5 cycle):

> Orchestrator dispatch paths silently degrade when a feature isn't
> wired into every entrypoint. v1 single-round path skipped the
> attachment resolver; MCP v3 multi-round path terminated after round
> 1. Same smell, different surface. Fix scope should include CI
> integration tests that exercise every dispatch path against every
> feature, not just the ones the original author thought to wire.
