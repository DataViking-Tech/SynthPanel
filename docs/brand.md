# Brand notes (synthpanel)

This file documents the brand decisions for synthpanel.dev as part of the
cross-property unification effort tracked under `hq-4ae`. The canonical
brand reference for the DataViking property family lives in the
dataviking-site repo.

## Family vs. product identity

synthpanel is part of the **DataViking** property family alongside
`dataviking.tech` (canonical brand reference) and `mtg-frontend`. The goal
is **family resemblance without sacrificing per-product identity** —
visitors should recognize a shared parent without the products feeling
homogenized.

Family resemblance is delivered structurally:

- A unified footer with the DataViking column structure (logomark +
  wordmark + © DataViking line + social row).
- A small `DataViking ↗` wordmark and back-link to `dataviking.tech` in
  the top-left of every page header.

Per-product identity is preserved through the accent palette and
typography, which deliberately differ across properties.

## Accent decision: retain terminal-green/teal

synthpanel retains its **terminal-green/teal accent** (`emerald-400`,
`#34d399`) rather than migrating to the dataviking-site gold/amber. The
rationale, per the `hq-4ae` umbrella bead:

> Green is **defensible as a per-product accent** — the terminal
> aesthetic suits an open-source developer tool, and the accent reads as
> a CLI prompt color rather than a brand drift. It can stay **if** the
> properties are unified by other means: typeface, footer, wordmark.

The unification work in this branch delivers exactly those "other means"
(footer + header back-link + DataViking wordmark), so the accent stays.

| Property        | Accent        | Reason                                       |
| --------------- | ------------- | -------------------------------------------- |
| dataviking.tech | gold / amber  | canonical brand reference                    |
| mtg-frontend    | gold (target) | re-skinning under separate brand-drift bead  |
| synthpanel      | terminal-teal | terminal aesthetic, OSS dev-tool affordance  |

If a future audit determines the accent split confuses users or weakens
the family signal, the migration path is straightforward: swap the
`emerald-*` Tailwind utilities used in `site/index.html.j2` and the
subpage HTMLs for the canonical gold token (one regex pass plus a
re-render). No structural HTML changes are needed.

## Tokens consumed from the canonical spec

The dataviking-site token-spec bead (`hq-pzy4`, re-filed as
`dataviking_site:dvs-cph`) is a soft dependency, not a hard one. The
structural patterns synthpanel adopts from it:

- **Footer column structure**: 3 nav columns + logomark column +
  copyright/social row.
- **Header wordmark**: small text wordmark with back-link to the parent
  property.
- **Soft-pill button radius**: synthpanel uses `rounded-md` on CTA
  buttons, which is close enough to the dataviking-site pill that no
  change is needed at this scale. A future pass can promote this to a
  shared token if the spec emits one.

## Surfaces touched by this work

- `site/index.html.j2` (home, source of truth) and the rendered
  `site/index.html`.
- `site/mcp/index.html`
- `site/recommended-models/index.html`
- `site/docs/calibration/index.html`
- `site/docs/panel-run/index.html`
- `site/blog/synthpanel-vs-commercial-alternatives.html`

All six surfaces share the same unified header and footer pattern.
