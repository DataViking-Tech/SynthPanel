# Attachments: showing panelists images, web pages, and HTML snippets

When you want a synthetic panel to react to something visual or external
— a screenshot, a live landing page, a marketing snippet — you give the
instrument an **attachment bank** and reference its entries from
individual questions.

A copy-pasteable starter lives at
[`examples/instruments/with-attachments.yaml`](../../examples/instruments/with-attachments.yaml).
Run it with:

```bash
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/instruments/with-attachments.yaml
```

## The four attachment types

| Type | Source | Use when |
|------|--------|----------|
| `image` | `base64` data, `url`, or uploaded `file_id` | You want panelists to react to a screenshot, mockup, or photo. |
| `document` | `base64` data, `url`, or `file_id` | You want panelists to read a PDF (spec, brief, contract). |
| `url` | A live URL fetched at run time | You want panelists to react to a real, currently-live page. |
| `html` | Inline string | You have a literal HTML/markup fragment — pricing block, email body, copy variant. |

## Shape

The bank is a top-level mapping keyed by attachment id. Each id must
match `^[a-z][a-z0-9_-]{0,63}$`. Questions reference ids by name:

```yaml
instrument:
  version: 3
  attachments:
    hero_screenshot:
      type: image
      media_type: image/png
      source:
        type: base64
        data: iVBORw0KGgoAAAANSUhEUgAAAAEA...
    landing_page:
      type: url
      url: https://example.com/pricing
      fetch_mode: markdown
    pricing_snippet:
      type: html
      text: |
        <div class="pricing">
          <h2>Starter — $9/mo</h2>
        </div>
  rounds:
    - name: visual_reaction
      questions:
        - text: "What stands out about the screenshot?"
          attachments: [hero_screenshot]
        - text: "Does the live pricing page land?"
          attachments: [landing_page]
        - text: "Is the Starter tier markup fairly priced?"
          attachments: [pricing_snippet]
      route_when:
        - else: __end__
```

## Notes

- **`fetch_mode` for `url`** — one of `auto`, `html_text`, `screenshot`,
  or `markdown`. `markdown` is usually the cleanest signal for
  content-heavy pages; `screenshot` is the right choice when layout or
  visual hierarchy is the thing under test.
- **Inline HTML** — YAML's `|` (literal block scalar) preserves newlines
  and indentation, which is what you want for HTML fragments.
- **Image media types** — `image/png`, `image/jpeg`, `image/gif`,
  `image/webp`. Anything else fails at parse time.
- **Document media types** — currently only `application/pdf`.
- **Attachment ids are validated at parse time** — typos in attachment
  references fail fast with `attachment ref '…' does not resolve to a
  top-level attachment`.

## See also

- [`examples/instruments/with-attachments.yaml`](../../examples/instruments/with-attachments.yaml)
  — the runnable starter this page describes.
- [Instrument regression testing in CI](./instrument-regression-testing.md)
  — locks panel behavior, including attachment-driven runs.
