# SynthBench integration (`--submit-to-synthbench`)

SynthBench is the public benchmark and leaderboard for measuring how
faithfully an LLM-driven panel reproduces a known human distribution.
SynthPanel can submit a calibrated panel run directly to SynthBench at
the end of the run with one extra flag.

This integration is **opt-in**, **gated on `--calibrate-against`**, and
ships nothing without explicit user consent.

## When you can submit

Only **calibrated** runs are submittable. SynthBench scores per-question
JSD against a published human baseline, which SynthPanel produces only
when you pass `--calibrate-against DATASET:QUESTION` (see
[`docs/convergence.md`](convergence.md) for the calibration mechanics).

A bare `synthpanel panel run --personas ... --topic "pricing"` is
qualitative output and has no SynthBench score to submit. The CLI
hard-fails at parse time if you try:

```bash
$ synthpanel panel run ... --submit-to-synthbench
Error: --submit-to-synthbench requires --calibrate-against. Only
calibrated runs produce a SynthBench-shaped score; bare panel runs
cannot be submitted to the leaderboard.
```

## Quickstart

```bash
# 1. Mint an API key at https://synthbench.org/account
export SYNTHBENCH_API_KEY=sk_synthbench_...

# 2. Run a calibrated panel and submit at completion.
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument happiness-probe \
  --calibrate-against gss:HAPPY \
  --convergence-check-every 20 \
  --submit-to-synthbench
```

First-time use prints a one-screen consent block; accept with `y`. The
acceptance is recorded at `~/.synthpanel/synthbench-consent.json` so
subsequent runs do not re-prompt. For CI pass `--yes` to bypass the
prompt:

```bash
synthpanel panel run \
  --personas ./ci-personas.yaml \
  --instrument happiness-probe \
  --calibrate-against gss:HAPPY \
  --convergence-check-every 20 \
  --submit-to-synthbench --yes
```

On success the CLI prints:

```
Submitted to SynthBench: https://synthbench.org/submit/sub_abc123
```

## What gets uploaded

Per the consent notice:

* Per-question categorical response distributions (the
  `model_distribution` used to compute the calibration JSD).
* The calibration spec (e.g. `gss:HAPPY`), extractor label, and panel
  sample size *n*.
* Run config: model identifier(s), persona pack name, instrument name.
* The SynthPanel client version.

## What does NOT get uploaded

* Free-text panelist responses or follow-ups.
* Persona definitions, system prompts, or any persona attributes.
* API keys, file paths, or local environment data.

**Do not use `--submit-to-synthbench` with confidential personas,
proprietary instruments, or topics you would not publish on a public
leaderboard.** The leaderboard is public; assume anything in the
uploaded payload is world-readable.

## Failure modes (and what they mean)

The submission step is **warned-but-non-fatal**: a slow or rejecting
SynthBench cannot turn a successful panel run into a non-zero CLI exit.
If something goes wrong you will see a `Warning: SynthBench submission
not accepted (...)` line on stderr but the panel data is still in the
JSON output and any `--save` location.

| `status`               | Meaning                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `not_submittable`      | Run was invalid or carried no calibration JSD.              |
| `missing_api_key`      | `SYNTHBENCH_API_KEY` was unset (caught at parse time).      |
| `consent_declined`     | User answered `n` at the consent prompt.                    |
| `empty_payload`        | No question had both a model and human distribution.        |
| `http_<code>`          | Server rejected with a specific status; `error` carries it. |
| `error`                | Network-level failure (timeout, DNS, refused).              |
| `accepted` / `validating` / ... | Server-reported terminal state on success.        |

The `http_422` case is the most informative: SynthBench's Tier-2
recomputation found a schema mismatch and returned the field-level
reason. Surface that to the SynthBench team if it persists across runs.

## Privacy + consent record

Consent is stored as JSON at `~/.synthpanel/synthbench-consent.json`:

```json
{
  "version": 1,
  "accepted": true,
  "client_version": "0.11.0"
}
```

Delete the file to be re-prompted on the next run. The file is
versioned: a future major change to what gets uploaded will bump
`version` and re-prompt even if consent is on disk.

## Configuration

| Variable                | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| `SYNTHBENCH_API_KEY`    | Required. Bearer token for the `/submit` endpoint.        |
| `SYNTHBENCH_API_URL`    | Optional. Override the default `https://api.synthbench.org`. |

Use `SYNTHBENCH_API_URL` to point at a staging instance during
SynthBench-side development. Both env vars match the names the
`synthbench` CLI itself uses.

## See also

* [`docs/convergence.md`](convergence.md) — the calibration / JSD
  mechanics that produce the score this integration uploads.
* `synth_panel.synthbench_submit` — the implementation, including the
  payload transformer and HTTP transport.
