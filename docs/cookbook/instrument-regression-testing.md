# Instrument regression testing in CI

Once you have calibrated a research instrument — `survey.yaml` plus a small
fixed persona pack — you usually want to *lock its behavior*. If a
teammate edits a question, you want CI to flag it: "this question used to
produce a 70/30 enum split; the new wording produces 40/60. Intentional?"

This recipe wires that up. It runs a small synthpanel run on every PR
against a baseline panel result committed to the repo, computes a
statistical diff, and fails CI if any answer distribution drifts past a
threshold you choose.

It's the same machinery a unit test suite gives you, applied to research
instruments.

## What you'll build

```
your-repo/
├── .github/workflows/instrument-regression.yml   # CI job
├── instrument-regression/
│   ├── personas.yaml                             # 3 fixed personas
│   ├── survey.yaml                               # the instrument under test
│   ├── baseline.json                             # committed panel result
│   └── check_drift.py                            # diff baseline vs new run
└── ...
```

Every PR runs the panel, diffs the new result against `baseline.json`, and
fails if any categorical question's Jensen–Shannon divergence exceeds your
threshold. When you intentionally change a question, you regenerate the
baseline and commit it.

## 1. The persona pack

Keep this small and diverse. Three panelists is enough for a smoke
regression; large panels make CI slow and expensive without adding much
sensitivity at this scale.

```yaml
# instrument-regression/personas.yaml
personas:
  - name: Sarah Chen
    age: 34
    occupation: Product Manager
    background: >
      Eight years in tech, currently at a mid-size SaaS company.
      Manages a team of five. Comfortable with technical detail.
    personality_traits: [analytical, pragmatic, detail-oriented]

  - name: Marcus Johnson
    age: 52
    occupation: Small Business Owner
    background: >
      Runs a family-owned restaurant chain with three locations. Not
      tech-savvy but recognizes the need for digital tools. Values
      simplicity and reliability over features.
    personality_traits: [practical, skeptical-of-technology, relationship-focused]

  - name: Priya Sharma
    age: 28
    occupation: Graduate Student
    background: >
      PhD candidate in computational linguistics. Heavy user of developer
      tools. Budget-conscious but pays for tools that save real time.
    personality_traits: [curious, technically-sophisticated, cost-conscious]
```

Pin it. If you regenerate the personas (rewording the background, adding
traits, swapping names) the baseline is no longer comparable.

## 2. The instrument

Mix free-text and structured (`enum` / `scale`) questions. The diff
command computes Jensen–Shannon divergence on categorical distributions
and theme drift on free text — the structured questions are what lets
you set a numeric threshold in CI.

```yaml
# instrument-regression/survey.yaml
instrument:
  version: 1
  questions:
    - text: >
        How would you describe your current relationship with workplace
        productivity tools?
      response_schema:
        type: enum
        options: [thriving, getting-by, struggling, disengaged]

    - text: >
        On a scale of 1-5, how likely are you to recommend your primary
        productivity tool to a colleague?
      response_schema:
        type: scale
        min: 1
        max: 5

    - text: >
        What is the single most frustrating part of your current workflow
        when collaborating with others on documents or projects?
      response_schema:
        type: text
      follow_ups:
        - "Can you describe a specific recent example?"

    - text: >
        Which of these matters most to you when adopting a new tool?
      response_schema:
        type: enum
        options: [price, ease-of-use, integration, security, support]
```

## 3. Generate the baseline

Run the panel once locally with the model and temperature you want
locked in, then commit the resulting JSON:

```bash
export SYNTH_PANEL_DATA_DIR="$PWD/.synthpanel"
mkdir -p instrument-regression

synthpanel --model haiku panel run \
  --personas instrument-regression/personas.yaml \
  --instrument instrument-regression/survey.yaml \
  --temperature 0.3 \
  --save 2>&1 | tee /tmp/run.log

# --save writes to $SYNTH_PANEL_DATA_DIR/results/result-<id>.json
# and prints "Result saved: result-<id>" to stderr.
result_id=$(grep -oE 'result-[0-9-]+-[a-f0-9]+' /tmp/run.log | head -1)
cp ".synthpanel/results/${result_id}.json" instrument-regression/baseline.json
git add instrument-regression/baseline.json
git commit -m "instrument-regression: lock baseline"
```

A few minutes of cost up-front in exchange for a reproducible reference
point. With three personas × four questions on `haiku` this is well
under $0.05.

## 4. The drift check

This script is the glue. It runs `synthpanel runs diff` between the
committed baseline and a fresh run, then enforces thresholds on the
structured questions and reports theme drift on the free-text ones.

```python
# instrument-regression/check_drift.py
"""Diff a fresh panel run against the committed baseline.

Exits non-zero if any categorical/scale question's Jensen-Shannon
divergence exceeds JSD_THRESHOLD, or if a free-text question gains
or drops more than TEXT_THEME_DELTA top themes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

JSD_THRESHOLD = 0.20         # 0.0 == identical, 1.0 == disjoint
TEXT_THEME_DELTA = 2         # max new+dropped top themes per question


def main(baseline: Path, new_result: Path) -> int:
    proc = subprocess.run(
        [
            "synthpanel", "--output-format", "json",
            "runs", "diff",
            str(baseline), str(new_result),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    diff = json.loads(proc.stdout)

    violations: list[str] = []

    for q in diff.get("categorical_questions", []):
        jsd = q["jsd"]
        if jsd > JSD_THRESHOLD:
            violations.append(
                f"  [categorical] {q['question_text'].strip()[:70]}…\n"
                f"    JSD {jsd:.3f} > {JSD_THRESHOLD}\n"
                f"    baseline={q['distribution_a']}\n"
                f"    new=     {q['distribution_b']}"
            )

    for q in diff.get("text_questions", []):
        churn = len(q["new_themes"]) + len(q["dropped_themes"])
        if churn > TEXT_THEME_DELTA:
            violations.append(
                f"  [text] {q['question_text'].strip()[:70]}…\n"
                f"    {churn} top-theme changes (max {TEXT_THEME_DELTA})\n"
                f"    new themes: {q['new_themes']}\n"
                f"    dropped:    {q['dropped_themes']}"
            )

    if violations:
        print("Instrument regression detected:\n", file=sys.stderr)
        print("\n\n".join(violations), file=sys.stderr)
        print(
            "\nIf this change is intentional, regenerate the baseline:\n"
            "  synthpanel --model haiku panel run \\\n"
            "      --personas .../personas.yaml \\\n"
            "      --instrument .../survey.yaml \\\n"
            "      --temperature 0.3 --save\n"
            "  cp .synthpanel/results/<new-id>.json instrument-regression/baseline.json",
            file=sys.stderr,
        )
        return 1

    print("Instrument regression check passed. No drift beyond thresholds.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_drift.py <baseline.json> <new.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
```

## 5. The CI workflow

```yaml
# .github/workflows/instrument-regression.yml
name: instrument-regression

on:
  pull_request:
    paths:
      - "instrument-regression/personas.yaml"
      - "instrument-regression/survey.yaml"
      - "instrument-regression/baseline.json"
      - "instrument-regression/check_drift.py"
      - ".github/workflows/instrument-regression.yml"

jobs:
  drift-check:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install synthpanel

      - name: Run panel against current instrument
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          SYNTH_PANEL_DATA_DIR: ${{ github.workspace }}/.synthpanel
        run: |
          set -euo pipefail
          mkdir -p "$SYNTH_PANEL_DATA_DIR/results"
          synthpanel --model haiku panel run \
            --personas instrument-regression/personas.yaml \
            --instrument instrument-regression/survey.yaml \
            --temperature 0.3 \
            --save 2>&1 | tee run.log
          rid=$(grep -oE 'result-[0-9-]+-[a-f0-9]+' run.log | head -1)
          echo "RESULT_PATH=$SYNTH_PANEL_DATA_DIR/results/${rid}.json" >> "$GITHUB_ENV"

      - name: Compare against baseline
        run: |
          python instrument-regression/check_drift.py \
            instrument-regression/baseline.json \
            "$RESULT_PATH"

      - name: Upload diff artifacts on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: panel-result-${{ github.run_id }}
          path: |
            ${{ env.RESULT_PATH }}
            run.log
```

The job triggers only when the instrument, personas, baseline, or
workflow itself changes. That keeps the recurring cost predictable —
unrelated PRs don't burn API budget.

## Cost considerations

This pattern is *recurring* spend, so the model choice matters more
than for a one-off study.

- **Use cheap models in CI.** `claude-haiku-4-5`, `gemini-2.5-flash`, or
  `grok-3-mini` give you stable distributions for regression detection
  at roughly an order of magnitude less cost than frontier models.
  Reserve Sonnet/Opus for ad-hoc deep studies, not the regression suite.
- **Keep the panel small.** Three to five personas is plenty for a
  regression smoke test — what you're detecting is *systematic* drift in
  question wording, not subtle individual-level effects.
- **Trigger on path changes only.** The workflow above runs only when
  files inside `instrument-regression/` change, not on every commit.

A 3-persona × 4-question Haiku run is well under $0.05. Even if every
PR touches the instrument, you'd struggle to spend $5/month on this.

## Determinism caveats

LLM panels are not deterministic, but you can dampen the variance:

- **Pin everything you can.** `--model haiku` (a specific version
  alias), `--temperature 0.3`, the exact persona YAML. Anything you
  leave to defaults can shift between runs.
- **Pick thresholds empirically.** Run the panel three or four times
  against the unchanged baseline to see the natural noise floor for
  your instrument. Set `JSD_THRESHOLD` comfortably above that floor —
  starting at `0.20` and tuning down once you have data is a reasonable
  default for a 3-persona panel.
- **Seeds aren't a silver bullet.** Even providers that accept a
  request-level `seed` parameter only treat it as best-effort. If you
  need stronger determinism, raise the panel size and lower the
  threshold instead.
- **Re-baseline deliberately.** When you intentionally change a
  question, regenerate the baseline JSON and commit it in the same PR
  as the instrument change. That makes the diff auditable: anyone
  reviewing the PR sees both the question delta and the resulting
  distribution shift.

## Going further

- **Multi-model regression.** Run the same panel under `--models
  haiku:0.5,gemini-flash:0.5` and diff per-model. This catches drift
  that's specific to one provider's update cycle.
- **Calibration as the source of truth.** If you have an external
  human dataset, use `synthpanel pack calibrate` to lock JSD against a
  real-world distribution rather than a synthetic baseline. The same
  CI pattern applies — just diff against the calibration target
  instead.
- **Theme taxonomies for stricter text checks.** Free-text drift is
  the noisiest signal. Switch the text questions to
  `response_schema: tagged_themes` with a fixed taxonomy so the diff
  is over a closed set of tags rather than over freeform synthesizer
  output.

That's the whole loop: an instrument committed to git, a baseline
committed to git, a small panel run on every change, and a numeric
threshold separating "fine" from "look at this." Same shape as a
unit test suite, applied to research design.
