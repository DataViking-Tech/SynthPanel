#!/usr/bin/env bash
# Verify that the LATEST CI check run per check name passed on a commit.
#
# Called by auto-tag.yml's "Verify CI checks passed" step, and runnable
# directly (read-only; needs only `gh` auth) to debug any SHA:
#
#   REPO=owner/name HEAD_SHA=<sha> bash .github/scripts/verify-ci-checks.sh
#
# Why "latest per (app, name)" instead of "every run on the SHA": the
# check-runs API returns runs from EVERY check suite ever started on the
# SHA. Two triggers landing on the same SHA — e.g. `gh pr create --label`
# fires `opened` and `labeled` simultaneously, and the CI workflow's
# concurrency group cancels one of the twin runs — park a cancelled or
# failed run next to the successful same-named run, and those superseded
# runs never turn green. Counting all of them blocked SynthBench v0.3.1
# (PR #334) and Althing's first v1.6.0 attempt, forcing manual tags.
# Only the newest run per (app, name) reflects the commit's real status;
# GitHub's own merge box collapses check runs the same way.
#
# CHECK_RUNS_FILTER=all additionally surfaces superseded ATTEMPTS of
# re-run check runs (the API default, `latest`, already collapses those).
# Useful only for reproducing historical states; the verdict grouping
# makes both filters converge on the same latest-run answer.

set -euo pipefail

: "${REPO:?REPO (owner/name) is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"
FILTER="${CHECK_RUNS_FILTER:-latest}"

echo "Verifying CI status for commit: $HEAD_SHA (filter=${FILTER})"

# --paginate emits one JSON array per page; `jq -s 'add'` flattens them.
# The auto-tag job's own check run is excluded, as before — a prior
# failed tagging attempt on the SHA must not block the recovery re-run.
ALL_RUNS=$(gh api "repos/${REPO}/commits/${HEAD_SHA}/check-runs?filter=${FILTER}" \
  --paginate \
  --jq '[.check_runs[] | select(.name != "auto-tag")
         | {name, status, conclusion, started_at, id, app: (.app.slug // "unknown")}]' \
  | jq -s 'add // []')

TOTAL=$(echo "$ALL_RUNS" | jq 'length')
if [ "$TOTAL" -eq 0 ]; then
  echo "::error::No CI check runs found for commit $HEAD_SHA — refusing to tag"
  exit 1
fi

# Latest run per (app, name): started_at orders the runs; id breaks
# same-second ties (a later-created run always has a higher id).
RUNS=$(echo "$ALL_RUNS" | jq 'group_by([.app, .name]) | map(sort_by(.started_at, .id) | last)')

KEPT=$(echo "$RUNS" | jq 'length')
if [ "$KEPT" -lt "$TOTAL" ]; then
  echo "Ignoring $((TOTAL - KEPT)) superseded run(s) shadowed by a newer run of the same check:"
  echo "$ALL_RUNS" | jq 'group_by([.app, .name]) | map(sort_by(.started_at, .id) | .[:-1]) | add'
fi

echo "Latest check run per name:"
echo "$RUNS" | jq '.'

INCOMPLETE=$(echo "$RUNS" | jq '[.[] | select(.status != "completed")] | length')
if [ "$INCOMPLETE" -gt 0 ]; then
  echo "::error::${INCOMPLETE} check(s) still running — refusing to tag"
  echo "$RUNS" | jq '[.[] | select(.status != "completed")]'
  exit 1
fi

FAILED=$(echo "$RUNS" | jq '[.[] | select(.conclusion != "success" and .conclusion != "skipped" and .conclusion != "neutral")] | length')
if [ "$FAILED" -gt 0 ]; then
  echo "::error::${FAILED} check(s) did not pass — refusing to tag"
  echo "$RUNS" | jq '[.[] | select(.conclusion != "success" and .conclusion != "skipped" and .conclusion != "neutral")]'
  exit 1
fi

echo "All CI checks passed ✓"
