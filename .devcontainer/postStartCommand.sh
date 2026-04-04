#!/usr/bin/env bash
# postStartCommand: runs on EVERY container start.
# Ensures synth-panel is ready and prints status.
set -euo pipefail

# Verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-/workspaces/synth-panel/src}"

# Quick health check
if PYTHONPATH=src python3 -m synth_panel --help >/dev/null 2>&1; then
  echo "synth-panel is ready"
else
  echo "Warning: synth-panel not working, try: pip install -e ."
fi

# Check for API keys
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Anthropic API key is set"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "OpenAI API key is set"
elif [ -n "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]; then
  echo "Google/Gemini API key is set"
else
  echo "No API key set. Export one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY"
fi
