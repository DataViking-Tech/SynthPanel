#!/usr/bin/env bash
# postStartCommand: runs on EVERY container start.
# Ensures synthpanel is ready and prints status.
set -euo pipefail

# Ensure every new terminal activates the venv
BASHRC_SNIPPET='# synthpanel venv activation
if [ -d "/workspaces/synthpanel/.venv" ]; then
  export VIRTUAL_ENV="/workspaces/synthpanel/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
export PYTHONPATH="${PYTHONPATH:-/workspaces/synthpanel/src}"'

if ! grep -q "synthpanel venv activation" ~/.bashrc 2>/dev/null; then
  echo "$BASHRC_SNIPPET" >> ~/.bashrc
fi

# Activate for this script too
if [ -d "/workspaces/synthpanel/.venv" ]; then
  export VIRTUAL_ENV="/workspaces/synthpanel/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
fi

export PYTHONPATH="${PYTHONPATH:-/workspaces/synthpanel/src}"

# Quick health check
if python3 -m synth_panel --help >/dev/null 2>&1; then
  echo "synthpanel is ready"
else
  echo "Warning: synthpanel not working, try: uv sync"
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
