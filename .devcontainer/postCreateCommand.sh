#!/usr/bin/env bash
# postCreateCommand: runs ONCE on container creation.
# Sets up synth-panel for development and testing.
set -euo pipefail

echo "Setting up synth-panel development environment..."

cd /workspaces/synth-panel

# Initialize uv venv if one doesn't already exist
if [ ! -d ".venv" ]; then
  echo "Creating uv virtual environment..."
  uv venv .venv
fi

# Activate the venv so subsequent commands target it
export VIRTUAL_ENV="/workspaces/synth-panel/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Sync dependencies if pyproject.toml has them, otherwise install editable
if uv sync 2>/dev/null; then
  echo "uv sync complete"
else
  uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e . 2>/dev/null || {
    echo "uv pip install failed, falling back to PYTHONPATH mode"
  }
fi

# Install MCP SDK if available
uv pip install mcp 2>/dev/null || echo "MCP SDK not available, mcp-serve will not work"

# Verify CLI works
if PYTHONPATH=src python3 -m synth_panel --help >/dev/null 2>&1; then
  echo "synth-panel CLI is working"
else
  echo "Warning: synth-panel CLI failed to load"
fi

echo "synth-panel dev environment ready"
echo ""
echo "Quick start:"
echo "  export ANTHROPIC_API_KEY='sk-...'"
echo "  synth-panel prompt 'Say hello'"
echo "  synth-panel panel run --personas examples/personas.yaml --instrument examples/survey.yaml"
echo ""
echo "MCP server:"
echo "  synth-panel mcp-serve"
