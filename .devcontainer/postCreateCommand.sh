#!/usr/bin/env bash
# postCreateCommand: runs ONCE on container creation.
# Sets up synth-panel for development and testing.
set -euo pipefail

echo "Setting up synth-panel development environment..."

cd /workspaces/synth-panel

# Initialize uv project (creates .venv if needed, generates uv.lock)
echo "Initializing uv project..."
uv lock 2>&1 || echo "uv lock failed, continuing..."

# Install the package in editable mode with all extras
echo "Installing synth-panel (editable)..."
uv pip install -e ".[dev,mcp]" 2>&1 || uv pip install -e . 2>&1 || {
  echo "uv pip install failed, falling back to PYTHONPATH mode"
}

# Verify the synth-panel entry point is available
if command -v synth-panel >/dev/null 2>&1; then
  echo "synth-panel CLI installed and on PATH"
elif [ -x ".venv/bin/synth-panel" ]; then
  echo "synth-panel installed in .venv (activate with: source .venv/bin/activate)"
else
  echo "Warning: synth-panel entry point not found"
  echo "  Use: PYTHONPATH=src python3 -m synth_panel"
fi

# Verify CLI works
if synth-panel --help >/dev/null 2>&1 || .venv/bin/synth-panel --help >/dev/null 2>&1; then
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
