#!/usr/bin/env bash
# postCreateCommand: runs ONCE on container creation.
# Sets up synthpanel for development and testing.
set -euo pipefail

echo "Setting up synthpanel development environment..."

cd /workspaces/synthpanel

# Initialize uv project (creates .venv if needed, generates uv.lock)
echo "Initializing uv project..."
uv lock 2>&1 || echo "uv lock failed, continuing..."

# Install the package in editable mode with all extras
echo "Installing synthpanel (editable)..."
uv pip install -e ".[dev,mcp]" 2>&1 || uv pip install -e . 2>&1 || {
  echo "uv pip install failed, falling back to PYTHONPATH mode"
}

# Verify the synthpanel entry point is available
if command -v synthpanel >/dev/null 2>&1; then
  echo "synthpanel CLI installed and on PATH"
elif [ -x ".venv/bin/synthpanel" ]; then
  echo "synthpanel installed in .venv (activate with: source .venv/bin/activate)"
else
  echo "Warning: synthpanel entry point not found"
  echo "  Use: PYTHONPATH=src python3 -m synth_panel"
fi

# Verify CLI works
if synthpanel --help >/dev/null 2>&1 || .venv/bin/synthpanel --help >/dev/null 2>&1; then
  echo "synthpanel CLI is working"
else
  echo "Warning: synthpanel CLI failed to load"
fi

echo "synthpanel dev environment ready"
echo ""
echo "Quick start:"
echo "  export ANTHROPIC_API_KEY='sk-...'"
echo "  synthpanel prompt 'Say hello'"
echo "  synthpanel panel run --personas examples/personas.yaml --instrument examples/survey.yaml"
echo ""
echo "MCP server:"
echo "  synthpanel mcp-serve"
