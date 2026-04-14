# Contributing to synthpanel

## Development Setup

```bash
# Clone the repository
git clone https://github.com/DataViking-Tech/synthpanel.git
cd synthpanel

# Create a virtual environment (using uv or standard venv)
uv venv .venv && source .venv/bin/activate
# or: python3 -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# For MCP server development, also install MCP deps
pip install -e ".[dev,mcp]"

# Install pre-commit hooks (runs ruff auto-fixes on commit)
pip install pre-commit && pre-commit install
```

## Running Tests

```bash
# Run all unit tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_orchestrator.py

# Run acceptance tests (requires a live API key)
ANTHROPIC_API_KEY=sk-... pytest tests/test_acceptance.py -m acceptance
```

## Linting

```bash
# Check for lint issues
ruff check src/ tests/

# Auto-fix fixable issues
ruff check --fix src/ tests/
```

## Type Checking

```bash
mypy src/synth_panel/
```

## Running the MCP Server Locally

The MCP server uses stdio transport and is designed to be launched by an MCP-aware editor (Claude Code, Cursor, Windsurf, etc.):

```bash
synthpanel mcp-serve
```

For testing outside an editor, you can pipe JSON-RPC messages to stdin. See [docs/mcp.md](docs/mcp.md) for the full tool/resource/prompt reference.

## Running Without Installing

```bash
PYTHONPATH=src python3 -m synth_panel prompt "Hello"
```

## Project Structure

```
src/synth_panel/
├── llm/              # Provider-agnostic LLM client
│   ├── client.py     # Unified send/stream interface
│   ├── aliases.py    # Model alias resolution
│   ├── providers/    # Anthropic, OpenAI, xAI, Gemini
├── runtime.py        # Agent session loop
├── orchestrator.py   # Parallel panelist execution
├── structured/       # Schema-validated responses
├── cost.py           # Token tracking and pricing
├── persistence.py    # Session save/load/fork
├── plugins/          # Manifest-based extension system
├── instrument.py     # v1/v2/v3 instrument parser + DAG validator
├── routing.py        # v3 router predicates
├── mcp/              # MCP server (12 tools, stdio)
├── packs/            # Bundled persona and instrument packs
├── cli/              # CLI framework
└── main.py           # Entry point
```

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Add tests for new functionality.
3. Run the test suite: `pytest tests/`
4. Run lint: `ruff check src/ tests/`
5. Open a pull request against `main`.

### Commit Message Conventions

Use conventional-style prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `ci:` — CI/CD pipeline changes

Example: `feat: add --timeout flag to panel run`

## Release Process

Releases are published to PyPI via GitHub Actions:

1. A maintainer applies a `semver:patch`, `semver:minor`, or `semver:major` label to a merged PR.
2. The `auto-tag.yml` workflow creates a git tag (e.g., `v0.4.1`).
3. The `publish.yml` workflow builds and publishes to PyPI using trusted publishing.

See the [Versions table in README.md](README.md#versions) for release history.

## Sign-off

All contributions require a DCO sign-off. Use `git commit -s` to add a `Signed-off-by:` trailer. This certifies you have the right to submit the contribution under the project's MIT license. See https://developercertificate.org/.
