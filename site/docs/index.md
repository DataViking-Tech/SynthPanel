# Documentation — althing · Run synthetic focus groups with any LLM

[DataViking](https://dataviking.tech)

Docs · Index

# Documentation

Everything for running synthetic focus groups with althing — the CLI reference and MCP server guide live on this site; the full docs set (methodology, response contract, production operations, and more) is maintained in the GitHub repo.

## On this site

### [`panel run` CLI reference](/docs/panel-run)

→

The full flag reference — multi-model ensembles, convergence auto-stop, checkpointing & resume, cost gates, and SynthBench calibration.

### [MCP server](/mcp)

→

Drop-in config for Claude Code, Cursor, Windsurf, Zed, and Claude Desktop — 12 tools, sampling mode, determinism, observability, and troubleshooting.

### [Pack calibration](/docs/calibration)

→

Calibrate a persona pack against a known human baseline with `althing pack calibrate` — JSD interpretation and methodology.

### [Recommended models](/recommended-models)

→

SynthBench-validated model picks by use case, and the `--best-model-for` auto-selector.

## Full docs set on GitHub

The canonical, always-current documentation lives in the repo’s [`docs/` directory](https://github.com/DataViking-Tech/Althing/tree/main/docs). Highlights:

- [agent-quickstart.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/agent-quickstart.md) — end-to-end walkthrough: install → verify → dry-run → run → save → emit JSON, on both CLI and MCP surfaces.

- [mcp.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/mcp.md) — full MCP tool schemas, resources, prompt templates, and the model-resolution order.

- [response-contract.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/response-contract.md) — the frozen v1.0.0 response envelope, field by field (`panel_verdict`, typed errors, `schema_version`).

- [structured-polling.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/structured-polling.md) — schema-enforced polling patterns (pick-one, Likert, confidence, tagged themes), the agent default.

- [production-operations.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/production-operations.md) — running panels in production: checkpointing, cost gates, failure budgets, and observability.

- [methodology.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/methodology.md) — how panels are constructed, synthesized, and where synthetic research is (and isn’t) trustworthy.

- [ensemble.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/ensemble.md) — multi-model ensemble blending (`--models` / `--blend`) and the public ensemble API.

- [reproducibility.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/reproducibility.md) — seeds, model pinning, and what “reproducible” can and cannot mean across providers.

- [model-packs.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/model-packs.md) — bundled persona and instrument packs, and how to author your own.

- [registry.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/registry.md) — the decentralized pack registry: `pack import gh:<user>/<repo>`, search, and verification.

- [migration-v1.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/migration-v1.md) — migrating from v0.12 to the frozen v1.0.0 contract.

- [synthbench-integration.md](https://github.com/DataViking-Tech/Althing/blob/main/docs/synthbench-integration.md) — inline calibration against SynthBench baselines and opt-in leaderboard submission.
