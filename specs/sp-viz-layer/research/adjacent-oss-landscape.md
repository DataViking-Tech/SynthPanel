# Adjacent OSS Landscape — Panel-Result Presentation Patterns

Systematic documentation audit of 12 comparable OSS tools. How each presents run results to non-engineers.

## 1. Great Expectations

- **Report:** Generated HTML validation documentation + interactive reports
- **Packaging:** Core; rendering utilities built-in
- **Positioning:** "Structured results for CI/CD, alerting, dashboards." No "no-dashboard" stance.
- **Input:** In-memory validation → JSON → auto-generated HTML
- **Spectrum:** (b) opinionated default reports

## 2. Prefect

- **Report:** Web dashboard (localhost:4200 self-hosted, or Prefect Cloud)
- **Packaging:** Core; dashboard in server package
- **Positioning:** Real-time flow monitoring through an intuitive interface. Dashboard primary.
- **Input:** Live state + logs via API; no emphasized static file export
- **Spectrum:** (c) host interactive UI

## 3. Dagster

- **Report:** Web UI (Dagster Webserver) for asset lineage, run logs, materializations
- **Packaging:** Core; web UI included
- **Positioning:** "Integrated lineage, observability, unified control plane." UI primary; Slack mentioned alongside.
- **Input:** Live asset state + logs; rendered in webserver
- **Spectrum:** (c) interactive UI

## 4. Weights & Biases

- **Report:** Interactive dashboards + exportable static reports (PDF, LaTeX zip)
- **Packaging:** Core; export via standard UI
- **Positioning:** Document and share AI insights collaboratively. UI-centric.
- **Input:** `run.log()` API → persistent backend → live UI → web-interface exports
- **Spectrum:** (c) interactive UI, optional static export

## 5. MLflow

- **Report:** Web experiment tracking UI (localhost:5000); traces + metrics dashboards
- **Packaging:** Core; MLflow server in base package
- **Positioning:** Track models/parameters/metrics across experiments. UI-centric; no CLI-only emphasis.
- **Input:** `mlflow.log_*` API → local or remote backend → live UI
- **Spectrum:** (c) interactive UI

## 6. Evidently

- **Report:** Interactive reports (Python/Jupyter) + JSON export + HTML + monitoring dashboard
- **Packaging:** Core; Report class and monitoring UI in base
- **Positioning:** Flexible — Python/Jupyter native + dashboard alternative
- **Input:** In-memory dataframes → computed report objects → optional Evidently Cloud backend
- **Spectrum:** (a) → (b) bridge; emits raw JSON/dict by default; interactive HTML reports available

## 7. lm-eval-harness

- **Report:** JSON results files (`results.json`, `<task>_eval_samples.json`) + optional W&B / Zeno / HF Hub integrations
- **Packaging:** Core evaluation in base; viz integrations opt-in (require external auth)
- **Positioning:** Structured output + reproducibility. No built-in dashboard; viz delegated.
- **Input:** Eval run → flat JSON local or cloud; pushed externally on request
- **Spectrum:** (a) emit raw structured data — pure structured-first, viz delegated

## 8. HELM

- **Report:** Web leaderboard + local web UI (`helm-server` localhost:8000) + text summaries (`helm-summarize`)
- **Packaging:** Core; web UI + summarize commands in base install
- **Positioning:** "Holistic, reproducible, transparent evaluation." Multiple paths: leaderboard + local UI + text summary
- **Input:** Eval run → structured data → web + text
- **Spectrum:** (b) opinionated default reports (multi-format)

## 9. Jupyter

- **Report:** Interactive notebook (`.ipynb` JSON) + rich output (HTML, images, LaTeX, MIME) + static exports via nbconvert
- **Packaging:** Notebook viewer core; export via nbconvert separate
- **Positioning:** Open document format based on JSON; rich inline output; ecosystem (Voilà, JupyterLab) bridges to dashboards
- **Input:** Executed cells with inline outputs; `.ipynb` JSON persisted
- **Spectrum:** (a) → (b) bridge

## 10. nbconvert

- **Report:** Static exports (HTML, PDF, LaTeX, Markdown, Python, reST, asciidoc, slides)
- **Packaging:** Separate package (bundled in some distributions)
- **Positioning:** CLI tool for format conversion; pure conversion, no visualization engine
- **Input:** `.ipynb` file → static format
- **Spectrum:** (a) + infrastructure

## 11. Quarto

- **Report:** Multi-format publishing (HTML, PDF, Word, ePub, websites, dashboards, books) from single md/notebook source
- **Packaging:** Standalone tool; integrates via Jupyter; core rendering engine included
- **Positioning:** Reproducible, production-quality articles, dashboards, books from one source
- **Input:** Markdown or notebooks → rendered at build time
- **Spectrum:** (b) opinionated report generation (multi-format)

## 12. papermill

- **Report:** Executed `.ipynb` with injected-parameters cell
- **Packaging:** Core; execution + parameter injection in base
- **Positioning:** Execute notebooks with injected parameters; store to S3/Azure/GCS/local. Reproducibility + parameter management.
- **Input:** `.ipynb` + parameter dict → parameterized executed notebook
- **Spectrum:** (a) emit structured data

## Pattern Clusters

**Cluster A — Raw structured data + consumer choice** (lm-eval-harness, papermill): explicitly emit clean JSON/notebook/files; defer viz to consumers (W&B, Zeno, HF Hub, nbconvert, Quarto). CLI-first; minimal or no built-in UI. Tight ecosystem integration.

**Cluster B — Opinionated default reports** (Great Expectations, HELM, Quarto, evidently): ship structured results + one or more default rendered formats. Rendering in core or minimal extras. No live UI hosting; static artifacts or Jupyter-native. Flexible downstream use.

**Cluster C — Interactive dashboard-first** (Prefect, Dagster, MLflow, W&B): live web UI is primary consumption path. Real-time observability, collaborative features, integrated alerting. API access secondary. Often cloud-backed.

**Cluster D — Infrastructure & format-agnostic** (Jupyter, nbconvert, papermill, Quarto): treat notebooks/markdown as canonical structured form. Rendering/export handled by ecosystem tools or format-independent.

## Spectrum observation

Tools without explicit "no-dashboard" positioning tend to offer dashboards. Tools that *do* emphasize structured-data-first (lm-eval-harness, papermill) explicitly avoid hosting and position as CLI-first or notebook-native. Visualization is a strategic choice, not an afterthought, across all tools audited.
