# syntax=docker/dockerfile:1.23
#
# Production runtime image for synthpanel.
#
# This image is sized for ephemeral / serverless invocation (Lambda, Cloud Run,
# GitHub Actions, n8n, etc.) — small footprint, no editor tooling. The
# devcontainer at .devcontainer/devcontainer.json remains the source of truth
# for *development*; this file is a slimmed-down production sibling.
#
# Default CMD launches the MCP server on stdio. Override with any synthpanel
# subcommand:
#
#   docker run --rm -e ANTHROPIC_API_KEY=$KEY synthpanel/synthpanel \
#     prompt "Say hello"
#
# Build:
#   docker build -t synthpanel:local .
#
# Multi-arch builds happen in CI via .github/workflows/docker.yml.

ARG PYTHON_VERSION=3.12

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copy only what's needed to build the wheel. Source layout matches
# pyproject.toml's [tool.setuptools.packages.find] (where = ["src"]).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip build \
 && python -m build --wheel --outdir /build/dist

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG SYNTHPANEL_VERSION=unknown
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="synthpanel" \
      org.opencontainers.image.description="Run synthetic focus groups using AI personas. CLI, Python library, and MCP server." \
      org.opencontainers.image.url="https://synthpanel.dev" \
      org.opencontainers.image.documentation="https://github.com/DataViking-Tech/SynthPanel#readme" \
      org.opencontainers.image.source="https://github.com/DataViking-Tech/SynthPanel" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="DataViking-Tech" \
      org.opencontainers.image.version="${SYNTHPANEL_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user — Lambda and Cloud Run both honor USER, and avoiding root
# is good practice for any ephemeral container. UID 1000 (not --system) so
# bind mounts from host UID 1000 are writable in dev workflows.
RUN groupadd --gid 1000 synthpanel \
 && useradd --uid 1000 --gid synthpanel --create-home --home-dir /home/synthpanel synthpanel

COPY --from=builder /build/dist/*.whl /tmp/

# Install with [mcp] extras so `mcp-serve` works out of the box. The MCP
# server is the primary use case for this image (agent tool-call target).
# The shell-form RUN resolves the wheel via a variable so the [mcp] extras
# suffix doesn't get interpreted as a shell character class against the
# glob.
RUN set -eux \
 && wheel="$(ls /tmp/synthpanel-*.whl | head -n1)" \
 && pip install "${wheel}[mcp]" \
 && rm -f /tmp/synthpanel-*.whl

USER synthpanel
WORKDIR /home/synthpanel

# stdin must stay open for the MCP stdio transport. Callers running
# non-MCP subcommands can ignore this — it's only meaningful for mcp-serve.
ENTRYPOINT ["synthpanel"]
CMD ["mcp-serve"]
