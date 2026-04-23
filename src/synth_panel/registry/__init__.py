"""Pack registry support.

Resolves remote pack sources (`gh:`, `github.com/blob/`,
`raw.githubusercontent.com`) to raw-content URLs. Fetch + cache
layers live in sibling modules (added in later tickets).
"""

from synth_panel.registry.github import (
    DEFAULT_PATH,
    DEFAULT_REF,
    GitHubSource,
    parse_gh_source,
    resolve_source,
)

__all__ = [
    "DEFAULT_PATH",
    "DEFAULT_REF",
    "GitHubSource",
    "parse_gh_source",
    "resolve_source",
]
