"""Check that synthpanel's top-level subcommand list matches the site coverage manifest.

Run after installing the package (pip install -e .) to verify that every
subcommand in `synthpanel --help` is tracked in site/cli-coverage.txt.
Exits 0 on match, 1 on drift, 2 on parse failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "site" / "cli-coverage.txt"


def get_cli_subcommands() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "synth_panel", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    match = re.search(r"positional arguments:\s+\{([^}]+)\}", output, re.DOTALL)
    if not match:
        print("ERROR: Could not parse subcommand list from --help output.", file=sys.stderr)
        print(output, file=sys.stderr)
        sys.exit(2)
    return set(match.group(1).split(","))


def read_manifest() -> set[str]:
    lines = MANIFEST_PATH.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def main() -> None:
    cli = get_cli_subcommands()
    manifest = read_manifest()

    only_in_cli = cli - manifest
    only_in_manifest = manifest - cli

    if not only_in_cli and not only_in_manifest:
        print(f"OK: CLI subcommands match manifest ({len(cli)} subcommands)")
        return

    print("DRIFT DETECTED between `synthpanel --help` and site/cli-coverage.txt")
    print()
    if only_in_cli:
        print("New subcommands not in manifest (document on site then add to site/cli-coverage.txt):")
        for cmd in sorted(only_in_cli):
            print(f"  + {cmd}")
    if only_in_manifest:
        print("Stale manifest entries (subcommand removed from CLI; delete from site/cli-coverage.txt):")
        for cmd in sorted(only_in_manifest):
            print(f"  - {cmd}")
    sys.exit(1)


if __name__ == "__main__":
    main()
