"""``python -m synthpanel`` entry point (deprecated alias).

Delegates to :func:`althing.main.main` so the legacy form keeps working
after the SynthPanel → Althing rename. Prefer ``python -m althing``.
"""

from __future__ import annotations

import sys

from althing.main import main

print(
    "note: 'python -m synthpanel' is deprecated — the tool is now 'althing' "
    "(python -m althing).",
    file=sys.stderr,
)
sys.exit(main())
