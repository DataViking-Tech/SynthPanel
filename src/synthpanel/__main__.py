"""``python -m synthpanel`` entry point (sy-het).

Delegates to :func:`synth_panel.main.main` so the one-word and two-word forms
behave identically. Closes GH #509.
"""

from __future__ import annotations

import sys

from synth_panel.main import main

sys.exit(main())
