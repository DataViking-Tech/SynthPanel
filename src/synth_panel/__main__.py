"""Allow running synth-panel as ``python -m synth_panel``."""

from __future__ import annotations

import sys

from synth_panel.main import main

sys.exit(main())
