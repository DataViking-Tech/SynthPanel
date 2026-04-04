"""Allow running synth-panel as ``python -m synth_panel``."""

import sys

from synth_panel.main import main

sys.exit(main())
