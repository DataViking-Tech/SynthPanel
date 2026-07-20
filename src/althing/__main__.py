"""Allow running althing as ``python -m althing``."""

from __future__ import annotations

import sys

from althing.main import main

sys.exit(main())
