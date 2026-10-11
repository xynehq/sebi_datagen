#!/usr/bin/env python3
"""Detect text regions on rendered page images. Phase 4."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indic_ocr_pipeline.detection.regions import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
