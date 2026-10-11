#!/usr/bin/env python3
"""Render ingested PDF pages to PNG images. Source PDFs are not modified."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indic_ocr_pipeline.rendering.pages import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
