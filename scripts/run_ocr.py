#!/usr/bin/env python3
"""Run local OCR on full-line crops. Predictions are candidates, not gold labels."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", _ROOT / "models"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from indic_ocr_pipeline.ocr.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
