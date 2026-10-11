#!/usr/bin/env python3
"""Generate full and partial crops without assigning labels. Phase 5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indic_ocr_pipeline.stages import unavailable_stage  # noqa: E402
from indic_ocr_pipeline.utils.logging import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate full and partial crops without assigning labels. Phase 5."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return unavailable_stage("crops")


if __name__ == "__main__":
    sys.exit(main())
