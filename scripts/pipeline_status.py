#!/usr/bin/env python3
"""Print manifest counts and which pipeline stages are implemented."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indic_ocr_pipeline.stages import run_status  # noqa: E402
from indic_ocr_pipeline.utils.logging import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show dataset pipeline status.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return run_status()


if __name__ == "__main__":
    sys.exit(main())
