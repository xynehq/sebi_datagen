"""Entry point for `python -m pipeline`. The shell calls the same CLI."""

from __future__ import annotations

from pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
