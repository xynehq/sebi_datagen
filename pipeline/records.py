"""Stage records live with the rest of the data, not in a separate folder."""

from __future__ import annotations

from pathlib import Path

RECORDS = Path("data") / "output"


def records_dir(root: Path) -> Path:
    """Directory of the JSONL stage records for this repository."""
    return Path(root) / RECORDS
