"""Script entry points run without a network or a PDF."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_status_script_and_unimplemented_stages(repo_root: Path) -> None:
    status = subprocess.run(
        [sys.executable, "scripts/pipeline_status.py"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert status.returncode == 0
    assert "implemented_phase: 12" in status.stdout

    ocr = subprocess.run(
        [sys.executable, "scripts/run_ocr.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ocr.returncode == 0
    assert "gold" in ocr.stdout
