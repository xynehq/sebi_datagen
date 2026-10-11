#!/usr/bin/env python3
"""Score one checkpoint on the frozen Marathi exam.

The exam is ``data/eval_data``. It is not used to train or to choose
the checkpoint. Checkpoint choice uses ``scripts/validation/validate.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "models")]

from evaluation.__main__ import main as evaluation_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate data/eval_data.")
    parser.add_argument("--checkpoint", required=True, help="Selected .ckpt file.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "eval_data")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "manual" / "eval" / "frozen100",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    return evaluation_main(
        [
            "frozen",
            "--checkpoint",
            args.checkpoint,
            "--dataset",
            str(args.dataset),
            "--output",
            str(args.output),
            "--device",
            args.device,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
