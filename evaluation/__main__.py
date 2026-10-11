"""Canonical evaluation entrypoint.

    python -m evaluation score --expected a.txt --predicted b.txt
    python -m evaluation score --pairs pairs.jsonl --output metrics.json

``pairs.jsonl`` rows use ``expected_text`` and ``prediction``.
Checkpoint loading stays in ``evaluation.inference``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.harness import score_corpus


def _read_pairs(path: Path) -> tuple[list[str], list[str]]:
    expected: list[str] = []
    predicted: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        expected.append(str(row["expected_text"]))
        predicted.append(str(row.get("prediction") or row.get("predicted_text") or ""))
    return expected, predicted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score paired strings with the canonical harness")
    score.add_argument("--pairs", type=Path, required=True)
    score.add_argument("--output", type=Path, default=None)

    frozen = sub.add_parser(
        "frozen",
        help="Evaluate one checkpoint on a frozen benchmark directory",
    )
    frozen.add_argument("--checkpoint", required=True)
    frozen.add_argument("--dataset", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    frozen.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)
    if args.command == "score":
        expected, predicted = _read_pairs(args.pairs)
        report = score_corpus(expected, predicted)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "frozen":
        from evaluation.canonical import pipeline_fingerprint
        from evaluation.dataset import load_validation_dataset
        from evaluation.inference import load_model, recognise_canonical, resolve_device

        info = resolve_device(args.device)
        model = load_model(args.checkpoint, info.device)
        samples = load_validation_dataset(args.dataset)
        expected = [sample.expected_text for sample in samples]
        predicted = [
            recognise_canonical(model, info.device, sample.image_path).prediction
            for sample in samples
        ]
        report = score_corpus(expected, predicted)
        report["benchmark"] = "frozen100"
        report["used_for_selection"] = False
        report["checkpoint"] = args.checkpoint
        report["mode"] = "piecewise"
        report["pipeline"] = pipeline_fingerprint()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
