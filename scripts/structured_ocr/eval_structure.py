#!/usr/bin/env python3
"""Structure + recognition metrics on predictions JSONL or synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.structured_ocr.metrics import evaluate_structure, looks_column_major_dump
from training.structured_ocr.rewards import score_verifiable_output
from training.structured_ocr.synthetic import column_major_bad_prediction, holiday_table_reference
from training.structured_ocr.targets import validate_target


def _load_preds(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_synthetic() -> dict:
    ref = holiday_table_reference(rows=3, language="guj")
    bad = column_major_bad_prediction(3)
    good_reward = score_verifiable_output(ref, ref, expect_table_rows=3)
    bad_reward = score_verifiable_output(ref, bad, expect_table_rows=3)
    return {
        "reference_valid": validate_target(ref, expect_table_rows=3).ok,
        "good_self_reward": good_reward.total,
        "bad_column_dump_reward": bad_reward.total,
        "bad_column_dump_detected": looks_column_major_dump(bad),
        "bad_metrics": evaluate_structure(ref, bad).__dict__,
    }


def run_preds(path: Path) -> dict:
    rows = _load_preds(path)
    if not rows:
        return {"error": "empty predictions"}
    agg = {
        "n": len(rows),
        "cer_sum": 0.0,
        "exact": 0,
        "column_major": 0,
        "table_row_match": 0,
        "html_valid": 0,
    }
    for row in rows:
        ref = row.get("expected_text") or row.get("text") or row.get("target") or ""
        pred = row.get("pred") or row.get("prediction") or ""
        m = evaluate_structure(ref, pred)
        agg["cer_sum"] += m.cer
        agg["exact"] += int(m.exact_match)
        agg["column_major"] += int(m.column_major_dump)
        agg["table_row_match"] += int(m.table_row_count_match)
        agg["html_valid"] += int(m.html_valid)
    agg["cer_mean"] = agg["cer_sum"] / len(rows)
    del agg["cer_sum"]
    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, default=None, help="Predictions JSONL")
    parser.add_argument("--synthetic", action="store_true", help="Run deterministic synthetic checks")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report: dict = {}
    if args.synthetic:
        report["synthetic"] = run_synthetic()
    if args.preds:
        report["predictions"] = run_preds(args.preds)
    if not report:
        report["synthetic"] = run_synthetic()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
