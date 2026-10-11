"""Output checks used for resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.dataset_flow import CountError, assert_ocr_counts, read_jsonl
from pipeline.graph import STAGES


def output_base(root: Path, run_dir: Path | None, name: str, record: dict[str, Any] | None) -> Path:
    if name not in {"train", "eval_checkpoints", "select", "frozen", "report"}:
        return root
    recorded = None if record is None else record.get("artifact_root")
    if recorded:
        base = Path(recorded)
        if all((base / relative).is_file() for relative in STAGES[name].outputs):
            return base
    return run_dir or root


def _backend_count(root: Path, predictions: list[dict[str, Any]]) -> int:
    registry = root / "models" / "cache" / "resolved.json"
    if registry.is_file():
        import json

        payload = json.loads(registry.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return len(payload)
    del predictions
    return 3


def outputs_ready(
    root: Path,
    run_dir: Path | None,
    name: str,
    record: dict[str, Any] | None,
) -> bool:
    stage = STAGES[name]
    if not stage.outputs:
        return True
    base = output_base(root, run_dir, name, record)
    if not all((base / relative).is_file() for relative in stage.outputs):
        return False
    if name == "ingest":
        return len(read_jsonl(root / "manifests" / "documents.jsonl")) > 0
    if name == "render":
        return len(read_jsonl(root / "manifests" / "pages.jsonl")) > 0
    if name == "detect":
        return len(read_jsonl(root / "manifests" / "regions.jsonl")) > 0
    if name == "crop":
        lines = len(read_jsonl(root / "manifests" / "lines.jsonl"))
        crops = len(read_jsonl(root / "manifests" / "crops.jsonl"))
        return lines > 0 and lines == crops
    if name == "ocr":
        crops = len(read_jsonl(root / "manifests" / "crops.jsonl"))
        predictions = read_jsonl(root / "manifests" / "predictions.jsonl")
        return crops > 0 and len(predictions) == crops * _backend_count(root, predictions)
    if name == "compare":
        crops = len(read_jsonl(root / "manifests" / "crops.jsonl"))
        predictions = read_jsonl(root / "manifests" / "predictions.jsonl")
        candidates = len(read_jsonl(root / "manifests" / "consensus.jsonl"))
        try:
            assert_ocr_counts(
                crops,
                len(predictions),
                candidates,
                backends=_backend_count(root, predictions),
            )
        except CountError:
            return False
        return True
    return True
