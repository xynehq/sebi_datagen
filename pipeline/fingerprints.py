"""Fingerprints for resume and invalidation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pipeline.graph import STAGES, Stage
from pipeline.records import records_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_token(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.suffix.lower() == ".pdf":
        stat = path.stat()
        return f"pdf:{stat.st_size}:{stat.st_mtime_ns}"
    if path.is_file():
        return sha256_file(path)
    return "dir"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stage_fingerprints(
    root: Path,
    config: dict[str, Any],
    run_dir: Path | None,
    state: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Return input, config, and current-output hashes for every stage."""
    records = (state or {}).get("stages") or {}
    found: dict[str, dict[str, str]] = {}
    for name, stage in STAGES.items():
        found[name] = {
            "input": _input_hash(root, config, name),
            "config": _config_hash(root, config, stage),
            "output": _output_hash(root, run_dir, stage, records.get(name)),
        }
    return found


def _config_hash(root: Path, config: dict[str, Any], stage: Stage) -> str:
    section = {key: config.get(key) for key in stage.sections}
    digest = hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode())
    for relative in stage.files:
        path = root / relative
        digest.update(relative.encode())
        digest.update(_file_token(path).encode())
    if stage.name in {"ingest", "render", "detect", "crop", "ocr", "compare"}:
        digest.update(b"dataset_size")
        digest.update(str(config.get("dataset_size")).encode())
    if stage.name == "ocr":
        tokens = config.get("chitrapathak_max_new_tokens") or os.environ.get(
            "CHITRAPATHAK_MAX_NEW_TOKENS"
        )
        digest.update(b"chitrapathak_max_new_tokens")
        digest.update(str(tokens).encode())
    if stage.name in {"ocr", "train", "eval_checkpoints", "select", "frozen"}:
        local = root / "configs" / "local.yaml"
        digest.update(b"local")
        digest.update(_file_token(local).encode())
    return digest.hexdigest()


def _output_hash(
    root: Path,
    run_dir: Path | None,
    stage: Stage,
    record: dict[str, Any] | None,
) -> str:
    if not stage.outputs:
        return "none"
    base: Path | None
    if _run_scoped(stage.name):
        recorded = None if record is None else record.get("artifact_root")
        recorded_path = Path(recorded) if recorded else None
        outputs_present = recorded_path is not None and all(
            (recorded_path / relative).exists() for relative in stage.outputs
        )
        if outputs_present:
            base = recorded_path
        else:
            base = run_dir
    else:
        base = root
    if base is None:
        return "missing"
    digest = hashlib.sha256()
    for relative in stage.outputs:
        path = base / relative
        digest.update(relative.encode())
        digest.update(_file_token(path).encode())
    return digest.hexdigest()


def _run_scoped(name: str) -> bool:
    return name in {"train", "eval_checkpoints", "select", "frozen", "report"}


def _input_hash(root: Path, config: dict[str, Any], name: str) -> str:
    paths = config.get("paths") or {}
    source = (config.get("source") or {}).get("documents_dir", "data/documents")
    mapping = {
        "preflight": [root / "configs" / "pipeline.yaml"],
        "ingest": sorted((root / source).glob("*.pdf")) if (root / source).is_dir() else [],
        "render": [records_dir(root) / "documents.jsonl"],
        "detect": [records_dir(root) / "pages.jsonl"],
        "crop": [records_dir(root) / "regions.jsonl"],
        "ocr": [records_dir(root) / "crops.jsonl"],
        "compare": [records_dir(root) / "predictions.jsonl", records_dir(root) / "crops.jsonl"],
        "calibrate": [records_dir(root) / "consensus.jsonl"],
        "export": [
            records_dir(root) / "consensus.jsonl",
            records_dir(root) / "verified.jsonl",
            root / "data" / "review" / "archive" / "invalidated_phase10" / "invalidated_ids.json",
        ],
        "train": [
            root / paths.get("train_manifest", "data/datasets/train.jsonl"),
            root / paths.get("val_manifest", "data/datasets/val.jsonl"),
        ],
        "eval_checkpoints": [root / paths.get("val_manifest", "data/datasets/val.jsonl")],
        "select": [root / paths.get("val_manifest", "data/datasets/val.jsonl")],
        "frozen": [root / paths.get("frozen_benchmark", "data/eval_data") / "validation.jsonl"],
        "report": [root / paths.get("frozen_benchmark", "data/eval_data") / "validation.jsonl"],
    }
    digest = hashlib.sha256()
    for path in mapping[name]:
        if path.is_absolute() and str(path).startswith(str(root)):
            token = str(path.relative_to(root))
        else:
            token = str(path)
        digest.update(token.encode())
        digest.update(_file_token(path).encode())
    return digest.hexdigest()
