"""Load the canonical pipeline config and optional local overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "pipeline.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing pipeline config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    run_path = root / "configs" / "run.yaml"
    if run_path.is_file():
        overlay = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
        if isinstance(overlay, dict):
            loaded = _merge(loaded, overlay)
    local = root / "configs" / "local.yaml"
    if local.is_file():
        overlay = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        if isinstance(overlay, dict):
            loaded = _merge(loaded, overlay)
    paths = loaded.setdefault("paths", {})
    training = loaded.setdefault("training", {})
    evaluation = loaded.setdefault("evaluation", {})
    gpu = loaded.setdefault("gpu", {})
    paths.setdefault("data_dir", "data")
    paths.setdefault(
        "frozen_benchmark",
        evaluation.get("frozen_benchmark", "data/eval_data"),
    )
    paths.setdefault("train_manifest", training.get("train_manifest", "data/datasets/train.jsonl"))
    paths.setdefault("val_manifest", training.get("val_manifest", "data/datasets/val.jsonl"))
    if "seed" not in loaded and "random_seed" in loaded:
        loaded["seed"] = loaded["random_seed"]
    if os.environ.get("DATA_ROOT"):
        paths["data_dir"] = os.environ["DATA_ROOT"]
    if os.environ.get("MODEL_ROOT"):
        training["model_root"] = os.environ["MODEL_ROOT"]
    if os.environ.get("CHECKPOINT"):
        training["base_checkpoint"] = os.environ["CHECKPOINT"]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return loaded


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
