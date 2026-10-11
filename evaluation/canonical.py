"""Canonical (frozen) OCR evaluation pipeline configuration.

Default loads Marathi ``canonical_pipeline.yaml`` (``marathi_piecewise_canonical_v1``).
Gujarati uses ``gujarati_canonical_pipeline.yaml`` via ``config_path`` /
``language=`` — geometry constants match Marathi for fair PARSeq comparison;
pipeline_version labels differ so results are never mixed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
_CONFIG_PATH = _CONFIG_DIR / "canonical_pipeline.yaml"
_GUJARATI_CONFIG_PATH = _CONFIG_DIR / "gujarati_canonical_pipeline.yaml"

# Hard defaults mirrored from the YAML (used if file missing; tests pin YAML).
CANONICAL_MODE = "piecewise"
CANONICAL_PIECE_TARGET_ASPECT = 3.0
CANONICAL_JOIN_MODE = "boundary_aware"
CANONICAL_PIECE_OVERLAP_PX = 0
CANONICAL_BATCH_SIZE = 1
CANONICAL_TTA = False
PIPELINE_VERSION = "marathi_piecewise_canonical_v1"
GUJARATI_PIPELINE_VERSION = "gujarati_piecewise_canonical_v1"


def canonical_config_path(language: str = "marathi") -> Path:
    lang = (language or "marathi").strip().lower()
    if lang == "gujarati":
        return _GUJARATI_CONFIG_PATH
    return _CONFIG_PATH


def load_canonical_pipeline_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else _CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Canonical pipeline config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    inf = data.get("inference") or {}
    # Validate required keys
    required = {
        "mode": CANONICAL_MODE,
        "piece_target_aspect": CANONICAL_PIECE_TARGET_ASPECT,
        "join_mode": CANONICAL_JOIN_MODE,
        "piece_overlap_px": CANONICAL_PIECE_OVERLAP_PX,
    }
    for key, expected in required.items():
        if key not in inf:
            raise ValueError(f"{cfg_path.name} missing inference.{key}")
    if float(inf["piece_target_aspect"]) != float(CANONICAL_PIECE_TARGET_ASPECT):
        # Allow YAML to be source of truth if constants drift — sync constants.
        pass
    if int(inf.get("piece_overlap_px", 0)) != 0:
        raise ValueError(
            "Canonical pipeline forbids overlap "
            f"(got piece_overlap_px={inf.get('piece_overlap_px')})"
        )
    if str(inf.get("join_mode")) != CANONICAL_JOIN_MODE:
        raise ValueError(
            f"Canonical join_mode must be {CANONICAL_JOIN_MODE!r}, "
            f"got {inf.get('join_mode')!r}"
        )
    if str(inf.get("mode")) != CANONICAL_MODE:
        raise ValueError(
            f"Canonical mode must be {CANONICAL_MODE!r}, got {inf.get('mode')!r}"
        )
    return data


def canonical_inference_settings(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    language: str = "marathi",
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_canonical_pipeline_config(canonical_config_path(language))
    data = cfg
    inf = data["inference"]
    default_pv = (
        GUJARATI_PIPELINE_VERSION
        if (language or "").lower() == "gujarati"
        else PIPELINE_VERSION
    )
    return {
        "mode": str(inf["mode"]),
        "piece_target_aspect": float(inf["piece_target_aspect"]),
        "join_mode": str(inf["join_mode"]),
        "piece_overlap_px": int(inf["piece_overlap_px"]),
        "batch_size": int(inf.get("batch_size", CANONICAL_BATCH_SIZE)),
        "tta": bool(inf.get("tta", CANONICAL_TTA)),
        "pipeline_version": str(data.get("pipeline_version") or default_pv),
        "metric_version": data.get("metric_version"),
        "normalization_version": data.get("normalization_version"),
        "policies": data.get("policies") or {},
        "reference_baseline": data.get("reference_baseline") or {},
    }


def pipeline_fingerprint(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Stable fields recorded in every summary.json for comparability."""
    s = settings if settings is not None else canonical_inference_settings()
    return {
        "pipeline_version": s["pipeline_version"],
        "mode": s["mode"],
        "piece_target_aspect": s["piece_target_aspect"],
        "join_mode": s["join_mode"],
        "piece_overlap_px": s["piece_overlap_px"],
        "tta": s["tta"],
        "batch_size": s["batch_size"],
        "use_gt_for_inference": False,
        "use_gt_for_segmentation": False,
        "use_gt_for_joining": False,
    }


def apply_canonical_to_namespace(args, language: str = "marathi") -> None:
    """Force argparse namespace to canonical inference settings (in-place)."""
    s = canonical_inference_settings(language=language)
    args.mode = s["mode"]
    args.piece_target_aspect = s["piece_target_aspect"]
    args.join_mode = s["join_mode"]
    args.piece_overlap = s["piece_overlap_px"]
    args.batch_size = s["batch_size"]
    args.tta = s["tta"]
