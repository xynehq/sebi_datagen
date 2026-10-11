"""Shipped YAML configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from indic_ocr_pipeline.config import _build_settings, load_settings
from indic_ocr_pipeline.errors import ConfigError


def test_shipped_configuration(repo_root: Path) -> None:
    settings = load_settings(repo_root, environ={})
    assert settings.render.dpi == 200
    assert settings.render.allowed_dpi == (150, 200, 250, 300, 400)
    assert settings.render.overwrite is False
    assert settings.render.pdf_text_layer_role == "weak_diagnostic_not_ground_truth"
    assert settings.safety.allow_network is False
    assert settings.safety.overwrite_existing_outputs is False
    assert settings.random_seed == 3407
    assert settings.cropping.random_seed == 3407
    assert settings.cropping.multi_line is False
    assert settings.cropping.max_partial_per_line == 4
    assert abs(sum(settings.scoring.weights.values()) - 1.0) < 1e-9
    assert settings.scoring.score_meaning == "ranking_not_probability"
    ratios = (
        settings.dataset.split.train,
        settings.dataset.split.val,
        settings.dataset.split.test,
    )
    assert abs(sum(ratios) - 1.0) < 1e-9
    assert set(settings.dataset.gold_statuses) == {"human_verified", "human_corrected"}
    assert all(settings.dataset.quality_gate.values())
    assert settings.ocr.enabled_backends == ("tesseract", "indicocr", "chitrapathak")
    assert settings.ocr.indicocr.model_path is None
    assert settings.ocr.chitrapathak.model_path is None
    assert settings.documents_dir.name == "documents"
    assert settings.review.high_confidence_random_audit == 0.05
    assert settings.review.reviewer_id == "local_reviewer"
    assert settings.review.calibration_batch_size == 600
    assert settings.review.calibration_tier_quota["P0"] is None
    assert settings.review.host in {"127.0.0.1", "localhost"}


def test_environment_overrides_model_paths(repo_root: Path) -> None:
    settings = load_settings(
        repo_root,
        environ={
            "INDICOCR_MODEL_PATH": "/opt/models/indicocr",
            "CHITRAPATHAK_MODEL_PATH": "/opt/models/chitrapathak",
            "TESSERACT_CMD": "/opt/bin/tesseract",
            "MARATHI_OCR_DEVICE": "cuda:0",
        },
    )
    assert settings.ocr.indicocr.model_path == "/opt/models/indicocr"
    assert settings.ocr.chitrapathak.model_path == "/opt/models/chitrapathak"
    assert settings.ocr.tesseract.cmd == "/opt/bin/tesseract"
    assert settings.ocr.tesseract.device == "cpu"
    assert settings.ocr.device == "cuda:0"
    assert settings.ocr.indicocr.device == "cuda:0"
    assert settings.ocr.chitrapathak.device == "cuda:0"


def test_indic_device_overrides_the_older_env_name(repo_root: Path) -> None:
    settings = load_settings(
        repo_root,
        environ={"INDIC_OCR_DEVICE": "cpu", "MARATHI_OCR_DEVICE": "cuda:0"},
    )
    assert settings.ocr.device == "cpu"
    assert settings.ocr.indicocr.device == "cpu"
    assert settings.ocr.tesseract.device == "cpu"


def test_network_access_cannot_be_enabled(repo_root: Path) -> None:
    config_dir = repo_root / "configs"

    def load(name: str) -> dict:
        with (config_dir / name).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        assert isinstance(loaded, dict)
        return loaded

    pipeline = load("pipeline.yaml")
    pipeline["safety"]["allow_network"] = True
    with pytest.raises(ConfigError):
        _build_settings(
            root=repo_root,
            pipeline=pipeline,
            rendering=load("rendering.yaml"),
            detection=load("detection.yaml"),
            cropping=load("cropping.yaml"),
            ocr=load("ocr.yaml"),
            scoring=load("scoring.yaml"),
            review=load("review.yaml"),
            dataset=load("dataset.yaml"),
        )
