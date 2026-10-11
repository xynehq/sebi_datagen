"""Local IndicOCR adapter.

Weights come from ``INDICOCR_MODEL_PATH`` or ``configs/ocr.yaml``. This module
does not download them. The transcription is a candidate, not a gold label.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

from indic_ocr_pipeline.config import BackendSettings
from indic_ocr_pipeline.schemas import OCRPrediction
from indic_ocr_pipeline.text_policy import apply_storage_normalization
from models.tesseract.backend import OCRError, _reject_oversized

_LINE_TOKENS = 512


class IndicOCRBackend:
    """IndicBlockOCR on one full-line crop. Layout detection is not used."""

    name = "indicocr"

    def __init__(self, settings: BackendSettings) -> None:
        if not settings.model_path:
            raise OCRError(
                "indicocr model_path is empty. Set INDICOCR_MODEL_PATH to a local snapshot."
            )
        snapshot = Path(settings.model_path)
        weights = snapshot / "weights" / "ocr"
        ckpt = weights if (weights / "config.json").is_file() else snapshot
        if not (ckpt / "config.json").is_file():
            raise OCRError(f"indicocr weights were not found at {ckpt}")
        self._max_image_size = settings.max_image_size
        self._device = _require_device(settings.device)
        self._recognizer = _load_recognizer(snapshot, ckpt, settings, self._device)
        self.version = "indic-ocr"

    def predict(self, image_path: Path) -> OCRPrediction:
        _reject_oversized(image_path, self._max_image_size)
        rgb, prompt_type = _load_request(image_path)
        raw_values = self._recognizer.transcribe([prompt_type(rgb, _text_prompt())])
        raw_text = raw_values[0] if raw_values else ""
        return OCRPrediction(
            backend=self.name,
            backend_version=self.version,
            text=apply_storage_normalization(raw_text),
            raw_text=raw_text,
        )

    def close(self) -> None:
        recognizer = getattr(self, "_recognizer", None)
        if recognizer is not None:
            recognizer.close()
        self._recognizer = None
        gc.collect()
        _release_device(self._device)


def _text_prompt() -> str:
    from idp_contract import TEXT_PROMPT

    return str(TEXT_PROMPT)


def _load_request(image_path: Path) -> tuple[object, object]:
    from idp_recognizer import CropRequest
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB").copy()
    return rgb, CropRequest


def _load_recognizer(snapshot: Path, ckpt: Path, settings: BackendSettings, device: str) -> object:
    _offline_hub()
    if str(snapshot) not in sys.path:
        sys.path.insert(0, str(snapshot))
    from idp_recognizer import HfRecognizer
    from idp_types import RecognizerConfig

    return HfRecognizer(
        ckpt=str(ckpt),
        config=RecognizerConfig(max_tokens=_LINE_TOKENS, dtype=_torch_dtype(settings.dtype)),
        device=_device_map(device),
        attn_implementation="sdpa",
        batch_size=settings.batch_size,
    )


def _torch_dtype(name: str) -> str:
    allowed = {"float32", "float16", "bfloat16"}
    if name not in allowed:
        raise OCRError(f"indicocr dtype must be one of: {', '.join(sorted(allowed))}")
    return name


def _device_map(device: str) -> str:
    if device == "mps":
        return "mps"
    return device


def _require_device(device: str) -> str:
    from indic_ocr_pipeline.device import DeviceError, select_device

    try:
        return select_device(device)
    except DeviceError as exc:
        raise OCRError(str(exc)) from exc


def _offline_hub() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def _release_device(device: str) -> None:
    import torch

    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
