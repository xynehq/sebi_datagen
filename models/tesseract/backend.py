"""Local Tesseract adapter.

Confidence from Tesseract is a review signal. It is not a probability that
the transcription is correct, and it is never a gold label.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from indic_ocr_pipeline.config import BackendSettings
from indic_ocr_pipeline.errors import DatasetError
from indic_ocr_pipeline.schemas import OCRPrediction
from indic_ocr_pipeline.text_policy import apply_storage_normalization


class OCRError(DatasetError):
    """A local OCR backend could not transcribe one crop."""


class TesseractBackend:
    """CPU OCR adapter. One crop at a time. No network."""

    name = "tesseract"

    def __init__(self, settings: BackendSettings) -> None:
        if settings.cmd is None or settings.lang is None or settings.psm is None:
            raise OCRError("tesseract cmd, lang, and psm are required")
        self._cmd = settings.cmd
        self._lang = os.environ.get("OCR_LANG", "").strip() or settings.lang
        self._psm = settings.psm
        self._max_image_size = settings.max_image_size
        self.version = _tesseract_version(settings.cmd)

    def predict(self, image_path: Path) -> OCRPrediction:
        _reject_oversized(image_path, self._max_image_size)
        completed = subprocess.run(
            [
                self._cmd,
                str(image_path),
                "stdout",
                "-l",
                self._lang,
                "--psm",
                str(self._psm),
                "--oem",
                "1",
                "tsv",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit {completed.returncode}"
            raise OCRError(f"tesseract failed for {image_path.name}: {detail}")
        raw_text, confidence = _parse_tsv(completed.stdout)
        return OCRPrediction(
            backend=self.name,
            backend_version=self.version,
            text=apply_storage_normalization(raw_text),
            raw_text=raw_text,
            confidence=confidence,
        )

    def close(self) -> None:
        """Tesseract keeps no model in this process."""


def _tesseract_version(cmd: str) -> str:
    completed = subprocess.run(
        [cmd, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "not found"
        raise OCRError(f"tesseract is not available ({detail})")
    first = completed.stdout.splitlines()
    if not first or not first[0].strip():
        raise OCRError("tesseract did not report a version")
    return first[0].strip()


def _parse_tsv(payload: str) -> tuple[str, float | None]:
    words: list[tuple[int, str, float]] = []
    for line in payload.splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) < 12 or columns[0] != "5":
            continue
        text = columns[11].strip()
        if not text:
            continue
        try:
            left = int(columns[6])
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence < 0:
            continue
        words.append((left, text, confidence))
    words.sort(key=lambda item: (item[0], item[1]))
    raw = " ".join(word for _left, word, _confidence in words)
    if not words:
        return raw, None
    mean = sum(item[2] for item in words) / len(words) / 100.0
    return raw, round(min(1.0, max(0.0, mean)), 4)


def _reject_oversized(image_path: Path, limit: int) -> None:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    if width > limit or height > limit:
        raise OCRError(f"{image_path.name} is {width}x{height}, above max_image_size {limit}")
