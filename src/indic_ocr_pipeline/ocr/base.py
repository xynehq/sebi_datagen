"""OCR backend interface.

Implementations return candidates. Calling ``predict`` does not create a
gold label, even when several backends return the same string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from indic_ocr_pipeline.schemas import OCRPrediction


class OCRBackend(ABC):
    """Local OCR adapter. Model paths come from configuration, not from code."""

    name: str
    version: str

    @abstractmethod
    def predict(self, image_path: Path) -> OCRPrediction:
        """Propose a transcription for one crop image."""


class UnavailableOCRBackend(OCRBackend):
    """Placeholder used until a later phase implements the adapter."""

    def __init__(self, name: str, phase: int) -> None:
        self.name = name
        self.version = "unavailable"
        self.phase = phase

    def predict(self, image_path: Path) -> OCRPrediction:
        raise NotImplementedError(
            f"{self.name} adapter is not implemented (phase {self.phase}). "
            "A future prediction would be a candidate, not a gold label. "
            f"Image: {image_path}"
        )

    def close(self) -> None:
        """The placeholder holds no model."""
        return None
