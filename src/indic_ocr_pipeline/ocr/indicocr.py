"""Re-export the IndicOCR engine that lives in ``models/IndicPhotoOCR``."""

from models.IndicPhotoOCR.backend import (
    IndicOCRBackend,
    _offline_hub,
    _release_device,
    _require_device,
)

__all__ = ["IndicOCRBackend", "_offline_hub", "_release_device", "_require_device"]
