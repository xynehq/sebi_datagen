"""Normalize page images before OCR (vision model and Tesseract)."""

from __future__ import annotations

from PIL import Image, ImageOps, ImageStat


def normalize_document_image(image: Image.Image) -> Image.Image:
    """Invert dark-background scans and boost contrast for line OCR."""
    rgb = image.convert("RGB")
    mean = ImageStat.Stat(rgb.convert("L")).mean[0]
    if mean < 110:
        rgb = ImageOps.invert(rgb)
    return ImageOps.autocontrast(rgb, cutoff=1)
