"""Re-export the Tesseract engine that lives in ``models/tesseract``."""

from models.tesseract.backend import OCRError, TesseractBackend, _parse_tsv, _reject_oversized

__all__ = ["OCRError", "TesseractBackend", "_parse_tsv", "_reject_oversized"]
