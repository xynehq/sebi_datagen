"""Tesseract adapter. The binary is installed on the machine, not stored here."""

from models.tesseract.backend import OCRError, TesseractBackend

__all__ = ["OCRError", "TesseractBackend"]
