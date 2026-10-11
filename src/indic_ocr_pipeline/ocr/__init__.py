"""Local OCR backends. Each backend proposes candidates and can fail alone."""

from indic_ocr_pipeline.ocr.base import OCRBackend
from indic_ocr_pipeline.ocr.chitrapathak import ChitrapathakBackend
from indic_ocr_pipeline.ocr.indicocr import IndicOCRBackend
from indic_ocr_pipeline.ocr.tesseract import TesseractBackend

__all__ = [
    "ChitrapathakBackend",
    "IndicOCRBackend",
    "OCRBackend",
    "TesseractBackend",
]
