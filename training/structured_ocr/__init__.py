"""Structured full-page OCR targets, validation, evaluation, and rewards."""

from training.structured_ocr.manifest import assert_training_ready, load_jsonl, validate_manifest_row
from training.structured_ocr.prompts import (
    LINE_OCR_PROMPT,
    STRUCTURED_PAGE_PROMPT,
)
from training.structured_ocr.targets import (
    EMPTY_PAGE_TARGET,
    TargetValidationError,
    normalize_target,
    validate_target,
)

__all__ = [
    "EMPTY_PAGE_TARGET",
    "LINE_OCR_PROMPT",
    "STRUCTURED_PAGE_PROMPT",
    "TargetValidationError",
    "assert_training_ready",
    "load_jsonl",
    "normalize_target",
    "validate_manifest_row",
    "validate_target",
]
