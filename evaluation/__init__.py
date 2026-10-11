"""Marathi OCR evaluation package.

METRIC_VERSION and NORMALIZATION_VERSION are frozen identifiers recorded in
every summary.json so future checkpoints are comparable under the same
definitions.

PIPELINE_VERSION identifies the canonical inference geometry (piecewise AR=3,
boundary-aware join, zero overlap). See PIPELINE.md.
"""

from .canonical import PIPELINE_VERSION

METRIC_VERSION = "marathi_ocr_eval_v2"
NORMALIZATION_VERSION = "strict_v1+nfc_ws_diagnostic_v1"

__all__ = ["METRIC_VERSION", "NORMALIZATION_VERSION", "PIPELINE_VERSION"]
