"""Reference-ID detection and metrics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Pattern, Sequence

import yaml

from .core_metrics import levenshtein

MARATHI_DIGIT = re.compile(r"[०-९]")
GUJARATI_DIGIT = re.compile(r"[૦-૯]")
# Union used when config does not specify a digit class.
NATIVE_DIGIT = re.compile(r"[०-९૦-૯]")


def _digit_re(cfg: dict) -> Pattern[str]:
    """Pick digit class from config flags (Marathi default for BC)."""
    if cfg.get("require_native_digit"):
        # Gujarati profile (and future Indic): prefer Gujarati if markers are Guj.
        markers = "".join(cfg.get("markers") or [])
        if any("\u0A80" <= ch <= "\u0AFF" for ch in markers):
            return GUJARATI_DIGIT
        return NATIVE_DIGIT
    if cfg.get("require_marathi_digit", True):
        return MARATHI_DIGIT
    return NATIVE_DIGIT


def load_reference_config(config_path: Optional[str | Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "reference_id.yaml"
    path = Path(config_path)
    if not path.is_file():
        return {
            "markers": [
                "प्र.क्र.", "प्र.क्र", "जा.क्र.", "का.", "दि.", "शा.नि.",
                "शासन निर्णय", "अधिसूचना", "अनौ.संदर्भ", "क्र.",
            ],
            "require_marathi_digit": True,
            "dense_punct_chars": "/-.",
        }
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def contains_reference_id(text: str, config: Optional[dict] = None) -> bool:
    cfg = config or load_reference_config()
    digit_re = _digit_re(cfg)
    require_digit = bool(
        cfg.get("require_marathi_digit", True) or cfg.get("require_native_digit", False)
    )
    markers = cfg.get("markers") or []
    for m in markers:
        if m and m in text:
            if require_digit:
                if digit_re.search(text):
                    return True
            else:
                return True
    # dense numeral + punct pattern (e.g. २०२२/प्र.क्र.३५७ or ૨૦૨૨/…)
    dense = cfg.get("dense_punct_chars") or "/-."
    if digit_re.search(text):
        punct_hits = sum(1 for c in text if c in dense)
        digit_hits = len(digit_re.findall(text))
        if punct_hits >= 2 and digit_hits >= 3:
            return True
    return False

def extract_reference_span(text: str, config: Optional[dict] = None) -> Optional[str]:
    """Return the full text as the reference span when detected.

    For crop-level validation lines the entire expected_text is typically the
    reference line; using the full string keeps CER comparable.
    """
    if contains_reference_id(text, config):
        return text
    return None


@dataclass
class ReferenceSampleResult:
    has_reference_id: bool
    reference_id_exact_match: Optional[bool]
    reference_id_cer: Optional[float]
    reference_id_character_accuracy: Optional[float]
    gt_span: Optional[str] = None
    pred_span: Optional[str] = None


def evaluate_reference_id(
    expected: str,
    prediction: str,
    config: Optional[dict] = None,
) -> ReferenceSampleResult:
    cfg = config or load_reference_config()
    if not contains_reference_id(expected, cfg):
        return ReferenceSampleResult(
            has_reference_id=False,
            reference_id_exact_match=None,
            reference_id_cer=None,
            reference_id_character_accuracy=None,
        )
    gt_span = expected
    pred_span = prediction
    ed = levenshtein(gt_span, pred_span)
    ref_len = len(gt_span)
    cer = (ed / ref_len) if ref_len > 0 else (0.0 if pred_span == "" else 1.0)
    return ReferenceSampleResult(
        has_reference_id=True,
        reference_id_exact_match=(gt_span == pred_span),
        reference_id_cer=cer,
        reference_id_character_accuracy=1.0 - cer,
        gt_span=gt_span,
        pred_span=pred_span,
    )


@dataclass
class ReferenceCorpusMetrics:
    samples_with_reference_id: int
    reference_id_exact_match_rate: Optional[float]
    reference_id_cer: Optional[float]
    reference_id_character_accuracy: Optional[float]
    reference_id_error_count: int


def aggregate_reference_ids(results: Sequence[ReferenceSampleResult]) -> ReferenceCorpusMetrics:
    usable = [r for r in results if r.has_reference_id]
    if not usable:
        return ReferenceCorpusMetrics(0, None, None, None, 0)
    total_ed = 0
    total_len = 0
    exact = 0
    errors = 0
    for r in usable:
        gt = r.gt_span or ""
        pred = r.pred_span or ""
        total_ed += levenshtein(gt, pred)
        total_len += len(gt)
        if r.reference_id_exact_match:
            exact += 1
        else:
            errors += 1
    cer = (total_ed / total_len) if total_len else 0.0
    return ReferenceCorpusMetrics(
        samples_with_reference_id=len(usable),
        reference_id_exact_match_rate=exact / len(usable),
        reference_id_cer=cer,
        reference_id_character_accuracy=1.0 - cer,
        reference_id_error_count=errors,
    )
