"""Language profiles for OCR evaluation (Marathi / Gujarati / …).

Keeps a single metric engine while isolating script-specific inventories:
digits, vowel signs (matras), conjunct patterns, reference-ID markers,
punctuation overlays, and metric/pipeline version labels.

Marathi defaults remain the package defaults so existing Marathi runs and
tests are unchanged when ``language="marathi"`` (or omitted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"

# Unicode Gujarati dependent vowel signs + anusvara/visarga/candrabindu + virama
# Present in frozen Gujarati GT (U+0A80–U+0AFF). Do not reuse Devanagari matras.
GUJARATI_VOWEL_SIGNS: Set[str] = set("ાિીુૂૃૄેૈોૌૅૉંઃઁ્")

# Devanagari matras (Marathi) — kept here for documentation; matras.py owns DEFAULT.
MARATHI_MATRAS: Set[str] = set("ािीुूृेैोौंःँ्")

GUJARATI_DIGITS = "૦૧૨૩૪૫૬૭૮૯"
MARATHI_DIGITS = "०१२३४५६७८९"


@dataclass(frozen=True)
class LanguageProfile:
    language: str
    display_name: str
    metric_version: str
    pipeline_version: str
    canonical_config_name: str
    native_digits: str
    vowel_signs: Set[str]
    conjunct_config: Path
    punctuation_config: Path
    reference_id_config: Optional[Path]
    reference_id_applicable: bool
    summary_metrics_key: str  # e.g. "marathi" | "gujarati"
    numeral_accuracy_field: str  # field name in per-sample / summary
    default_jsonl_name: str
    notes: str = ""


def _cfg(name: str) -> Path:
    return _CONFIG_DIR / name


PROFILES: Dict[str, LanguageProfile] = {
    "marathi": LanguageProfile(
        language="marathi",
        display_name="Marathi",
        metric_version="marathi_ocr_eval_v2",
        pipeline_version="marathi_piecewise_canonical_v1",
        canonical_config_name="canonical_pipeline.yaml",
        native_digits=MARATHI_DIGITS,
        vowel_signs=set(MARATHI_MATRAS),
        conjunct_config=_cfg("conjuncts.yaml"),
        punctuation_config=_cfg("punctuation.yaml"),
        reference_id_config=_cfg("reference_id.yaml"),
        reference_id_applicable=True,
        summary_metrics_key="marathi",
        numeral_accuracy_field="marathi_numeral_character_accuracy",
        default_jsonl_name="validation.jsonl",
        notes="Frozen Marathi profile; do not change metric semantics.",
    ),
    "gujarati": LanguageProfile(
        language="gujarati",
        display_name="Gujarati",
        metric_version="gujarati_ocr_eval_v1",
        pipeline_version="gujarati_piecewise_canonical_v1",
        canonical_config_name="gujarati_canonical_pipeline.yaml",
        native_digits=GUJARATI_DIGITS,
        vowel_signs=set(GUJARATI_VOWEL_SIGNS),
        conjunct_config=_cfg("gujarati_conjuncts.yaml"),
        punctuation_config=_cfg("gujarati_punctuation.yaml"),
        reference_id_config=_cfg("gujarati_reference_id.yaml"),
        reference_id_applicable=True,
        summary_metrics_key="gujarati",
        numeral_accuracy_field="gujarati_numeral_character_accuracy",
        default_jsonl_name="special_chars_expansion.jsonl",
        notes=(
            "Same WER/CER definitions as marathi_ocr_eval_v2 where language-"
            "independent; script inventories are Gujarati-specific."
        ),
    ),
}


def get_profile(language: str) -> LanguageProfile:
    key = (language or "marathi").strip().lower()
    if key not in PROFILES:
        raise ValueError(
            f"Unknown language profile {language!r}. "
            f"Known: {sorted(PROFILES)}"
        )
    return PROFILES[key]


def load_profile_yaml_extras(profile: LanguageProfile) -> Dict[str, Any]:
    """Optional extras from language YAML (currently unused placeholder)."""
    path = _CONFIG_DIR / f"{profile.language}_profile.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
