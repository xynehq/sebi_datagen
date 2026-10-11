"""Storage form keeps punctuation and digits. Comparison keys are separate."""

from __future__ import annotations

from indic_ocr_pipeline.text_policy import (
    apply_storage_normalization,
    has_replacement_character,
    is_storage_form,
    normalize_for_comparison,
)


def test_storage_keeps_internal_spacing_punctuation_and_digits() -> None:
    text = "रु. १,२३०/- (२)"
    assert apply_storage_normalization(f"  {text}  ") == text
    assert apply_storage_normalization("महाराष्ट्र  शासन") == "महाराष्ट्र  शासन"
    assert is_storage_form(text)
    assert not is_storage_form(f" {text}")
    assert not is_storage_form("e\u0301")
    assert apply_storage_normalization("e\u0301") == "é"


def test_comparison_key_is_not_the_stored_label() -> None:
    stored = apply_storage_normalization("महाराष्ट्र  शासन ")
    compared = normalize_for_comparison("महाराष्ट्र  शासन")
    assert stored == "महाराष्ट्र  शासन"
    assert compared == "महाराष्ट्र शासन"
    assert stored != compared


def test_replacement_character_is_detected() -> None:
    assert has_replacement_character("शासन \ufffd")
    assert not has_replacement_character("शासन")
