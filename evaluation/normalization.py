"""Comparison normalization layers.

STRICT is primary. Diagnostic NFC+whitespace must never map numerals or
strip punctuation.
"""
from __future__ import annotations

import re
import unicodedata


def strict(text: str) -> str:
    """Identity — primary benchmark view. Never mutate GT or prediction."""
    return text


def diagnostic_nfc_whitespace(text: str) -> str:
    """Conservative diagnostic normalization only.

    - Unicode NFC
    - Trim edges
    - Collapse runs of whitespace to a single space

    Does NOT convert Marathi digits, strip punctuation, or alter Devanagari.
    """
    t = unicodedata.normalize("NFC", text)
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


NORMALIZERS = {
    "strict": strict,
    "nfc_whitespace": diagnostic_nfc_whitespace,
}


def content_without_whitespace(text: str) -> str:
    """Remove all whitespace after NFC. Does not alter punctuation/digits."""
    t = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", t)
