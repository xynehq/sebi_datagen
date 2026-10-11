"""Unicode policy for stored labels and for comparison keys.

Storage normalization is NFC plus removal of surrounding whitespace.
Punctuation, digits, and internal spaces stay in the label.

Comparison normalization is a separate key used to measure agreement.
It must never be written back as gold text.
"""

from __future__ import annotations

import unicodedata

REPLACEMENT_CHARACTER = "\ufffd"


def apply_storage_normalization(text: str) -> str:
    """Return the storage form of a human label.

    Callers must pass the result into a gold record explicitly.
    Gold construction checks the text is already in this form and does not
    rewrite it.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFC", text).strip()


def is_storage_form(text: str) -> bool:
    """Return whether ``text`` is already in storage form."""
    return text == apply_storage_normalization(text)


def normalize_for_comparison(text: str) -> str:
    """Return a comparison key. This is not a transcription."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).strip()
    return " ".join(normalized.split())


def has_replacement_character(text: str) -> bool:
    """Return whether ``text`` contains U+FFFD."""
    return REPLACEMENT_CHARACTER in text
