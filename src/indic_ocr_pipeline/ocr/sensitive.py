"""Flag lines whose errors are costly. Agreement does not clear the flag."""

from __future__ import annotations

import re
import unicodedata

_LINK = re.compile(r"(https?://|www\.|@)", re.IGNORECASE)
_MONEY = ("₹", "$", "€", "£", "%")


def is_sensitive(text: str) -> bool:
    """Return whether a transcription contains a number, money mark, or identifier."""
    if _LINK.search(text):
        return True
    if any(mark in text for mark in _MONEY):
        return True
    return any(unicodedata.category(char) in {"Nd", "No"} for char in text)


def line_is_sensitive(texts: dict[str, str | None]) -> bool:
    """Return whether any model string on the line is sensitive."""
    return any(is_sensitive(text) for text in texts.values() if isinstance(text, str) and text)
