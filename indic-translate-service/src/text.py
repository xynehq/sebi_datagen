"""Text helpers around IndicTrans2: sentence splitting before, repetition trimming after."""

from __future__ import annotations

import re

_SENT_END = re.compile(r"(?<=[।॥.?!])\s+")

# IndicTrans2 is trained on single sentences (≤ 256 tokens). OCR text often has no sentence
# punctuation (tables, lists), so very long "sentences" are cut into pieces of this many words.
MAX_WORDS_PER_PIECE = 60


def split_sentences(text: str, max_words: int = MAX_WORDS_PER_PIECE) -> list[str]:
    """Join OCR line wraps, split on sentence ends, cap very long sentences."""
    joined = re.sub(r"\s*\n\s*", " ", text).strip()
    out: list[str] = []
    for sentence in _SENT_END.split(joined):
        words = sentence.split()
        if len(re.findall(r"\w", sentence)) < 3:
            continue
        out.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
    return out


def trim_repeats(text: str) -> str:
    """Collapse the repetitions IndicTrans2 sometimes produces on noisy OCR input."""
    text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)  # "ohhhhhh" -> "ohhh"
    return re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", text)  # repeated words
