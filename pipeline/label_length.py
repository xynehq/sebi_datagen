"""Keep dataset labels short.

Digits count. Spaces do not. Nothing longer than 10 characters is stored.
At least 95% of the stored labels are shorter than 5 characters. The rest,
when the set is large enough to allow them, are 8 to 10 characters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelLengthPolicy:
    """Character limits for one exported dataset."""

    max_characters: int = 10
    short_below: int = 5
    short_fraction: float = 0.95
    long_min_characters: int = 8

    def __post_init__(self) -> None:
        if self.max_characters < self.long_min_characters:
            raise ValueError("max_characters must be at least long_min_characters")
        if self.short_below < 2:
            raise ValueError("short_below must be at least 2")
        if self.long_min_characters <= self.short_below:
            raise ValueError("long labels must start above the short limit")
        if not 0 < self.short_fraction <= 1:
            raise ValueError("short_fraction must be between 0 and 1")


def policy_from_config(raw: object) -> LabelLengthPolicy:
    """Read ``label_length`` from dataset.yaml. Missing keys keep the defaults."""
    if not isinstance(raw, dict):
        return LabelLengthPolicy()
    values = raw.get("label_length")
    if not isinstance(values, dict):
        return LabelLengthPolicy()
    base = LabelLengthPolicy()
    return LabelLengthPolicy(
        max_characters=int(values.get("max_characters", base.max_characters)),
        short_below=int(values.get("short_below", base.short_below)),
        short_fraction=float(values.get("short_fraction", base.short_fraction)),
        long_min_characters=int(values.get("long_min_characters", base.long_min_characters)),
    )


def compact_label(text: str) -> str:
    """Return the label the trainer sees: internal spaces removed, digits kept."""
    return "".join(text.split())


@dataclass(frozen=True)
class LabelPiece:
    text: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return len(self.text)


def plan_pieces(
    texts: Sequence[str],
    policy: LabelLengthPolicy | None = None,
) -> list[list[LabelPiece]]:
    """Split each label into pieces that obey the length rule.

    A piece shorter than ``short_below`` is the common case. A piece of
    ``long_min_characters`` to ``max_characters`` is kept only while at least
    ``short_fraction`` of all pieces stay short.
    """
    rules = policy or LabelLengthPolicy()
    units = [compact_label(text) for text in texts]
    planned = [_cut_short(text, rules.short_below - 1) for text in units]
    short_count = sum(len(pieces) for pieces in planned)
    long_count = 0
    for index, text in enumerate(units):
        if len(text) < rules.long_min_characters:
            continue
        take = min(rules.max_characters, len(text))
        if take < rules.long_min_characters:
            continue
        original = len(planned[index])
        tail = _cut_short(text[take:], rules.short_below - 1, offset=take)
        new_short = short_count - original + len(tail)
        new_long = long_count + 1
        if new_short <= 0 or new_short / (new_short + new_long) < rules.short_fraction:
            continue
        planned[index] = [LabelPiece(text[:take], 0, take), *tail]
        short_count = new_short
        long_count = new_long
    return planned


def _cut_short(text: str, width: int, offset: int = 0) -> list[LabelPiece]:
    pieces: list[LabelPiece] = []
    cursor = 0
    while cursor < len(text):
        take = min(width, len(text) - cursor)
        start = offset + cursor
        pieces.append(LabelPiece(text[cursor : cursor + take], start, start + take))
        cursor += take
    return pieces

