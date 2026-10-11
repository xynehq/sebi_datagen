"""Normalized edit similarity between OCR strings. This is not a gold label."""

from __future__ import annotations


def edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance. Comparison keys should already be normalized."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char != other)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    """Return 1 when the strings match and 0 when they share no aligned characters."""
    if not left and not right:
        return 1.0
    width = max(len(left), len(right))
    if width == 0:
        return 1.0
    return 1.0 - (edit_distance(left, right) / width)


def pairwise_similarity(texts: dict[str, str]) -> dict[str, float]:
    """Score every unordered pair. Keys are ``left__right`` in input order."""
    names = list(texts)
    scored: dict[str, float] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            scored[f"{left}__{right}"] = round(similarity(texts[left], texts[right]), 4)
    return scored
