"""Review-priority rules that do not require OCR models.

High-confidence rows still need a random audit. Until that audit measures a
low error rate, those rows stay candidates.
"""

from __future__ import annotations


def high_confidence_audit_count(
    population: int,
    fraction: float,
    minimum: int,
    maximum: int,
) -> int:
    """Return how many high-confidence candidates to sample for audit.

    The sample is capped by ``maximum`` and by the population. A population
    smaller than ``minimum`` is sampled in full so a pilot is still inspected.
    """
    if isinstance(population, bool) or not isinstance(population, int) or population < 0:
        raise ValueError("population must be a non-negative integer")
    if population == 0:
        return 0
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise ValueError("fraction must be a number")
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise ValueError("audit bounds must be integers")
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise ValueError("audit bounds must be integers")
    if minimum < 0 or maximum < 1 or minimum > maximum:
        raise ValueError("audit bounds are invalid")
    target = int(round(population * float(fraction)))
    target = max(target, minimum)
    return min(target, maximum, population)
