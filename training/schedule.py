"""Validation cadence during training.

Two checks run while training is in progress, at 30% and 70% of the optimizer
steps. The last check is the evaluation stage after training finishes, so it
is not one of these steps. The frozen benchmark is not an input to this
schedule.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_VAL_CHECKS = 10
DURING_TRAINING_FRACTIONS = (0.30, 0.70)


def validation_interval(steps: int, checks: int = DEFAULT_VAL_CHECKS) -> int:
    """Return how many optimizer steps to run between validation passes.

    The final step is always a validation step. For a budget that divides
    evenly, the result is ``steps // checks`` (1000 steps, 10 checks → 100).
    """
    if steps < 1:
        raise ValueError("optimizer steps must be positive")
    if checks < 1:
        raise ValueError("validation checks must be positive")
    return max(1, steps // min(checks, steps))


def validation_steps(
    steps: int,
    checks: int = DEFAULT_VAL_CHECKS,
    interval: int | None = None,
) -> list[int]:
    """Optimizer steps at which the development validation set is scored."""
    step_interval = interval or validation_interval(steps, checks)
    points = list(range(step_interval, steps + 1, step_interval))
    if not points or points[-1] != steps:
        points.append(steps)
    return points


def during_training_steps(
    total: int,
    fractions: tuple[float, ...] = DURING_TRAINING_FRACTIONS,
) -> list[int]:
    """Optimizer steps for the checks that run before training finishes.

    Each fraction is rounded to a step. The final step is left out: that
    score is the evaluation stage after training. Fractions that land on the
    same step, or only on the last step, are kept once or dropped.
    """
    if total < 1:
        raise ValueError("optimizer steps must be positive")
    points: list[int] = []
    for fraction in fractions:
        step = int(round(float(fraction) * total))
        step = min(max(step, 1), total - 1)
        if 1 <= step < total and step not in points:
            points.append(step)
    return points


def resolve_training_schedule(
    *,
    train_count: int,
    batch_size: int,
    epochs: int,
    steps: int | None,
    val_checks: int = DEFAULT_VAL_CHECKS,
) -> dict[str, Any]:
    """Turn a step budget, or epochs times batches, into the 30% and 70% checks.

    ``steps`` wins when it is set. Otherwise the budget is one optimizer step
    per training batch, repeated for ``epochs``. ``val_checks`` is kept so
    older callers still pass, and does not change the 30% and 70% points.
    Frozen-100 is not consulted.
    """
    del val_checks
    if steps is not None:
        total = int(steps)
    else:
        batches = math.ceil(train_count / max(int(batch_size), 1)) if train_count else 1
        total = max(1, batches * max(int(epochs), 1))
    during = during_training_steps(total)
    return {
        "optimizer_steps": total,
        "val_checks": len(during) + 1,
        "val_check_interval": (
            during[1] - during[0] if len(during) >= 2 else (during[0] if during else 0)
        ),
        "validation_steps": during,
        "validation_after_training": True,
        "during_fractions": list(DURING_TRAINING_FRACTIONS),
        "frozen_used_for_selection": False,
    }
