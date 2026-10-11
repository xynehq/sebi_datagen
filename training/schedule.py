"""Validation cadence during training.

The interval follows the optimizer-step budget. One thousand steps with the
default of ten checks validates at 100, 200, ..., 1000. The frozen benchmark
is not an input to this schedule.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_VAL_CHECKS = 10


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


def resolve_training_schedule(
    *,
    train_count: int,
    batch_size: int,
    epochs: int,
    steps: int | None,
    val_checks: int = DEFAULT_VAL_CHECKS,
) -> dict[str, Any]:
    """Turn a step budget, or epochs times batches, into a validation grid.

    ``steps`` wins when it is set. Otherwise the budget is one optimizer step
    per training batch, repeated for ``epochs``. Frozen-100 is not consulted.
    """
    if steps is not None:
        total = int(steps)
    else:
        batches = math.ceil(train_count / max(int(batch_size), 1)) if train_count else 1
        total = max(1, batches * max(int(epochs), 1))
    interval = validation_interval(total, val_checks)
    return {
        "optimizer_steps": total,
        "val_checks": min(int(val_checks), total),
        "val_check_interval": interval,
        "validation_steps": validation_steps(total, val_checks),
        "frozen_used_for_selection": False,
    }
