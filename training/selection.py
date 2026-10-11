"""Checkpoint selection on the development validation split.

The primary metric is strict CER from the canonical evaluation harness.
Lower is better. Frozen-100 is not an input to this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pipeline.gold import GoldError


class SelectionError(GoldError):
    """Checkpoint selection was asked to use a forbidden set."""


def select_best(
    scored: Sequence[Mapping[str, Any]],
    *,
    frozen_ids: set[str],
    primary_metric: str = "cer",
) -> dict[str, Any]:
    """Return the scoreboard and the winning checkpoint.

    ``scored`` items need ``checkpoint``, ``metrics``, and ``sample_ids``.
    """
    if not scored:
        raise SelectionError("no checkpoints to select")
    rows: list[dict[str, Any]] = []
    for item in scored:
        sample_ids = [str(value) for value in item.get("sample_ids") or []]
        leaked = sorted(set(sample_ids) & set(frozen_ids))
        if leaked:
            raise SelectionError(
                "checkpoint selection cannot use frozen-100 "
                f"(checkpoint {item.get('checkpoint')}, ids {leaked[:3]})"
            )
        if item.get("split") == "frozen" or item.get("benchmark") == "frozen100":
            raise SelectionError("checkpoint selection cannot use the frozen benchmark")
        metrics = dict(item["metrics"])
        if primary_metric not in metrics:
            raise SelectionError(f"missing primary metric {primary_metric}")
        rows.append(
            {
                "checkpoint": str(item["checkpoint"]),
                "metrics": metrics,
                "primary_metric": primary_metric,
                "primary_value": metrics[primary_metric],
                "sample_count": len(sample_ids),
                "sample_ids": sample_ids,
                "frozen_used": False,
            }
        )
    rows.sort(key=lambda row: (float(row["primary_value"]), row["checkpoint"]))
    winner = rows[0]["checkpoint"]
    return {
        "primary_metric": primary_metric,
        "goal": "minimize",
        "frozen_used": False,
        "rows": rows,
        "selected": winner,
    }
