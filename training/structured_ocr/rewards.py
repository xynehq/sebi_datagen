"""Verifiable rewards for synthetic / known-structure documents (RLVR-ready)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.structured_ocr.metrics import (
    evaluate_structure,
    looks_column_major_dump,
    table_cell_association_score,
)
from training.structured_ocr.targets import validate_target


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    structure_valid: float
    row_count: float
    cell_association: float
    no_column_dump: float
    no_repetition: float
    termination: float
    details: dict[str, Any]


def score_verifiable_output(
    reference: str,
    prediction: str,
    *,
    expect_table_rows: int | None = None,
    expect_empty: bool = False,
) -> RewardBreakdown:
    """Deterministic reward in [0, 1] for synthetic or fully specified references."""
    report = validate_target(
        reference,
        expect_table_rows=expect_table_rows,
        allow_empty=expect_empty,
    )
    pred_report = validate_target(prediction, allow_empty=expect_empty)
    metrics = evaluate_structure(reference, prediction)

    structure_valid = 1.0 if pred_report.ok else 0.0
    row_count = 1.0
    if expect_table_rows is not None:
        row_count = 1.0 if metrics.table_row_count_pred == expect_table_rows else 0.0
    elif report.table_row_count:
        row_count = 1.0 if metrics.table_row_count_match else max(
            0.0, 1.0 - abs(metrics.table_row_count_pred - metrics.table_row_count_ref) / report.table_row_count
        )

    cell_assoc = table_cell_association_score(reference, prediction)
    no_dump = 0.0 if looks_column_major_dump(prediction) else 1.0
    no_rep = max(0.0, 1.0 - min(1.0, metrics.repetition_score * 2))
    termination = 1.0 if prediction.strip() and not prediction.strip().endswith("...") else 0.5

    weights = {
        "structure_valid": 0.15,
        "row_count": 0.25,
        "cell_association": 0.35,
        "no_column_dump": 0.15,
        "no_repetition": 0.05,
        "termination": 0.05,
    }
    parts = {
        "structure_valid": structure_valid,
        "row_count": row_count,
        "cell_association": cell_assoc,
        "no_column_dump": no_dump,
        "no_repetition": no_rep,
        "termination": termination,
    }
    total = sum(weights[k] * parts[k] for k in weights)
    return RewardBreakdown(
        total=total,
        structure_valid=structure_valid,
        row_count=row_count,
        cell_association=cell_assoc,
        no_column_dump=no_dump,
        no_repetition=no_rep,
        termination=termination,
        details={"metrics": metrics, "reference_report": report, "prediction_report": pred_report},
    )
