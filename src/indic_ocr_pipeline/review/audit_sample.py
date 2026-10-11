"""Deterministic calibration samples.

AUTO_3_AGREE and AUTO_2_AGREE stay consensus rows. A sampled audit does not
turn them into gold. ALL_DIFFER, EMPTY, and BAD_BOUNDARY are queued in full.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

DEFAULT_SEED = 3407
DEFAULT_RATES = {
    "ALL_3_AGREE": 0.05,
    "TWO_AGREE": 0.20,
    "ALL_DIFFER": 1.0,
    "EMPTY": 1.0,
    "BAD_BOUNDARY": 1.0,
}


def audit_unit(seed: int, sample_id: str) -> float:
    """Return a stable unit interval for one sample. Order does not matter."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def selected_for_audit(
    agreement_class: str,
    sample_id: str,
    *,
    seed: int = DEFAULT_SEED,
    rates: Mapping[str, float] | None = None,
) -> bool:
    """Return whether this crop is in the calibration audit."""
    table = dict(DEFAULT_RATES)
    if rates:
        table.update(rates)
    if agreement_class not in table:
        raise ValueError(f"unknown agreement class {agreement_class!r}")
    rate = float(table[agreement_class])
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return audit_unit(seed, sample_id) < rate


def sample_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    rates: Mapping[str, float] | None = None,
    id_field: str = "crop_id",
    class_field: str = "agreement_class",
) -> list[dict[str, object]]:
    """Return the audit subset. The same seed and ids always yield the same subset."""
    chosen: list[dict[str, object]] = []
    for row in rows:
        sample_id = str(row[id_field])
        agreement_class = str(row[class_field])
        if selected_for_audit(agreement_class, sample_id, seed=seed, rates=rates):
            item = dict(row)
            item["audit_selected"] = True
            item["audit_seed"] = seed
            item["gold"] = False
            chosen.append(item)
    chosen.sort(key=lambda item: str(item[id_field]))
    return chosen
