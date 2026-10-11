"""Spend a fixed human-audit budget across consensus categories.

Sensitive lines are preferred inside a category, but each main category keeps
a reserved share so sensitive lines cannot consume the whole budget.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from indic_ocr_pipeline.review.audit_sample import audit_unit

FAMILY_WEIGHTS = {
    "disagree": 0.50,
    "majority": 0.30,
    "unanimous": 0.15,
    "other": 0.05,
}


def allocate_audit(
    rows: Sequence[Mapping[str, object]],
    budget: int,
    *,
    seed: int = 3407,
) -> list[dict[str, object]]:
    """Return a deterministic audit sample of at most ``budget`` rows."""
    if budget < 1:
        return []
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[_family(row)].append(dict(row))
    quotas = _family_quotas(by_family, budget)
    chosen: list[dict[str, object]] = []
    picked_ids: set[str] = set()
    for name in FAMILY_WEIGHTS:
        pool = [row for row in by_family.get(name, []) if str(row.get("crop_id")) not in picked_ids]
        batch = _fill_family(pool, quotas.get(name, 0), seed)
        chosen.extend(batch)
        picked_ids.update(str(row.get("crop_id")) for row in batch)
    return _mark(chosen[:budget], seed)


def allocation_summary(
    rows: Sequence[Mapping[str, object]],
    budget: int,
    seed: int,
) -> dict[str, object]:
    """Record how the budget was actually spent."""
    by_class: dict[str, int] = defaultdict(int)
    by_document: dict[str, int] = defaultdict(int)
    for row in rows:
        by_class[str(row.get("agreement_class") or "unknown")] += 1
        by_document[str(row.get("document_id") or "")] += 1
    return {
        "seed": seed,
        "budget": budget,
        "selected": len(rows),
        "sensitive": sum(1 for row in rows if row.get("sensitive")),
        "by_agreement_class": dict(by_class),
        "by_document": dict(by_document),
    }


def _family(row: Mapping[str, object]) -> str:
    kind = str(row.get("agreement_class") or row.get("consensus_status") or "")
    if kind in {"ALL_DIFFER", "NO_CONSENSUS"}:
        return "disagree"
    if kind in {"EMPTY", "BAD_BOUNDARY"}:
        return "other"
    if kind in {"ALL_3_AGREE"} or _unanimous_name(kind):
        return "unanimous"
    return "majority"


def _unanimous_name(kind: str) -> bool:
    parts = kind.split("_")
    return len(parts) == 3 and parts[0] == "CONSENSUS" and parts[1] == parts[2]


def _family_quotas(
    by_family: dict[str, list[dict[str, object]]],
    budget: int,
) -> dict[str, int]:
    present = [name for name in FAMILY_WEIGHTS if by_family.get(name)]
    if not present:
        return {}
    quotas = {name: 0 for name in present}
    if budget >= len(present):
        for name in present:
            quotas[name] = 1
    leftover = budget - sum(quotas.values())
    weights = {name: FAMILY_WEIGHTS[name] for name in present}
    total = sum(weights.values()) or 1.0
    raw = {name: leftover * (weights[name] / total) for name in present}
    extra = {name: min(len(by_family[name]) - quotas[name], int(raw[name])) for name in present}
    for name in present:
        extra[name] = max(0, extra[name])
        quotas[name] += extra[name]
    remainder = budget - sum(quotas.values())
    order = sorted(present, key=lambda name: (raw[name] - extra[name], name), reverse=True)
    while remainder > 0 and any(quotas[name] < len(by_family[name]) for name in order):
        for name in order:
            if remainder <= 0:
                break
            if quotas[name] < len(by_family[name]):
                quotas[name] += 1
                remainder -= 1
    return quotas


def _fill_family(rows: list[dict[str, object]], count: int, seed: int) -> list[dict[str, object]]:
    """Prefer sensitive lines, and take turns across source documents."""
    if count <= 0 or not rows:
        return []
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row.get("sensitive") else 1,
            audit_unit(seed, str(row.get("crop_id"))),
            str(row.get("crop_id")),
        ),
    )
    for row in ordered:
        buckets[str(row.get("document_id") or "")].append(row)
    keys = sorted(buckets)
    picked: list[dict[str, object]] = []
    index = 0
    while len(picked) < count and any(buckets.values()):
        key = keys[index % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop(0))
        index += 1
        if index > len(rows) * 4:
            break
    return picked


def _mark(rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    marked: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["audit_selected"] = True
        item["audit_seed"] = seed
        item["gold"] = False
        marked.append(item)
    marked.sort(key=lambda item: str(item.get("crop_id")))
    return marked
