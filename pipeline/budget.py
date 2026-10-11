"""Choose whole documents until the crop budget is filled."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def select_crop_ids(crops: Sequence[Mapping[str, Any]], dataset_size: int | None) -> list[str]:
    """Return crop ids, keeping every line of a selected document together."""
    if dataset_size is not None and dataset_size < 1:
        raise ValueError("dataset-size must be at least 1")
    ordered_docs: list[str] = []
    by_doc: dict[str, list[str]] = {}
    for crop in crops:
        document_id = str(crop["document_id"])
        crop_id = str(crop["crop_id"])
        if document_id not in by_doc:
            ordered_docs.append(document_id)
            by_doc[document_id] = []
        by_doc[document_id].append(crop_id)
    if dataset_size is None:
        return [crop_id for document_id in ordered_docs for crop_id in by_doc[document_id]]
    chosen: list[str] = []
    for document_id in ordered_docs:
        group = by_doc[document_id]
        if chosen and len(chosen) + len(group) > dataset_size:
            break
        chosen.extend(group)
        if len(chosen) >= dataset_size:
            break
    return chosen


def document_ids_for(crops: Sequence[Mapping[str, Any]], crop_ids: Sequence[str]) -> list[str]:
    """Return the documents that own the selected crops, in first-seen order."""
    wanted = set(crop_ids)
    seen: list[str] = []
    counted: Counter[str] = Counter()
    for crop in crops:
        if str(crop["crop_id"]) not in wanted:
            continue
        document_id = str(crop["document_id"])
        if document_id not in counted:
            seen.append(document_id)
        counted[document_id] += 1
    return seen
