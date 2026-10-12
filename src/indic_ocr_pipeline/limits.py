"""Cap how many source pages and lines a run builds.

``--dataset-size N`` asks for about N line crops. The cap is applied while
PDFs are ingested, so a small test does not render the whole corpus.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path


def dataset_size_from_env() -> int | None:
    """Return the line budget from ``DATASET_SIZE``, or None when it is unset."""
    raw = os.environ.get("DATASET_SIZE", "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("DATASET_SIZE must be at least 1")
    return value


def pages_to_keep(page_counts: Sequence[int], dataset_size: int) -> list[int]:
    """Return how many leading pages to take from each PDF, in order.

    About four lines are expected per page. A second PDF is included when one
    exists, so the later train and validation splits can both receive lines.
    """
    if dataset_size < 1:
        raise ValueError("dataset-size must be at least 1")
    usable = [count for count in page_counts if count >= 1]
    if not usable:
        return []
    target_pages = max(2, -(-dataset_size // 4))
    plan: list[int] = []
    kept = 0
    for index, count in enumerate(usable):
        if len(plan) >= 2 and kept >= target_pages:
            break
        if index == 0 and len(usable) > 1:
            take = min(count, max(1, target_pages - 1))
        else:
            take = min(count, max(1, target_pages - kept))
        plan.append(take)
        kept += take
    return plan


def line_quotas(dataset_size: int, documents: int) -> list[int]:
    """Split a line budget across documents. Earlier documents get the remainder."""
    if dataset_size < 1:
        raise ValueError("dataset-size must be at least 1")
    if documents < 1:
        return []
    base, extra = divmod(dataset_size, documents)
    return [base + (1 if index < extra else 0) for index in range(documents)]


def pdf_page_count(path: Path) -> int:
    """Return the number of pages in a PDF."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()
