"""Run options for dataset build, train-split prep, and holdout eval.

Train prepares a document-level split. It does not update model weights.
Eval reads that split and can run later on its own.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from indic_ocr_pipeline.config import KNOWN_BACKENDS
from indic_ocr_pipeline.errors import DatasetError
from indic_ocr_pipeline.review.analysis import decision_label_of, is_candidate_exact

MODEL_ORDER = ("indicocr", "chitrapathak", "tesseract")
DEFAULT_MODELS = MODEL_ORDER
SPLIT_NAMES = ("train", "val", "test")


class JobError(DatasetError):
    """A run option or stage could not be applied."""


# enabled=False languages are accepted by the flag and rejected at runtime
# until a profile is switched on. Marathi is the only active profile.
LANGUAGES: dict[str, dict[str, Any]] = {
    "marathi": {"enabled": True, "tesseract_lang": "mar+eng", "script": "Devanagari"},
    "hindi": {"enabled": False, "tesseract_lang": "hin+eng", "script": "Devanagari"},
    "english": {"enabled": False, "tesseract_lang": "eng", "script": "Latin"},
}


def parse_models(raw: str | None) -> tuple[str, ...]:
    """Return OCR backends in the standard order."""
    if raw is None or not raw.strip():
        return DEFAULT_MODELS
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if not requested:
        raise JobError("models list is empty")
    unknown = [name for name in requested if name not in KNOWN_BACKENDS]
    if unknown:
        known = ", ".join(KNOWN_BACKENDS)
        raise JobError(f"unknown model(s): {', '.join(unknown)}. Known models: {known}")
    if len(set(requested)) != len(requested):
        raise JobError("models list contains a duplicate")
    return tuple(name for name in MODEL_ORDER if name in requested)


def parse_language(name: str | None) -> str:
    """Return an enabled language profile name."""
    chosen = "marathi" if name is None or not name.strip() else name.strip().lower()
    profile = LANGUAGES.get(chosen)
    if profile is None:
        known = ", ".join(sorted(LANGUAGES))
        raise JobError(f"unknown language {chosen!r}. Known languages: {known}")
    if not profile["enabled"]:
        raise JobError(
            f"language {chosen!r} is reserved and not enabled yet. Use marathi."
        )
    return chosen


def require_datapoints(value: int | None) -> int | None:
    """Reject a non-positive datapoint cap. None means no cap."""
    if value is None:
        return None
    if value < 1:
        raise JobError("datapoints must be at least 1")
    return value


def select_documents_for_datapoints(
    rows_by_document: Mapping[str, int],
    datapoints: int | None,
) -> list[str]:
    """Keep whole documents until the row cap is reached."""
    ordered = sorted(rows_by_document)
    if datapoints is None:
        return ordered
    chosen: list[str] = []
    total = 0
    for document_id in ordered:
        count = rows_by_document[document_id]
        if chosen and total + count > datapoints:
            break
        chosen.append(document_id)
        total += count
        if total >= datapoints:
            break
    return chosen


def assign_document_splits(
    document_ids: Sequence[str],
    *,
    seed: int,
    train: float,
    val: float,
) -> dict[str, str]:
    """Assign each document to one split. Descendants stay with that document."""
    assignments: dict[str, str] = {}
    for document_id in sorted(document_ids):
        digest = hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < train:
            split = "train"
        elif bucket < train + val:
            split = "val"
        else:
            split = "test"
        assignments[document_id] = split
    return assignments


def prepare_train_split(
    verified_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train: float,
    val: float,
    datapoints: int | None,
) -> dict[str, Any]:
    """Build a document-level split from gold-eligible rows. Does not export gold."""
    eligible = [
        row
        for row in verified_rows
        if decision_label_of(row) in {"human_verified", "human_corrected"}
    ]
    counts: dict[str, int] = defaultdict(int)
    for row in eligible:
        counts[str(row["document_id"])] += 1
    documents = select_documents_for_datapoints(counts, datapoints)
    if not documents:
        raise JobError("no gold-eligible rows are available for a train split")
    assignments = assign_document_splits(documents, seed=seed, train=train, val=val)
    selected = [row for row in eligible if str(row["document_id"]) in assignments]
    by_split: dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    for row in selected:
        by_split[assignments[str(row["document_id"])]] += 1
    return {
        "weights_trained": False,
        "note": "Document split only. Model weights were not updated.",
        "datapoints_requested": datapoints,
        "rows": len(selected),
        "documents": assignments,
        "rows_by_split": by_split,
    }


def evaluate_holdout(
    verified_rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    """Score human decisions on val and test documents. Train rows are unused."""
    assignments = split.get("documents")
    if not isinstance(assignments, dict) or not assignments:
        raise JobError("train split has no document assignments")
    holdout = {doc for doc, name in assignments.items() if name in {"val", "test"}}
    rows = [row for row in verified_rows if str(row.get("document_id")) in holdout]
    exact = 0
    corrected = 0
    other = 0
    by_document: dict[str, dict[str, int]] = {}
    for row in rows:
        document_id = str(row["document_id"])
        bucket = by_document.setdefault(
            document_id,
            {"reviewed": 0, "candidate_exact": 0, "human_corrected": 0},
        )
        bucket["reviewed"] += 1
        label = decision_label_of(row)
        if is_candidate_exact(label):
            exact += 1
            bucket["candidate_exact"] += 1
        elif label == "human_corrected":
            corrected += 1
            bucket["human_corrected"] += 1
        else:
            other += 1
    reviewed = len(rows)
    return {
        "holdout_documents": sorted(holdout),
        "reviewed": reviewed,
        "candidate_exact": exact,
        "human_corrected": corrected,
        "other": other,
        "candidate_exact_rate": round(100.0 * exact / reviewed, 2) if reviewed else 0.0,
        "by_document": by_document,
        "weights_trained": False,
    }


def load_state(path: Path) -> dict[str, Any]:
    """Return the run ledger, or an empty ledger when none exists."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise JobError(f"{path} is not a JSON object")
    return value


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    """Write the run ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_eval_report(report: Mapping[str, Any]) -> str:
    """Return the holdout eval text."""
    lines = [
        "Holdout eval",
        "Model weights were not trained or updated.",
        f"reviewed: {report['reviewed']}",
        f"candidate_exact: {report['candidate_exact']}",
        f"human_corrected: {report['human_corrected']}",
        f"other: {report['other']}",
        f"candidate_exact_rate: {report['candidate_exact_rate']}%",
        "by_document:",
    ]
    by_document = report["by_document"]
    if isinstance(by_document, dict):
        for document_id in sorted(by_document):
            bucket = by_document[document_id]
            lines.append(
                f"  {document_id}: reviewed={bucket['reviewed']} "
                f"exact={bucket['candidate_exact']} "
                f"corrected={bucket['human_corrected']}"
            )
    return "\n".join(lines) + "\n"
