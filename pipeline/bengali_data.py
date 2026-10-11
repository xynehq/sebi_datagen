"""Prepare a Bengali image/text split from a local source. Nothing is invented."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from training.dataset import load_frozen_ids

TARGET = 10000
TRAIN_COUNT = 9200
VAL_COUNT = 800
SEED = 3407


class BengaliDataError(RuntimeError):
    """The Bengali source is missing or cannot fill the requested split."""


def prepare(
    root: Path,
    source: Path | None = None,
    *,
    target: int = TARGET,
    train_count: int = TRAIN_COUNT,
    val_count: int = VAL_COUNT,
) -> dict[str, object]:
    """Audit ``source`` and write ``data/bengali/real10k`` only when enough rows exist."""
    source_dir = source or (root / "data" / "bengali" / "source")
    created = ensure_source_layout(source_dir)
    labels = source_dir / "labels.jsonl"
    report_path = root / "reports" / "bengali" / "dataset_audit.txt"
    summary = _audit(root, source_dir, labels, target=target)
    summary["layout_created"] = created
    summary["train_count"] = train_count
    summary["val_count"] = val_count
    summary["target"] = target
    _write_report(report_path, summary)
    if int(summary["eligible"]) < target:
        raise BengaliDataError(
            f"Bengali source folder is ready at {source_dir}. "
            f"Eligible rows: {summary['eligible']}, not {target}. "
            "Put images in images/ and rows with image_filename and expected_text "
            f"in {labels}. The 10k split was not written."
        )
    if train_count + val_count != target:
        raise BengaliDataError("train_count and val_count must add up to the target")
    rows = summary["rows"]
    assert isinstance(rows, list)
    _write_split(root, source_dir, rows, train_count=train_count, target=target)
    summary["written"] = True
    _write_report(report_path, summary)
    return summary


def ensure_source_layout(source_dir: Path) -> bool:
    """Create ``images/`` and an empty ``labels.jsonl`` when they are absent.

    An existing labels file is left unchanged. Returns True when the labels
    file was created by this call.
    """
    (source_dir / "images").mkdir(parents=True, exist_ok=True)
    labels = source_dir / "labels.jsonl"
    if labels.exists():
        return False
    labels.write_text("", encoding="utf-8")
    return True


def _audit(
    root: Path, source_dir: Path, labels: Path, *, target: int = TARGET
) -> dict[str, object]:
    frozen = load_frozen_ids(root / "frozen" / "marathi100")
    summary: dict[str, object] = {
        "source": str(source_dir),
        "source_present": labels.is_file(),
        "images_found": 0,
        "eligible": 0,
        "train": 0,
        "val": 0,
        "missing_images": 0,
        "unreadable_images": 0,
        "invalid_records": 0,
        "duplicate_images": 0,
        "empty_labels": 0,
        "overlap": 0,
        "unicode_errors": 0,
        "frozen_excluded": 0,
        "seed": SEED,
        "target_reached": False,
        "written": False,
        "rows": [],
    }
    image_dir = source_dir / "images"
    if image_dir.is_dir():
        summary["images_found"] = sum(1 for path in image_dir.iterdir() if path.is_file())
    if not labels.is_file():
        return summary
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for line in labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            summary["invalid_records"] = int(summary["invalid_records"]) + 1
            continue
        if not isinstance(row, dict):
            summary["invalid_records"] = int(summary["invalid_records"]) + 1
            continue
        text = row.get("expected_text")
        name = row.get("image_filename")
        if not isinstance(text, str) or not text.strip() or "\ufffd" in text:
            if isinstance(text, str) and "\ufffd" in text:
                summary["unicode_errors"] = int(summary["unicode_errors"]) + 1
            else:
                summary["empty_labels"] = int(summary["empty_labels"]) + 1
            continue
        if not isinstance(name, str) or not name.strip():
            summary["invalid_records"] = int(summary["invalid_records"]) + 1
            continue
        key = Path(name.lstrip("/")).name
        if key in seen or name in frozen or key in frozen:
            if key in seen:
                summary["duplicate_images"] = int(summary["duplicate_images"]) + 1
            else:
                summary["frozen_excluded"] = int(summary["frozen_excluded"]) + 1
            continue
        image = _resolve_image(source_dir, name)
        if image is None or not image.is_file():
            summary["missing_images"] = int(summary["missing_images"]) + 1
            continue
        if not _readable(image):
            summary["unreadable_images"] = int(summary["unreadable_images"]) + 1
            continue
        seen.add(key)
        rows.append({"image_filename": f"/images/{key}", "expected_text": text})
    summary["eligible"] = len(rows)
    summary["rows"] = rows
    summary["target_reached"] = len(rows) >= target
    return summary


def _write_split(
    root: Path,
    source_dir: Path,
    rows: list[dict[str, str]],
    *,
    train_count: int,
    target: int,
) -> None:
    ordered = sorted(rows, key=lambda row: _rank(row["image_filename"]))
    chosen = ordered[:target]
    train = chosen[:train_count]
    val = chosen[train_count:target]
    train_names = {row["image_filename"] for row in train}
    val_names = {row["image_filename"] for row in val}
    if train_names & val_names:
        raise BengaliDataError("train and validation overlap")
    destination = root / "data" / "bengali" / "real10k"
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_images = source_dir / "images"
    for row in chosen:
        name = Path(row["image_filename"]).name
        copied = image_dir / name
        origin = source_images / name
        if origin.is_file() and not copied.exists():
            copied.write_bytes(origin.read_bytes())
    _jsonl(destination / "train.jsonl", train)
    _jsonl(destination / "val.jsonl", val)


def _resolve_image(source: Path, name: str) -> Path | None:
    relative = Path(name.lstrip("/"))
    if relative.is_absolute():
        return None
    candidate = (source / relative).resolve()
    try:
        candidate.relative_to(source.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    fallback = source / "images" / relative.name
    return fallback if fallback.is_file() else None


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        return False
    return True


def _rank(name: str) -> str:
    return hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _write_report(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Bengali dataset audit",
        f"source: {summary['source']}",
        f"seed: {summary['seed']}",
        f"images found: {summary['images_found']}",
        f"eligible records: {summary['eligible']}",
        f"training rows: {summary.get('train_count', TRAIN_COUNT) if summary['written'] else 0}",
        f"validation rows: {summary.get('val_count', VAL_COUNT) if summary['written'] else 0}",
        f"missing images: {summary['missing_images']}",
        f"unreadable images: {summary['unreadable_images']}",
        f"invalid records: {summary['invalid_records']}",
        f"duplicate image references: {summary['duplicate_images']}",
        f"empty labels: {summary['empty_labels']}",
        f"train/validation overlap: {summary['overlap']}",
        f"unicode errors: {summary['unicode_errors']}",
        f"frozen rows excluded: {summary['frozen_excluded']}",
        f"source folder created: {summary.get('layout_created', False)}",
        f"target of {summary.get('target', TARGET)} reached: {summary['target_reached']}",
        "OCR text is not treated as human gold by this command.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
