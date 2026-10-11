"""Dataset loading, path resolution, and fail-fast validation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


class DatasetError(ValueError):
    """Raised when the evaluation dataset is malformed."""


@dataclass(frozen=True)
class EvalSample:
    index: int
    image_filename: str
    image_path: str
    expected_text: str
    issue_type: str
    difficulty: Optional[str]  # "hard" | "normal" | None


def resolve_image_path(dataset_root: str | Path, image_filename: str) -> Path:
    """Resolve ``/images/0001.png`` under dataset_root (never FS root)."""
    root = Path(dataset_root)
    rel = str(image_filename).lstrip("/")
    return root / rel


def load_validation_dataset(
    dataset_root: str | Path,
    jsonl_name: str = "validation.jsonl",
    limit: Optional[int] = None,
) -> List[EvalSample]:
    """Load and validate validation.jsonl + images/.

    Fails before inference on:
    - missing JSONL / images dir
    - invalid JSON rows
    - missing required fields
    - missing image files
    - duplicate image_filename
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {root}")

    jsonl_path = root / jsonl_name
    if not jsonl_path.is_file():
        raise DatasetError(f"Missing {jsonl_name} in {root}")

    images_dir = root / "images"
    if not images_dir.is_dir():
        raise DatasetError(f"Missing images/ directory in {root}")

    samples: List[EvalSample] = []
    seen: set[str] = set()
    errors: List[str] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSON ({e})")
                continue

            if not isinstance(row, dict):
                errors.append(f"line {line_no}: expected object, got {type(row).__name__}")
                continue

            if "image_filename" not in row:
                errors.append(f"line {line_no}: missing image_filename")
                continue
            if "expected_text" not in row:
                errors.append(f"line {line_no}: missing expected_text")
                continue

            image_filename = row["image_filename"]
            expected_text = row["expected_text"]
            if not isinstance(image_filename, str):
                errors.append(f"line {line_no}: image_filename must be str")
                continue
            if not isinstance(expected_text, str):
                errors.append(f"line {line_no}: expected_text must be str")
                continue

            if image_filename in seen:
                errors.append(f"line {line_no}: duplicate image_filename {image_filename!r}")
                continue
            seen.add(image_filename)

            image_path = resolve_image_path(root, image_filename)
            if not image_path.is_file():
                errors.append(
                    f"line {line_no}: image not found for {image_filename!r} "
                    f"(resolved {image_path})"
                )
                continue

            issue_type = row.get("issue_type")
            if issue_type is None:
                issue_type = "unspecified"
            elif not isinstance(issue_type, str):
                errors.append(f"line {line_no}: issue_type must be str")
                continue

            difficulty = row.get("difficulty")
            if difficulty is not None:
                if not isinstance(difficulty, str):
                    errors.append(f"line {line_no}: difficulty must be str")
                    continue
                difficulty = difficulty.strip().lower()
                if difficulty not in ("hard", "normal"):
                    errors.append(
                        f"line {line_no}: difficulty must be 'hard' or 'normal', "
                        f"got {difficulty!r}"
                    )
                    continue

            samples.append(
                EvalSample(
                    index=len(samples) + 1,
                    image_filename=image_filename,
                    image_path=str(image_path),
                    expected_text=expected_text,  # immutable GT — never mutate
                    issue_type=issue_type,
                    difficulty=difficulty,
                )
            )

            if limit is not None and len(samples) >= limit:
                break

    if errors:
        preview = "\n  ".join(errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise DatasetError(
            f"Dataset validation failed with {len(errors)} error(s):\n  {preview}{more}"
        )

    if not samples:
        raise DatasetError(f"No samples loaded from {jsonl_path}")

    return samples


def apply_hard_issue_type_mapping(
    samples: Sequence[EvalSample],
    hard_issue_types: Sequence[str],
) -> List[EvalSample]:
    """Return new samples with difficulty set from configured issue_type mapping.

    Only used when CLI ``--hard-issue-types`` is provided. Does not invent
    difficulty silently.
    """
    hard_set = {t.strip() for t in hard_issue_types if t.strip()}
    out: List[EvalSample] = []
    for s in samples:
        if s.difficulty is not None:
            out.append(s)
            continue
        diff = "hard" if s.issue_type in hard_set else "normal"
        out.append(
            EvalSample(
                index=s.index,
                image_filename=s.image_filename,
                image_path=s.image_path,
                expected_text=s.expected_text,
                issue_type=s.issue_type,
                difficulty=diff,
            )
        )
    return out
