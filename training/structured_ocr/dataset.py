"""Build Chitrapathak SFT JSONL from verified structured manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.structured_ocr.manifest import assert_training_ready, validate_manifest_row
from training.structured_ocr.prompts import LINE_OCR_PROMPT, STRUCTURED_PAGE_PROMPT
from training.structured_ocr.targets import normalize_target


def row_to_sft(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    structured_prompt: bool = True,
) -> dict[str, Any] | None:
    issues = validate_manifest_row(row, repo_root=repo_root)
    if issues:
        return None
    task = str(row.get("task") or row.get("sample_type") or "line_crop")
    image = str(row.get("image") or row.get("image_path") or row.get("image_filename") or "")
    text = normalize_target(str(row.get("text") or row.get("expected_text") or row.get("target") or ""))
    out: dict[str, Any] = {"image": image, "text": text, "task": task}
    if structured_prompt and task == "full_page":
        out["prompt"] = STRUCTURED_PAGE_PROMPT
    elif row.get("prompt"):
        out["prompt"] = str(row["prompt"])
    else:
        out["prompt"] = LINE_OCR_PROMPT
    for key in ("sample_id", "document_id", "page", "language", "split", "category"):
        if row.get(key) is not None:
            out[key] = row[key]
    return out


def export_sft_jsonl(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    repo_root: Path,
    require_verified: bool = True,
    structured_prompt: bool = True,
) -> int:
    if require_verified:
        assert_training_ready(rows, repo_root=repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = row_to_sft(row, repo_root=repo_root, structured_prompt=structured_prompt)
            if item is None:
                continue
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    return written
