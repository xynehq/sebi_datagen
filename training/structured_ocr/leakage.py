"""Split integrity and duplicate checks for structured manifests."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_document_split_leaks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """One document_id must not appear in more than one split."""
    by_doc: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc = str(row.get("document_id") or "")
        split = str(row.get("split") or "")
        if doc and split:
            by_doc[doc].add(split)
    issues = []
    for doc, splits in sorted(by_doc.items()):
        if len(splits) > 1:
            issues.append(f"document_split_leak:{doc}:{','.join(sorted(splits))}")
    return issues


def check_duplicate_image_hashes(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> list[str]:
    seen: dict[str, str] = {}
    issues: list[str] = []
    for row in rows:
        image = str(row.get("image") or row.get("image_path") or "")
        if not image or image.startswith("http"):
            continue
        path = Path(image)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            continue
        digest = sha256_file(path)
        sid = str(row.get("sample_id") or row.get("id") or "")
        if digest in seen and seen[digest] != sid:
            issues.append(f"duplicate_image_hash:{digest[:12]}:{seen[digest]}:{sid}")
        else:
            seen[digest] = sid
    return issues
