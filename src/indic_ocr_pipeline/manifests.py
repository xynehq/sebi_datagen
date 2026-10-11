"""Append-only JSONL manifests.

A completed row is frozen. A second write of the same id is a resume skip
when the payload matches, and an error when it does not. Rewrites go to a
temporary file in the same directory and then replace the manifest.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from indic_ocr_pipeline.errors import ImmutableFieldError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.schemas import StageStatus
from indic_ocr_pipeline.utils.jsonutil import dumps_record

ALLOWED_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    StageStatus.PENDING.value: frozenset(
        {StageStatus.RUNNING.value, StageStatus.SKIPPED.value}
    ),
    StageStatus.RUNNING.value: frozenset(
        {
            StageStatus.COMPLETE.value,
            StageStatus.FAILED.value,
            StageStatus.SKIPPED.value,
        }
    ),
    StageStatus.FAILED.value: frozenset({StageStatus.RUNNING.value}),
    StageStatus.SKIPPED.value: frozenset({StageStatus.RUNNING.value}),
    StageStatus.COMPLETE.value: frozenset(),
}

DEFAULT_IMMUTABLE_FIELDS = frozenset(
    {
        "document_id",
        "page_id",
        "region_id",
        "line_id",
        "parent_line_id",
        "crop_id",
        "prediction_id",
        "candidate_id",
        "review_id",
        "verified_id",
        "sample_id",
        "sha256",
        "source_sha256",
        "image_sha256",
        "source_path",
        "original_filename",
        "bbox",
        "parent_bbox",
        "image_path",
        "gold_text",
        "text",
        "raw_text",
        "crop_type",
        "page_number",
        "original_candidate_texts",
        "texts_by_backend",
        "split",
        "final_text",
    }
)

# filename, id field
MANIFEST_SPECS: dict[str, tuple[str, str]] = {
    "documents": ("documents.jsonl", "document_id"),
    "pages": ("pages.jsonl", "page_id"),
    "regions": ("regions.jsonl", "region_id"),
    "lines": ("lines.jsonl", "line_id"),
    "crops": ("crops.jsonl", "crop_id"),
    "predictions": ("predictions.jsonl", "prediction_id"),
    "candidates": ("candidates.jsonl", "candidate_id"),
    "review": ("review.jsonl", "review_id"),
    "review_batch": ("review_batch.jsonl", "batch_item_id"),
    "verified": ("verified.jsonl", "verified_id"),
    "gold": ("gold.jsonl", "sample_id"),
}


class JsonlManifest:
    """A JSONL file keyed by one string field."""

    def __init__(
        self,
        path: Path,
        id_field: str,
        *,
        immutable_fields: frozenset[str] | None = None,
    ) -> None:
        if not id_field:
            raise ValueError("id_field is required")
        self.path = Path(path)
        self.id_field = id_field
        self.immutable_fields = (
            DEFAULT_IMMUTABLE_FIELDS if immutable_fields is None else immutable_fields
        )

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self) -> list[dict[str, Any]]:
        """Return every record. Duplicate ids are an error."""
        with self._exclusive():
            return self._load_unlocked()

    def get(self, record_id: str) -> dict[str, Any] | None:
        """Return one record, or ``None`` when the id is absent."""
        for record in self.load():
            if record[self.id_field] == record_id:
                return record
        return None

    def append(self, record: Mapping[str, Any]) -> Literal["created", "skipped"]:
        """Add a record.

        Returns ``skipped`` when the same id is already stored with the same
        content. Raises when the same id is stored with different content.
        """
        payload = _copy_record(record, self.id_field)
        with self._exclusive():
            records = self._load_unlocked()
            for existing in records:
                if existing[self.id_field] != payload[self.id_field]:
                    continue
                if existing == payload:
                    return "skipped"
                raise ManifestConflictError(
                    f"{self.id_field}={payload[self.id_field]} "
                    "already exists with different content"
                )
            records.append(payload)
            self._rewrite_unlocked(records)
            return "created"

    def update(self, record_id: str, changes: Mapping[str, Any]) -> Literal["updated", "skipped"]:
        """Update mutable fields on a record that is not complete.

        Identity fields can be filled when they are absent. They cannot change
        once they have a value. A ``complete`` record cannot change at all.
        """
        if not isinstance(record_id, str) or not record_id:
            raise SchemaError("record_id must be a non-empty string")
        change_map = dict(changes)
        if self.id_field in change_map and change_map[self.id_field] != record_id:
            raise ImmutableFieldError(f"cannot change {self.id_field}")
        with self._exclusive():
            records = self._load_unlocked()
            index = next(
                (
                    position
                    for position, row in enumerate(records)
                    if row[self.id_field] == record_id
                ),
                None,
            )
            if index is None:
                raise ManifestConflictError(f"no record with {self.id_field}={record_id}")
            current = records[index]
            updated = dict(current)
            updated.update(change_map)
            if updated == current:
                return "skipped"
            if current.get("status") == StageStatus.COMPLETE.value:
                raise ImmutableFieldError("complete records are frozen")
            _check_status_transition(current.get("status"), updated.get("status"))
            for key, value in change_map.items():
                if key == "status" or key not in self.immutable_fields:
                    continue
                if key in current and current[key] is not None and current[key] != value:
                    raise ImmutableFieldError(f"cannot change immutable field {key}")
            records[index] = updated
            self._rewrite_unlocked(records)
            return "updated"

    def commit_batch(
        self,
        records: list[dict[str, Any]],
        *,
        fill_unscored: bool = False,
    ) -> dict[str, int]:
        """Insert new rows and skip identical rows in one rewrite.

        ``fill_unscored`` replaces a row whose ``confidence_score`` and
        ``risk_score`` are both null. Any other content change is a conflict.
        Immutable fields cannot change. Rows already in the file but absent
        from ``records`` are left in place only when ``fill_unscored`` is
        false; a scored manifest must account for every existing id.
        """
        payloads = [_copy_record(record, self.id_field) for record in records]
        seen: set[str] = set()
        for payload in payloads:
            record_id = payload[self.id_field]
            if record_id in seen:
                raise ManifestConflictError(f"duplicate {self.id_field}={record_id} in batch")
            seen.add(record_id)
        created = 0
        skipped = 0
        updated = 0
        with self._exclusive():
            stored = self._load_unlocked()
            by_id = {row[self.id_field]: index for index, row in enumerate(stored)}
            if fill_unscored:
                missing = [record_id for record_id in by_id if record_id not in seen]
                if missing:
                    listed = ", ".join(missing[:10])
                    raise ManifestConflictError(
                        f"{self.path.name} has rows that this ranking does not cover: {listed}"
                    )
            changed = False
            for payload in payloads:
                record_id = payload[self.id_field]
                index = by_id.get(record_id)
                if index is None:
                    stored.append(payload)
                    created += 1
                    changed = True
                    continue
                current = stored[index]
                if current == payload:
                    skipped += 1
                    continue
                if current.get("status") == StageStatus.COMPLETE.value:
                    raise ImmutableFieldError(
                        f"complete record {self.id_field}={record_id} is frozen"
                    )
                for key in self.immutable_fields:
                    stored_value = current.get(key)
                    if stored_value is not None and stored_value != payload.get(key):
                        raise ImmutableFieldError(
                            f"cannot change immutable field {key} on {self.id_field}={record_id}"
                        )
                confidence = current.get("confidence_score")
                risk = current.get("risk_score")
                unscored = confidence is None and risk is None
                if not (fill_unscored and unscored):
                    raise ManifestConflictError(
                        f"{self.id_field}={record_id} already exists with different content"
                    )
                stored[index] = payload
                updated += 1
                changed = True
            if changed:
                self._rewrite_unlocked(stored)
        return {"created": created, "skipped": skipped, "updated": updated}

    def _load_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ManifestConflictError(
                        f"{self.path}:{line_number} is not valid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise SchemaError(f"{self.path}:{line_number} must be a JSON object")
                record_id = value.get(self.id_field)
                if not isinstance(record_id, str) or not record_id:
                    raise SchemaError(
                        f"{self.path}:{line_number} is missing string field {self.id_field}"
                    )
                if record_id in seen:
                    raise ManifestConflictError(
                        f"duplicate {self.id_field}={record_id} in {self.path}"
                    )
                seen.add(record_id)
                records.append(value)
        return records

    def _rewrite_unlocked(self, records: list[dict[str, Any]]) -> None:
        payload = "".join(dumps_record(record) + "\n" for record in records)
        descriptor, temp_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
            raise


def manifest_path(manifests_dir: Path, name: str) -> Path:
    """Return the JSONL path for a known manifest name."""
    try:
        filename, _id_field = MANIFEST_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(MANIFEST_SPECS))
        raise SchemaError(f"unknown manifest {name!r}; expected one of: {known}") from exc
    return Path(manifests_dir) / filename


def open_manifest(manifests_dir: Path, name: str) -> JsonlManifest:
    """Open a manifest declared in ``MANIFEST_SPECS``."""
    filename, id_field = MANIFEST_SPECS[name]
    return JsonlManifest(Path(manifests_dir) / filename, id_field)


def _copy_record(record: Mapping[str, Any], id_field: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise SchemaError("manifest record must be an object")
    payload = dict(record)
    record_id = payload.get(id_field)
    if not isinstance(record_id, str) or not record_id:
        raise SchemaError(f"{id_field} must be a non-empty string")
    return payload


def _check_status_transition(old_status: object, new_status: object) -> None:
    if old_status == new_status:
        return
    if not isinstance(new_status, str):
        raise SchemaError("status must be a string")
    if old_status is None:
        allowed = frozenset({StageStatus.PENDING.value, StageStatus.RUNNING.value})
    else:
        if not isinstance(old_status, str) or old_status not in ALLOWED_STATUS_TRANSITIONS:
            raise SchemaError(f"unknown status {old_status!r}")
        allowed = ALLOWED_STATUS_TRANSITIONS[old_status]
    if new_status not in allowed:
        raise ImmutableFieldError(f"cannot change status from {old_status} to {new_status}")
