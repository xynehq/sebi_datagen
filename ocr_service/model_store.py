"""SQLite registry for user-added OCR models."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from registry import (
    ModelError,
    ModelSpec,
    _REGISTRY_PATH,
    _USER_REGISTRY_PATH,
    _row_to_spec,
    slugify_model_id,
)

_DB_LOCK = threading.Lock()
_INITIALIZED = False

_READY = "ready"
_FAILED = "failed"


@dataclass
class CustomModelRow:
    id: str
    label: str
    description: str
    family: str
    model_type: str
    repo_url: str
    repo_id: str
    revision: str | None
    base_repo_id: str | None
    base_revision: str | None
    adapter_repo_id: str | None
    languages: str
    status: str
    enabled: bool
    is_default: bool
    failure_reason: str | None
    created_at: str
    updated_at: str
    subfolder: str | None = None
    job_id: str | None = None


def registry_db_path() -> Path:
    raw = os.environ.get("OCR_MODEL_REGISTRY_DB", "").strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        root = Path(__file__).resolve().parent.parent / "data" / "ocr_model_registry"
        path = root / "registry.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = registry_db_path()
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_store() -> None:
    global _INITIALIZED
    with _DB_LOCK:
        with _connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_models (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    family TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    repo_url TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    revision TEXT,
                    base_repo_id TEXT,
                    base_revision TEXT,
                    adapter_repo_id TEXT,
                    languages TEXT NOT NULL DEFAULT 'unknown',
                    status TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_jobs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES custom_models(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            _ensure_schema(conn)
        if not _INITIALIZED:
            _migrate_yaml_user_models()
        _INITIALIZED = True


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(custom_models)")}
    if "subfolder" not in cols:
        conn.execute("ALTER TABLE custom_models ADD COLUMN subfolder TEXT")


def _migrate_yaml_user_models() -> None:
    if not _USER_REGISTRY_PATH.exists():
        return
    payload = yaml.safe_load(_USER_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    rows = payload.get("models") or []
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id or get_custom_model(model_id) is not None:
            continue
        try:
            spec = _row_to_spec(row, user_added=True)
        except ModelError:
            continue
        adapter = spec.adapter
        now = _now()
        insert_custom_model(
            CustomModelRow(
                id=spec.id,
                label=spec.label,
                description=spec.description,
                family=spec.family,
                model_type="adapter" if adapter else "full",
                repo_url=f"https://huggingface.co/{adapter or spec.base}",
                repo_id=adapter or spec.base,
                revision=None,
                base_repo_id=spec.base if adapter else None,
                base_revision=None,
                adapter_repo_id=adapter,
                languages="unknown",
                status=_READY,
                enabled=True,
                is_default=False,
                failure_reason=None,
                created_at=now,
                updated_at=now,
            ),
            allow_existing=False,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def insert_custom_model(row: CustomModelRow, *, allow_existing: bool = False) -> None:
    init_store()
    with _DB_LOCK, _connect() as conn:
        if not allow_existing and conn.execute(
            "SELECT 1 FROM custom_models WHERE id = ?", (row.id,)
        ).fetchone():
            raise ModelError(f"A model named {row.id!r} is already registered.")
        conn.execute(
            """
            INSERT OR REPLACE INTO custom_models (
                id, label, description, family, model_type, repo_url, repo_id, revision,
                base_repo_id, base_revision, adapter_repo_id, languages, status, enabled,
                is_default, failure_reason, created_at, updated_at, subfolder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.id,
                row.label,
                row.description,
                row.family,
                row.model_type,
                row.repo_url,
                row.repo_id,
                row.revision,
                row.base_repo_id,
                row.base_revision,
                row.adapter_repo_id,
                row.languages,
                row.status,
                1 if row.enabled else 0,
                1 if row.is_default else 0,
                row.failure_reason,
                row.created_at,
                row.updated_at,
                row.subfolder,
            ),
        )


def update_custom_inference_meta(
    model_id: str,
    *,
    family: str | None = None,
    subfolder: str | None = None,
) -> None:
    init_store()
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [_now()]
    if family is not None:
        fields.append("family = ?")
        values.append(family)
    if subfolder is not None:
        fields.append("subfolder = ?")
        values.append(subfolder)
    values.append(model_id)
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            f"UPDATE custom_models SET {', '.join(fields)} WHERE id = ?",
            values,
        )


def update_custom_status(
    model_id: str,
    *,
    status: str,
    enabled: bool | None = None,
    failure_reason: str | None = None,
) -> None:
    init_store()
    fields = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, _now()]
    if enabled is not None:
        fields.append("enabled = ?")
        values.append(1 if enabled else 0)
    if failure_reason is not None:
        fields.append("failure_reason = ?")
        values.append(failure_reason)
    values.append(model_id)
    with _DB_LOCK, _connect() as conn:
        conn.execute(f"UPDATE custom_models SET {', '.join(fields)} WHERE id = ?", values)


def get_custom_model(model_id: str) -> CustomModelRow | None:
    init_store()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM custom_models WHERE id = ?", (model_id,)).fetchone()
    return _row_from_db(row) if row else None


def list_custom_models(*, include_disabled: bool = True) -> list[CustomModelRow]:
    init_store()
    with _connect() as conn:
        if include_disabled:
            rows = conn.execute("SELECT * FROM custom_models ORDER BY created_at").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM custom_models WHERE enabled = 1 AND status = ? ORDER BY created_at",
                (_READY,),
            ).fetchall()
    return [_row_from_db(r) for r in rows]


def _row_from_db(row: sqlite3.Row) -> CustomModelRow:
    return CustomModelRow(
        id=row["id"],
        label=row["label"],
        description=row["description"],
        family=row["family"],
        model_type=row["model_type"],
        repo_url=row["repo_url"],
        repo_id=row["repo_id"],
        revision=row["revision"],
        base_repo_id=row["base_repo_id"],
        base_revision=row["base_revision"],
        adapter_repo_id=row["adapter_repo_id"],
        languages=row["languages"],
        status=row["status"],
        enabled=bool(row["enabled"]),
        is_default=bool(row["is_default"]),
        failure_reason=row["failure_reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        subfolder=row["subfolder"] if "subfolder" in row.keys() else None,
    )


def custom_model_to_spec(row: CustomModelRow) -> ModelSpec:
    if row.model_type == "adapter":
        base = row.base_repo_id or ""
        adapter = row.adapter_repo_id or row.repo_id
    else:
        base = row.repo_id
        adapter = None
    yaml_row: dict[str, object] = {
        "id": row.id,
        "label": row.label,
        "base": base,
        "description": row.description,
        "family": row.family,
    }
    if adapter:
        yaml_row["adapter"] = adapter
    subfolder = (row.subfolder or "").strip()
    if subfolder:
        yaml_row["subfolder"] = subfolder
    elif row.family == "indicocr":
        yaml_row["subfolder"] = "weights/ocr"
    return _row_to_spec(yaml_row, user_added=True)


def selectable_custom_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for row in list_custom_models(include_disabled=False):
        if row.status != _READY or not row.enabled:
            continue
        specs.append(custom_model_to_spec(row))
    return specs


def builtin_ids() -> set[str]:
    rows = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    models = rows.get("models") or []
    return {str(m.get("id") or "").strip() for m in models if isinstance(m, dict)}


def reserve_model_id(label: str, model_id: str = "") -> str:
    chosen = re_sub_id(model_id or slugify_model_id(label))
    taken = builtin_ids() | {r.id for r in list_custom_models()}
    if chosen not in taken:
        return chosen
    suffix = 2
    while f"{chosen}-{suffix}" in taken:
        suffix += 1
    return f"{chosen}-{suffix}"


def re_sub_id(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "model"


def create_job(model_id: str, message: str = "Waiting to download.") -> str:
    init_store()
    job_id = uuid.uuid4().hex
    now = _now()
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO model_jobs (id, model_id, status, progress, message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, model_id, "waiting", 0.0, message, now, now),
        )
    return job_id


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
) -> None:
    init_store()
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [_now()]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if message is not None:
        fields.append("message = ?")
        values.append(message)
    values.append(job_id)
    with _DB_LOCK, _connect() as conn:
        conn.execute(f"UPDATE model_jobs SET {', '.join(fields)} WHERE id = ?", values)


def get_job(job_id: str) -> dict[str, Any] | None:
    init_store()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM model_jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    return dict(row)


def delete_custom_model(model_id: str) -> None:
    init_store()
    if model_id in builtin_ids():
        raise ModelError("Built-in models cannot be removed.")
    with _DB_LOCK, _connect() as conn:
        conn.execute("DELETE FROM model_jobs WHERE model_id = ?", (model_id,))
        cur = conn.execute("DELETE FROM custom_models WHERE id = ?", (model_id,))
        if cur.rowcount == 0:
            raise ModelError(f"No custom model {model_id!r} was found.")


def set_user_default_model(model_id: str | None) -> None:
    init_store()
    with _DB_LOCK, _connect() as conn:
        conn.execute("UPDATE custom_models SET is_default = 0")
        if model_id:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('default_model_id', ?)",
                (model_id,),
            )
            conn.execute("UPDATE custom_models SET is_default = 1 WHERE id = ?", (model_id,))
        else:
            conn.execute("DELETE FROM settings WHERE key = 'default_model_id'")


def get_user_default_model_id() -> str | None:
    init_store()
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'default_model_id'"
        ).fetchone()
    return str(row["value"]) if row else None


def set_enabled(model_id: str, enabled: bool) -> None:
    row = get_custom_model(model_id)
    if row is None:
        raise ModelError(f"Unknown custom model {model_id!r}.")
    if enabled and row.status != _READY:
        raise ModelError("Only models that finished download and testing can be enabled.")
    init_store()
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "UPDATE custom_models SET enabled = ?, updated_at = ? WHERE id = ?",
            (1 if enabled else 0, _now(), model_id),
        )


def custom_model_public_dict(row: CustomModelRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "label": row.label,
        "description": row.description,
        "family": row.family,
        "model_type": row.model_type,
        "repo_id": row.repo_id,
        "revision": row.revision,
        "base_repo_id": row.base_repo_id,
        "languages": row.languages,
        "status": row.status,
        "enabled": row.enabled,
        "is_default": row.is_default,
        "failure_reason": row.failure_reason,
        "user_added": True,
    }
