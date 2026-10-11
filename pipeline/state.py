"""Durable pipeline state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.graph import ORDER, descendants

NOT_STARTED = "NOT_STARTED"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
INVALIDATED = "INVALIDATED"


def now() -> str:
    return datetime.now(UTC).isoformat()


def empty_state() -> dict[str, Any]:
    return {"version": 1, "updated_at": None, "sealed": False, "stages": {}}


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_state()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("stages", {})
    payload.setdefault("sealed", False)
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def blank_stage(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": NOT_STARTED,
        "started_at": None,
        "ended_at": None,
        "input_fingerprint": None,
        "output_fingerprint": None,
        "config_hash": None,
        "code_version": None,
        "row_counts": {},
        "errors": [],
        "module": None,
    }


def sync_invalidation(state: dict[str, Any], fingerprints: dict[str, dict[str, str]]) -> list[str]:
    """Mark completed stages INVALIDATED when their inputs or config changed.

    Downstream completed stages are invalidated with them. Returns the names
    that changed status.
    """
    changed: list[str] = []
    stages = state["stages"]
    for name in ORDER:
        record = stages.get(name)
        if not record or record.get("status") != COMPLETE:
            continue
        current = fingerprints[name]
        if (
            record.get("input_fingerprint") != current["input"]
            or record.get("config_hash") != current["config"]
            or record.get("output_fingerprint") != current["output"]
        ):
            record["status"] = INVALIDATED
            record["errors"] = ["input, config, or output changed"]
            changed.append(name)
            for child in descendants(name):
                child_record = stages.get(child)
                if child_record and child_record.get("status") == COMPLETE:
                    child_record["status"] = INVALIDATED
                    child_record["errors"] = [f"invalidated because {name} changed"]
                    changed.append(child)
    return changed
