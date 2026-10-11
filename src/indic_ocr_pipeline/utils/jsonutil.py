"""JSON helpers for manifest records."""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert pipeline values into JSON-compatible objects."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be stored in a manifest")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    as_list = getattr(value, "as_list", None)
    if callable(as_list):
        return to_jsonable(as_list())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def dumps_record(record: dict[str, Any]) -> str:
    """Serialize one manifest record on a single line."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_json_value(value: Any, field_name: str = "metadata") -> None:
    """Raise ``ValueError`` when ``value`` cannot be stored in JSON."""
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_value(item, f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            assert_json_value(item, f"{field_name}.{key}")
        return
    raise ValueError(f"{field_name} contains unsupported type {type(value).__name__}")
