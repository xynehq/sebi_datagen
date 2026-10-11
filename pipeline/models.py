"""Resolve local weights first. Hugging Face is a label, not a download."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ModelMissing(RuntimeError):
    """A model file is not on this machine."""


def resolve_model(spec: dict[str, Any], *, offline: bool) -> str | None:
    """Return a local path when one exists.

    When the path is missing, raise ModelMissing with the exact variable and
    how to obtain the files. This function never downloads.
    """
    env_name = str(spec.get("env") or "")
    raw = os.environ.get(env_name, "").strip() if env_name else ""
    if raw and Path(raw).exists():
        return raw
    huggingface = spec.get("huggingface")
    obtain = str(spec.get("obtain") or "Set the local path and rerun.")
    if raw and not Path(raw).exists():
        raise ModelMissing(
            f"{env_name} is set to {raw}, but that path does not exist. {obtain}"
        )
    if huggingface and not offline:
        return None
    if huggingface and offline:
        raise ModelMissing(
            f"{env_name or 'model'} is not set. "
            f"Hugging Face id {huggingface} is configured, but --offline forbids a download. "
            f"{obtain}"
        )
    if not raw:
        raise ModelMissing(f"{env_name or 'model'} is not set. {obtain}")
    return raw
