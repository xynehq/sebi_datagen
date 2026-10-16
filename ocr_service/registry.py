"""Model list, cache status, and Hugging Face downloads.

Repo ids come from models.yaml and the environment. A local filesystem path
is refused so a production host cannot be pointed at a developer machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_SERVICE_DIR = Path(__file__).resolve().parent
_REGISTRY_PATH = _SERVICE_DIR / "models.yaml"


class ModelError(Exception):
    """A model cannot be selected or downloaded."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    base: str
    adapter: str | None
    configured: bool
    family: str = "chitrapathak"
    subfolder: str | None = None
    detail: str | None = None


def cache_dir() -> Path:
    raw = os.environ.get("MODEL_CACHE_DIR", "").strip()
    path = Path(raw) if raw else _SERVICE_DIR / "model_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_specs() -> list[ModelSpec]:
    payload = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    rows = payload.get("models")
    if not isinstance(rows, list) or not rows:
        raise ModelError("models.yaml has no models")
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ModelError("models.yaml rows must be mappings")
        model_id = str(row.get("id") or "").strip()
        label = str(row.get("label") or "").strip()
        base = str(row.get("base") or "").strip()
        description = " ".join(str(row.get("description") or "").split())
        if not model_id or not label or not base:
            raise ModelError("each model needs id, label, and base")
        if model_id in seen:
            raise ModelError(f"duplicate model id: {model_id}")
        seen.add(model_id)
        _require_repo_id(base, what=f"{model_id} base")
        adapter, configured, detail = _adapter(row, model_id)
        family = str(row.get("family") or "chitrapathak").strip()
        if family not in {"chitrapathak", "indicocr"}:
            raise ModelError(f"{model_id} family must be chitrapathak or indicocr")
        subfolder = str(row.get("subfolder") or "").strip() or None
        if subfolder and (".." in subfolder or subfolder.startswith(("/", "."))):
            raise ModelError(f"{model_id} subfolder must stay inside the repo")
        specs.append(
            ModelSpec(
                id=model_id,
                label=label,
                description=description,
                base=base,
                adapter=adapter,
                configured=configured,
                family=family,
                subfolder=subfolder,
                detail=detail,
            )
        )
    return specs


def get_spec(model_id: str) -> ModelSpec:
    for spec in load_specs():
        if spec.id == model_id:
            return spec
    known = ", ".join(spec.id for spec in load_specs())
    raise ModelError(f"Unknown model {model_id!r}. Known models: {known}", status_code=404)


def model_status(spec: ModelSpec) -> dict[str, object]:
    if not spec.configured:
        cached = False
        status = "repo_not_set"
    elif _spec_cached(spec):
        cached = True
        status = "ready"
    else:
        cached = False
        status = "not_downloaded"
    return {
        "id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "base": spec.base,
        "adapter": spec.adapter,
        "cached": cached,
        "status": status,
        "detail": spec.detail,
    }


def list_models() -> list[dict[str, object]]:
    return [model_status(spec) for spec in load_specs()]


def ensure_weights(spec: ModelSpec) -> tuple[Path, Path | None]:
    """Download any missing repo and return local snapshot directories."""
    if not spec.configured:
        raise ModelError(spec.detail or "model repo is not set", status_code=409)
    base = _download(spec.base, subfolder=spec.subfolder)
    adapter = _download(spec.adapter) if spec.adapter else None
    return base, adapter


def _adapter(row: dict[str, object], model_id: str) -> tuple[str | None, bool, str | None]:
    if "adapter" in row and "adapter_env" in row:
        raise ModelError(f"{model_id} cannot set both adapter and adapter_env")
    if "adapter_env" in row:
        env_name = str(row.get("adapter_env") or "").strip()
        if not env_name:
            raise ModelError(f"{model_id} adapter_env is empty")
        value = os.environ.get(env_name, "").strip()
        if not value:
            return (
                None,
                False,
                f"Set {env_name} to a Hugging Face repo id. The first run downloads it.",
            )
        _require_repo_id(value, what=env_name)
        return value, True, None
    if "adapter" in row and row.get("adapter"):
        adapter = str(row["adapter"]).strip()
        _require_repo_id(adapter, what=f"{model_id} adapter")
        return adapter, True, None
    return None, True, None


def _require_repo_id(value: str, *, what: str) -> None:
    if value.startswith(("/", ".", "~")) or "\\" in value or ":" in value:
        raise ModelError(f"{what} must be a Hugging Face repo id, not a local path")
    if value.count("/") != 1 or any(part == "" for part in value.split("/")):
        raise ModelError(f"{what} must look like org/name, got {value!r}")


def _spec_cached(spec: ModelSpec) -> bool:
    if not _repo_cached(spec.base, subfolder=spec.subfolder):
        return False
    if spec.adapter and not _repo_cached(spec.adapter):
        return False
    return True


def _repo_cached(repo_id: str, *, subfolder: str | None = None) -> bool:
    try:
        _snapshot(repo_id, local_files_only=True, subfolder=subfolder)
    except Exception:
        return False
    return True


def _download(repo_id: str, *, subfolder: str | None = None) -> Path:
    return Path(_snapshot(repo_id, local_files_only=False, subfolder=subfolder))


def _snapshot(repo_id: str, *, local_files_only: bool, subfolder: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN", "").strip() or None
    patterns = [f"{subfolder.strip('/')}/**"] if subfolder else None
    return snapshot_download(
        repo_id,
        cache_dir=str(cache_dir()),
        token=token,
        local_files_only=local_files_only,
        allow_patterns=patterns,
    )
