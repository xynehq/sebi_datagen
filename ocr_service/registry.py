"""Model list, cache status, and Hugging Face downloads.

Built-in models live in models.yaml. Anyone can add more from the page; those
rows are stored in user_models.yaml. Local filesystem paths are refused.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SERVICE_DIR = Path(__file__).resolve().parent
_REGISTRY_PATH = _SERVICE_DIR / "models.yaml"
_USER_REGISTRY_PATH = _SERVICE_DIR / "user_models.yaml"
_FAMILIES = frozenset({"chitrapathak", "indicocr", "vlm", "trocr"})
_HF_HOSTS = ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/")


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
    adapter_path_env: str | None = None
    detail: str | None = None
    user_added: bool = False


def cache_dir() -> Path:
    raw = os.environ.get("MODEL_CACHE_DIR", "").strip()
    path = Path(raw) if raw else _SERVICE_DIR / "model_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_registry_path() -> Path:
    return _USER_REGISTRY_PATH


def load_specs() -> list[ModelSpec]:
    from model_store import init_store, list_custom_models, custom_model_to_spec

    init_store()
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for row in _read_model_rows(_REGISTRY_PATH, required=True):
        spec = _row_to_spec(row, user_added=False)
        if spec.id in seen:
            raise ModelError(f"duplicate model id: {spec.id}")
        seen.add(spec.id)
        specs.append(spec)
    for custom in list_custom_models(include_disabled=True):
        if custom.id in seen:
            raise ModelError(f"custom model id {custom.id!r} already exists in models.yaml")
        if custom.status == "ready" and custom.enabled:
            seen.add(custom.id)
            specs.append(custom_model_to_spec(custom))
        elif custom.status == "ready" and not custom.enabled:
            continue
        elif custom.status in ("failed", "waiting", "downloading", "validating", "testing"):
            continue
    if _USER_REGISTRY_PATH.exists():
        for row in _read_model_rows(_USER_REGISTRY_PATH, required=False):
            model_id = str(row.get("id") or "").strip()
            if model_id in seen:
                continue
            spec = _row_to_spec(row, user_added=True)
            seen.add(spec.id)
            specs.append(spec)
    if not specs:
        raise ModelError("models.yaml has no models")
    return specs


def get_spec(model_id: str) -> ModelSpec:
    for spec in load_specs():
        if spec.id == model_id:
            return spec
    known = ", ".join(spec.id for spec in load_specs())
    raise ModelError(f"Unknown model {model_id!r}. Known models: {known}", status_code=404)


def model_status(spec: ModelSpec) -> dict[str, object]:
    from model_store import get_custom_model

    custom = get_custom_model(spec.id) if spec.user_added else None
    if custom is not None and custom.status != "ready":
        return {
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "base": spec.base,
            "adapter": spec.adapter,
            "cached": False,
            "status": custom.status,
            "detail": custom.failure_reason,
            "user_added": True,
        }
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
        "user_added": spec.user_added,
    }


def list_models() -> list[dict[str, object]]:
    from model_store import custom_model_public_dict, init_store, list_custom_models

    init_store()
    rows = [model_status(spec) for spec in load_specs()]
    seen = {str(row["id"]) for row in rows}
    for custom in list_custom_models(include_disabled=True):
        if custom.id in seen:
            continue
        spec = None
        try:
            from model_store import custom_model_to_spec

            spec = custom_model_to_spec(custom)
        except ModelError:
            spec = None
        entry = custom_model_public_dict(custom)
        entry["base"] = spec.base if spec else custom.base_repo_id or custom.repo_id
        entry["adapter"] = spec.adapter if spec else custom.adapter_repo_id
        entry["cached"] = bool(spec and _spec_cached(spec))
        rows.append(entry)
        seen.add(custom.id)
    return rows


def list_user_models() -> list[ModelSpec]:
    from model_store import init_store, list_custom_models, custom_model_to_spec

    init_store()
    specs = [custom_model_to_spec(row) for row in list_custom_models()]
    if _USER_REGISTRY_PATH.exists():
        seen = {s.id for s in specs}
        for row in _read_model_rows(_USER_REGISTRY_PATH, required=False):
            model_id = str(row.get("id") or "").strip()
            if model_id in seen:
                continue
            specs.append(_row_to_spec(row, user_added=True))
    return specs


def normalize_hf_ref(value: str, *, what: str) -> str:
    """Accept org/name or a huggingface.co link and return org/name."""
    text = (value or "").strip()
    if not text:
        raise ModelError(f"{what} is empty")
    lowered = text.lower()
    for host in _HF_HOSTS:
        if lowered.startswith(host):
            text = text[len(host) :]
            break
    text = text.strip().strip("/")
    if text.endswith(".git"):
        text = text[: -len(".git")]
    # Drop tree/blob/resolve suffixes from copied URLs.
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        text = f"{parts[0]}/{parts[1]}"
    _require_repo_id(text, what=what)
    return text


def slugify_model_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (label or "").lower()).strip("-")
    return slug or "model"


def add_user_model(
    *,
    label: str,
    base: str,
    adapter: str = "",
    family: str = "chitrapathak",
    description: str = "",
    model_id: str = "",
) -> ModelSpec:
    """Register a model from the simple form (stored in the SQLite registry)."""
    from model_store import CustomModelRow, insert_custom_model, reserve_model_id, _now

    name = " ".join((label or "").split())
    if not name:
        raise ModelError("Give the model a display name.")
    family_id = str(family or "chitrapathak").strip().lower()
    if family_id not in _FAMILIES:
        raise ModelError("Pick Chitrapathak or IndicOCR.")
    base_id = normalize_hf_ref(base, what="base model")
    adapter_raw = (adapter or "").strip()
    adapter_id = normalize_hf_ref(adapter_raw, what="adapter") if adapter_raw else None
    requested = re.sub(r"[^a-z0-9-]+", "-", (model_id or "").lower()).strip("-")
    chosen_id = reserve_model_id(name, model_id)
    if requested and requested != chosen_id:
        raise ModelError(f"A model named {requested!r} is already in the list.")
    note = " ".join((description or "").split()) or (
        f"Added from the page. Base `{base_id}`"
        + (f", adapter `{adapter_id}`." if adapter_id else ".")
    )
    yaml_row: dict[str, object] = {
        "id": chosen_id,
        "label": name,
        "base": base_id,
        "description": note,
        "family": family_id,
    }
    if adapter_id:
        yaml_row["adapter"] = adapter_id
    if family_id == "indicocr":
        yaml_row["subfolder"] = "weights/ocr"
    spec = _row_to_spec(yaml_row, user_added=True)
    now = _now()
    primary = adapter_id or base_id
    insert_custom_model(
        CustomModelRow(
            id=chosen_id,
            label=name,
            description=note,
            family=family_id,
            model_type="adapter" if adapter_id else "full",
            repo_url=f"https://huggingface.co/{primary}",
            repo_id=primary,
            revision=None,
            base_repo_id=base_id if adapter_id else None,
            base_revision=None,
            adapter_repo_id=adapter_id,
            languages="unknown",
            status="ready",
            enabled=True,
            is_default=False,
            failure_reason=None,
            created_at=now,
            updated_at=now,
        ),
    )
    return spec


def remove_user_model(model_id: str) -> None:
    """Remove a page-added model. Built-in models.yaml rows cannot be removed here."""
    from model_store import delete_custom_model, get_custom_model

    target = (model_id or "").strip()
    if not target:
        raise ModelError("Pick a model to remove.")
    builtin = {
        str(row.get("id") or "").strip() for row in _read_model_rows(_REGISTRY_PATH, required=True)
    }
    if target in builtin:
        raise ModelError("Built-in models cannot be removed from the page.")
    if get_custom_model(target) is not None:
        delete_custom_model(target)
        return
    payload = _load_user_payload()
    models = payload.get("models") or []
    if not isinstance(models, list):
        raise ModelError("user_models.yaml is invalid")
    kept = [
        row
        for row in models
        if isinstance(row, dict) and str(row.get("id") or "").strip() != target
    ]
    if len(kept) == len(models):
        raise ModelError(f"No page-added model {target!r} was found.")
    if kept:
        _write_user_payload({"models": kept})
    elif _USER_REGISTRY_PATH.exists():
        _USER_REGISTRY_PATH.unlink()


def ensure_weights(spec: ModelSpec) -> tuple[Path, Path | None]:
    """Download any missing repo and return local snapshot directories."""
    if not spec.configured:
        raise ModelError(spec.detail or "model repo is not set", status_code=409)
    base = _download(spec.base, subfolder=spec.subfolder)
    adapter = _resolve_adapter_path(spec)
    return base, adapter


def local_adapter_from_env(env_name: str) -> Path | None:
    """PEFT adapter directory from an environment variable, when present."""
    value = os.environ.get(env_name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ModelError(f"{env_name} must be a directory, got {value!r}")
    if not (path / "adapter_config.json").is_file():
        raise ModelError(f"{env_name} does not look like a PEFT adapter (no adapter_config.json).")
    return path.resolve()


def _resolve_adapter_path(spec: ModelSpec) -> Path | None:
    if spec.adapter_path_env:
        local = local_adapter_from_env(spec.adapter_path_env)
        if local is not None:
            return local
    if spec.adapter:
        return _download(spec.adapter)
    return None


def _read_model_rows(path: Path, *, required: bool) -> list[dict[str, object]]:
    if not path.exists():
        if required:
            raise ModelError(f"{path.name} is missing")
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("models")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ModelError(f"{path.name} models must be a list")
    out: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ModelError(f"{path.name} rows must be mappings")
        out.append(row)
    return out


def _row_to_spec(row: dict[str, object], *, user_added: bool) -> ModelSpec:
    model_id = str(row.get("id") or "").strip()
    label = str(row.get("label") or "").strip()
    base = str(row.get("base") or "").strip()
    description = " ".join(str(row.get("description") or "").split())
    if not model_id or not label or not base:
        raise ModelError("each model needs id, label, and base")
    _require_repo_id(base, what=f"{model_id} base")
    adapter, configured, detail = _adapter(row, model_id)
    family = str(row.get("family") or "chitrapathak").strip()
    if family not in _FAMILIES:
        raise ModelError(f"{model_id} family must be chitrapathak, indicocr, vlm, or trocr")
    subfolder = str(row.get("subfolder") or "").strip() or None
    if subfolder and (".." in subfolder or subfolder.startswith(("/", "."))):
        raise ModelError(f"{model_id} subfolder must stay inside the repo")
    adapter_path_env = str(row.get("adapter_path_env") or "").strip() or None
    return ModelSpec(
        id=model_id,
        label=label,
        description=description,
        base=base,
        adapter=adapter,
        configured=configured,
        family=family,
        subfolder=subfolder,
        adapter_path_env=adapter_path_env,
        detail=detail,
        user_added=user_added,
    )


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
        raise ModelError(f"{what} must be a Hugging Face repo id or link, not a local path")
    if value.count("/") != 1 or any(part == "" for part in value.split("/")):
        raise ModelError(f"{what} must look like org/name, got {value!r}")


def _load_user_payload() -> dict[str, object]:
    if not _USER_REGISTRY_PATH.exists():
        return {"models": []}
    payload = yaml.safe_load(_USER_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ModelError("user_models.yaml is invalid")
    payload.setdefault("models", [])
    return payload


def _write_user_payload(payload: dict[str, object]) -> None:
    models = payload.get("models") or []
    body = (
        "# Models added from the Indic Vachak page.\n"
        "# Built-in models stay in models.yaml.\n"
        + yaml.safe_dump({"models": models}, sort_keys=False, allow_unicode=True)
    )
    _USER_REGISTRY_PATH.write_text(body, encoding="utf-8")


def _spec_cached(spec: ModelSpec) -> bool:
    if not _repo_cached(spec.base, subfolder=spec.subfolder):
        return False
    if spec.adapter_path_env:
        try:
            if local_adapter_from_env(spec.adapter_path_env) is not None:
                return True
        except ModelError:
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
