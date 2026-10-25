"""High-level model registration workflow for API and UI."""

from __future__ import annotations

from typing import Any

from auth import check_model_admin, check_ui_model_admin
from hf_inspect import RepoValidation, parse_hf_url, validate_repository
from model_jobs import get_job_status, start_register_job
from model_store import (
    custom_model_public_dict,
    delete_custom_model,
    get_custom_model,
    get_user_default_model_id,
    init_store,
    list_custom_models,
    set_enabled,
    set_user_default_model,
)
from registry import ModelError, load_specs, slugify_model_id


def validation_to_dict(result: RepoValidation) -> dict[str, Any]:
    return {
        "repo_id": result.repo_id,
        "revision": result.revision,
        "model_type": result.model_type,
        "family": result.family,
        "base_repo_id": result.base_repo_id,
        "architecture": result.architecture,
        "label_suggestion": result.label_suggestion,
        "languages": result.languages,
        "warnings": result.warnings,
        "size_bytes": result.size_bytes,
        "file_count": result.file_count,
        "compatible": result.compatible,
        "message": result.message,
        "subfolder": result.subfolder,
    }


def resolve_add_model_urls(base_url: str, adapter_url: str = "") -> tuple[str, str]:
    """Return (primary_hf_url, base_override_repo_id) for validation/register."""
    base = (base_url or "").strip()
    adapter = (adapter_url or "").strip()
    if not base:
        raise ModelError("Paste the Hugging Face base model link.")
    base_id, _ = parse_hf_url(base)
    if adapter:
        return adapter, base_id
    return base, ""


def validate_add_model_urls(
    base_url: str,
    adapter_url: str = "",
    *,
    family_hint: str = "",
    authorization: str | None = None,
    admin_key: str | None = None,
    _authorized: bool = False,
) -> dict[str, Any]:
    if not _authorized:
        if authorization:
            check_model_admin(authorization=authorization, admin_key=admin_key)
        else:
            check_ui_model_admin(admin_key=admin_key)
    primary, base_override = resolve_add_model_urls(base_url, adapter_url)
    result = validate_repository(
        primary,
        family_hint=family_hint,
        base_override=base_override,
    )
    data = validation_to_dict(result)
    data["register_url"] = primary
    data["register_base_override"] = base_override
    data["base_url"] = (base_url or "").strip()
    data["adapter_url"] = (adapter_url or "").strip()
    return data


def validate_hf_url(
    url: str,
    *,
    revision: str | None = None,
    family_hint: str = "",
    base_override: str = "",
    authorization: str | None = None,
    admin_key: str | None = None,
    _authorized: bool = False,
) -> dict[str, Any]:
    if not _authorized:
        if authorization:
            check_model_admin(authorization=authorization, admin_key=admin_key)
        else:
            check_ui_model_admin(admin_key=admin_key)
    result = validate_repository(
        url,
        revision=revision,
        family_hint=family_hint,
        base_override=base_override,
    )
    return validation_to_dict(result)


def register_from_validation(
    *,
    url: str,
    label: str,
    description: str = "",
    model_id: str = "",
    languages: str = "unknown",
    family: str = "",
    revision: str | None = None,
    base_override: str = "",
    authorization: str | None = None,
    admin_key: str | None = None,
    _authorized: bool = False,
) -> dict[str, Any]:
    if not _authorized:
        if authorization:
            check_model_admin(authorization=authorization, admin_key=admin_key)
        else:
            check_ui_model_admin(admin_key=admin_key)
    result = validate_repository(
        url,
        revision=revision,
        family_hint=family,
        base_override=base_override,
    )
    if not result.compatible:
        raise ModelError(result.message or "This repository is not supported.")
    display = " ".join((label or result.label_suggestion).split())
    if not display:
        raise ModelError("Give the model a display name.")
    chosen_family = (family or result.family or "vlm").strip().lower()
    if chosen_family not in ("chitrapathak", "indicocr", "vlm", "trocr"):
        chosen_family = "vlm"
    job_id, new_id = start_register_job(
        validation=result,
        label=display,
        description=description,
        model_id=model_id or slugify_model_id(display),
        languages=languages,
        family=chosen_family,
    )
    row = get_custom_model(new_id)
    return {
        "job_id": job_id,
        "model": custom_model_public_dict(row) if row else None,
        "message": "Download started. Check progress until the model is ready.",
    }


def list_custom_with_jobs() -> list[dict[str, Any]]:
    init_store()
    return [custom_model_public_dict(row) for row in list_custom_models()]


def remove_custom(
    model_id: str,
    *,
    active_model_id: str | None,
    authorization: str | None = None,
    admin_key: str | None = None,
) -> None:
    if authorization:
        check_model_admin(authorization=authorization, admin_key=admin_key)
    else:
        check_ui_model_admin(admin_key=admin_key)
    if active_model_id and active_model_id == model_id:
        raise ModelError("Switch to another model before removing this one.")
    delete_custom_model(model_id)


def set_default(
    model_id: str,
    *,
    authorization: str | None = None,
    admin_key: str | None = None,
) -> None:
    if authorization:
        check_model_admin(authorization=authorization, admin_key=admin_key)
    else:
        check_ui_model_admin(admin_key=admin_key)
    known = {spec.id for spec in load_specs()}
    if model_id not in known:
        raise ModelError(f"Unknown model {model_id!r}.")
    set_user_default_model(model_id)


def clear_default(
    *,
    authorization: str | None = None,
    admin_key: str | None = None,
) -> None:
    check_model_admin(authorization=authorization, admin_key=admin_key)
    set_user_default_model(None)


_BUILTIN_DEFAULT_MODEL = "chitrapathak-marguj-mixed"


def resolve_default_model_id() -> str:
    custom = get_user_default_model_id()
    if custom:
        ids = {spec.id for spec in load_specs()}
        if custom in ids:
            return custom
    ids = {spec.id for spec in load_specs()}
    if _BUILTIN_DEFAULT_MODEL in ids:
        return _BUILTIN_DEFAULT_MODEL
    return next(iter(ids))
