"""Background download and validation jobs for custom models."""

from __future__ import annotations

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from hf_inspect import RepoValidation, validate_repository
from model_store import (
    _FAILED,
    _READY,
    create_job,
    custom_model_to_spec,
    get_custom_model,
    insert_custom_model,
    update_custom_status,
    update_job,
)
from registry import ModelError, ensure_weights

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-model-job")
_ACTIVE_LOCK = threading.Lock()


def start_register_job(
    *,
    validation: RepoValidation,
    label: str,
    description: str,
    model_id: str,
    languages: str,
    family: str,
) -> str:
    """Create DB row (disabled) and enqueue download + smoke test."""
    from model_store import CustomModelRow, _now, reserve_model_id

    chosen_id = reserve_model_id(label, model_id)
    now = _now()
    if validation.model_type == "adapter":
        base = validation.base_repo_id
        adapter = validation.repo_id
        repo_url = f"https://huggingface.co/{validation.repo_id}"
    else:
        base = None
        adapter = None
        repo_url = f"https://huggingface.co/{validation.repo_id}"

    row = CustomModelRow(
        id=chosen_id,
        label=label.strip(),
        description=(description or "").strip()
        or f"Added from Hugging Face ({validation.repo_id}).",
        family=family or validation.family,
        model_type=validation.model_type,
        repo_url=repo_url,
        repo_id=validation.repo_id,
        revision=validation.revision,
        base_repo_id=base,
        base_revision=validation.base_revision,
        adapter_repo_id=adapter,
        languages=languages or validation.languages,
        subfolder=validation.subfolder,
        status="waiting",
        enabled=False,
        is_default=False,
        failure_reason=None,
        created_at=now,
        updated_at=now,
    )
    insert_custom_model(row, allow_existing=False)
    job_id = create_job(chosen_id, message="Waiting to download.")
    _EXECUTOR.submit(_run_job, job_id, chosen_id)
    return job_id, chosen_id


def _run_job(job_id: str, model_id: str) -> None:
    row = get_custom_model(model_id)
    if row is None:
        update_job(job_id, status=_FAILED, message="Model record missing.")
        return
    try:
        update_job(job_id, status="downloading", progress=0.1, message="Downloading model files.")
        update_custom_status(model_id, status="downloading", enabled=False)
        spec = custom_model_to_spec(row)
        ensure_weights(spec)
        update_job(job_id, status="validating", progress=0.7, message="Validating downloaded files.")
        update_custom_status(model_id, status="validating")
        update_job(job_id, status="testing", progress=0.85, message="Running model test.")
        update_custom_status(model_id, status="testing")
        smoke_test_spec(spec)
        update_custom_status(model_id, status=_READY, enabled=True, failure_reason="")
        update_job(job_id, status=_READY, progress=1.0, message="Ready to use.")
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        update_custom_status(model_id, status=_FAILED, enabled=False, failure_reason=reason)
        update_job(
            job_id,
            status=_FAILED,
            progress=1.0,
            message=f"Failed: {reason}",
        )


def smoke_test_spec(spec: Any) -> None:
    """Load weights in isolation without touching the shared Engine instance."""
    from engine import _load_model, _release, pick_device
    from registry import ensure_weights

    device = pick_device()
    base_path, adapter_path = ensure_weights(spec)
    model = None
    processor = None
    try:
        model, processor = _load_model(
            base_path,
            adapter_path,
            device=device,
            subfolder=spec.subfolder,
            family=spec.family,
        )
    finally:
        model = None
        processor = None
        gc.collect()
        _release(device)


def retry_failed_job(model_id: str) -> str:
    """Re-run download/smoke test for a failed custom model (weights may already be cached)."""
    row = get_custom_model(model_id)
    if row is None:
        raise ModelError(f"Unknown model {model_id!r}.")
    if row.status != _FAILED:
        raise ModelError("Only failed models can be retried.")
    validation = validate_repository(row.repo_url, revision=row.revision)
    if not validation.compatible:
        raise ModelError(validation.message or "This model is not compatible.")
    from model_store import update_custom_inference_meta

    update_custom_inference_meta(
        model_id,
        family=validation.family,
        subfolder=validation.subfolder,
    )
    update_custom_status(model_id, status="waiting", enabled=False, failure_reason="")
    job_id = create_job(model_id, message="Retrying after a fix.")
    _EXECUTOR.submit(_run_job, job_id, model_id)
    return job_id


def get_job_status(job_id: str) -> dict[str, Any]:
    from model_store import get_job

    job = get_job(job_id)
    if job is None:
        raise ModelError(f"Unknown job {job_id!r}.", status_code=404)
    row = get_custom_model(str(job["model_id"]))
    return {
        "job_id": job_id,
        "model_id": job["model_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "model_status": row.status if row else None,
        "enabled": row.enabled if row else False,
    }
