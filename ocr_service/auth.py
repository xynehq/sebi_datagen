"""Authorization helpers for model-management actions."""

from __future__ import annotations

import os

from registry import ModelError


def model_admin_required() -> bool:
    """True when an admin secret is configured on the server."""
    return bool(
        os.environ.get("OCR_MODEL_ADMIN_KEY", "").strip()
        or os.environ.get("OCR_API_KEY", "").strip()
    )


def expected_model_admin_secret() -> str:
    return (
        os.environ.get("OCR_MODEL_ADMIN_KEY", "").strip()
        or os.environ.get("OCR_API_KEY", "").strip()
    )


def check_model_admin(
    *,
    authorization: str | None = None,
    admin_key: str | None = None,
) -> None:
    """Raise ModelError(403) for HTTP API model-management calls."""
    secret = expected_model_admin_secret()
    if not secret:
        raise ModelError(
            "Adding models is turned off on this server. "
            "An administrator must set OCR_API_KEY or OCR_MODEL_ADMIN_KEY.",
            status_code=403,
        )
    token = (admin_key or "").strip()
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    if token != secret:
        raise ModelError("Invalid or missing admin key.", status_code=403)


def check_ui_model_admin(admin_key: str | None = None) -> None:
    """Gradio UI runs on the server; allow adds when no admin secret is configured."""
    if not model_admin_required():
        return
    check_model_admin(admin_key=admin_key)
