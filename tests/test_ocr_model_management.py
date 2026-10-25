"""Self-service OCR model registration (HF validate, SQLite registry, auth)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

import auth  # noqa: E402
import hf_inspect  # noqa: E402
import model_store  # noqa: E402
import registry  # noqa: E402
from hf_inspect import RepoValidation  # noqa: E402


@pytest.fixture()
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "registry.sqlite"
    monkeypatch.setattr(model_store, "registry_db_path", lambda: db)
    monkeypatch.setattr(model_store, "_INITIALIZED", False)
    builtin = tmp_path / "models.yaml"
    builtin.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "chitrapathak-2",
                        "label": "Chitrapathak 2",
                        "base": "krutrim-ai-labs/Chitrapathak-2",
                        "description": "builtin",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "_REGISTRY_PATH", builtin)
    monkeypatch.setattr(registry, "_USER_REGISTRY_PATH", tmp_path / "user_models.yaml")
    monkeypatch.setenv("OCR_API_KEY", "test-secret")
    return db


def test_builtin_models_still_listed(isolated_registry: Path) -> None:
    del isolated_registry
    ids = [spec.id for spec in registry.load_specs()]
    assert ids[0] == "chitrapathak-2"


def test_reject_non_hf_url() -> None:
    with pytest.raises(registry.ModelError, match="huggingface"):
        hf_inspect.parse_hf_url("https://example.com/evil/repo")


def test_detect_adapter_validation_shape() -> None:
    result = RepoValidation(
        repo_id="snxtyle/demo-adapter",
        revision="abc123",
        model_type="adapter",
        family="chitrapathak",
        base_repo_id="krutrim-ai-labs/Chitrapathak-2",
        architecture="peft_lora",
        compatible=True,
        message="ok",
    )
    assert result.model_type == "adapter"
    assert result.base_repo_id.endswith("Chitrapathak-2")


@patch("hf_inspect.validate_repository")
def test_validate_requires_admin(mock_validate: MagicMock, isolated_registry: Path) -> None:
    del isolated_registry
    from model_management import validate_hf_url

    mock_validate.return_value = RepoValidation(
        repo_id="org/full",
        revision=None,
        model_type="full",
        family="chitrapathak",
        architecture="Qwen2_5_VLForConditionalGeneration",
        compatible=True,
        message="ok",
    )
    with pytest.raises(registry.ModelError, match="admin"):
        validate_hf_url("https://huggingface.co/org/full")
    data = validate_hf_url(
        "https://huggingface.co/org/full",
        admin_key="test-secret",
    )
    assert data["compatible"] is True
    mock_validate.assert_called_once()


@patch("model_management.start_register_job")
@patch("hf_inspect.validate_repository")
def test_register_starts_job(
    mock_validate: MagicMock,
    mock_job: MagicMock,
    isolated_registry: Path,
) -> None:
    del isolated_registry
    from model_management import register_from_validation

    mock_validate.return_value = RepoValidation(
        repo_id="snxtyle/adapt",
        revision="rev1",
        model_type="adapter",
        family="chitrapathak",
        base_repo_id="krutrim-ai-labs/Chitrapathak-2",
        compatible=True,
        message="ok",
    )
    mock_job.return_value = ("job-1", "adapt")
    payload = register_from_validation(
        url="https://huggingface.co/snxtyle/adapt",
        label="My Adapter",
        admin_key="test-secret",
    )
    assert payload["job_id"] == "job-1"
    mock_job.assert_called_once()


def test_failed_model_not_in_selector(isolated_registry: Path) -> None:
    from model_store import CustomModelRow, init_store, insert_custom_model, _now

    init_store()
    now = _now()
    insert_custom_model(
        CustomModelRow(
            id="broken",
            label="Broken",
            description="",
            family="chitrapathak",
            model_type="full",
            repo_url="https://huggingface.co/org/broken",
            repo_id="org/broken",
            revision=None,
            base_repo_id=None,
            base_revision=None,
            adapter_repo_id=None,
            languages="unknown",
            status="failed",
            enabled=False,
            is_default=False,
            failure_reason="download error",
            created_at=now,
            updated_at=now,
        ),
    )
    ids = {spec.id for spec in registry.load_specs()}
    assert "broken" not in ids


def test_duplicate_id_rejected(isolated_registry: Path) -> None:
    del isolated_registry
    registry.add_user_model(label="Dup Model", base="krutrim-ai-labs/Chitrapathak-2")
    with pytest.raises(registry.ModelError, match="already"):
        registry.add_user_model(
            label="Dup Model",
            base="krutrim-ai-labs/Chitrapathak-2",
            model_id="dup-model",
        )


def test_persistence_after_reinit(isolated_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry.add_user_model(label="Persist Me", base="krutrim-ai-labs/Chitrapathak-2")
    monkeypatch.setattr(model_store, "_INITIALIZED", False)
    ids = [spec.id for spec in registry.load_specs()]
    assert "persist-me" in ids


def test_api_validate_unauthorized(isolated_registry: Path) -> None:
    del isolated_registry
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    response = client.post(
        "/v1/models/validate",
        json={"url": "https://huggingface.co/org/model"},
    )
    assert response.status_code == 403


def test_api_validate_no_secrets_in_body(isolated_registry: Path) -> None:
    del isolated_registry
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    with patch("model_management.validate_hf_url", return_value={"compatible": True}):
        response = client.post(
            "/v1/models/validate",
            headers={"Authorization": "Bearer test-secret"},
            json={"url": "https://huggingface.co/org/model"},
        )
    assert response.status_code == 200
    assert "test-secret" not in response.text


def test_unsupported_architecture_message() -> None:
    result = RepoValidation(
        repo_id="org/txt",
        revision=None,
        model_type="full",
        family="unknown",
        architecture="bert",
        compatible=False,
        message="Unsupported",
    )
    ok, msg = hf_inspect._compatibility_verdict(result, "")
    assert not ok
    assert not ok
    assert "text-only" in msg.lower() or "unsupported" in msg.lower()


def test_model_admin_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    monkeypatch.delenv("OCR_MODEL_ADMIN_KEY", raising=False)
    assert not auth.model_admin_required()
    with pytest.raises(registry.ModelError, match="turned off"):
        auth.check_model_admin(admin_key="anything")
