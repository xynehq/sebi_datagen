"""Local PEFT adapter paths via adapter_path_env."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

import registry  # noqa: E402


def test_local_adapter_env_takes_precedence(tmp_path: Path, monkeypatch) -> None:
    adapter = tmp_path / "mixed"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("TEST_MIXED_ADAPTER", str(adapter))
    row = {
        "id": "test-mixed",
        "label": "Test Mixed",
        "base": "org/base",
        "adapter": "org/hf-adapter",
        "adapter_path_env": "TEST_MIXED_ADAPTER",
    }
    spec = registry._row_to_spec(row, user_added=False)
    resolved = registry._resolve_adapter_path(spec)
    assert resolved == adapter.resolve()


def test_local_adapter_env_missing_falls_back_to_hf_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_ADAPTER", raising=False)
    row = {
        "id": "test-mixed",
        "label": "Test Mixed",
        "base": "org/base",
        "adapter": "org/hf-adapter",
        "adapter_path_env": "MISSING_ADAPTER",
    }
    spec = registry._row_to_spec(row, user_added=False)
    monkeypatch.setattr(registry, "_download", lambda repo_id: tmp_path / repo_id.replace("/", "_"))
    resolved = registry._resolve_adapter_path(spec)
    assert resolved == tmp_path / "org_hf-adapter"
