"""Page-added models are stored in user_models.yaml and merged into the list."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

import registry  # noqa: E402


@pytest.fixture()
def isolated_user_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import model_store  # noqa: E402

    db = tmp_path / "registry.sqlite"
    monkeypatch.setattr(model_store, "registry_db_path", lambda: db)
    monkeypatch.setattr(model_store, "_INITIALIZED", False)
    builtin = tmp_path / "models.yaml"
    user = tmp_path / "user_models.yaml"
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
    monkeypatch.setattr(registry, "_USER_REGISTRY_PATH", user)
    return db


def test_normalize_hf_link(isolated_user_registry: Path) -> None:
    del isolated_user_registry
    assert (
        registry.normalize_hf_ref(
            "https://huggingface.co/snxtyle/chitrapathak_marguj_mixed",
            what="adapter",
        )
        == "snxtyle/chitrapathak_marguj_mixed"
    )
    assert (
        registry.normalize_hf_ref("snxtyle/chitrapathak_marguj_mixed", what="adapter")
        == "snxtyle/chitrapathak_marguj_mixed"
    )


def test_add_and_remove_user_model(isolated_user_registry: Path) -> None:
    db = isolated_user_registry
    spec = registry.add_user_model(
        label="MarGuj Mixed",
        base="https://huggingface.co/krutrim-ai-labs/Chitrapathak-2",
        adapter="https://huggingface.co/snxtyle/chitrapathak_marguj_mixed",
        family="chitrapathak",
        description="shared LoRA",
    )
    assert spec.id == "marguj-mixed"
    assert spec.adapter == "snxtyle/chitrapathak_marguj_mixed"
    assert db.exists()
    ids = [item.id for item in registry.load_specs()]
    assert ids == ["chitrapathak-2", "marguj-mixed"]
    registry.remove_user_model("marguj-mixed")
    assert [item.id for item in registry.load_specs()] == ["chitrapathak-2"]


def test_cannot_remove_builtin(isolated_user_registry: Path) -> None:
    del isolated_user_registry
    with pytest.raises(registry.ModelError, match="Built-in"):
        registry.remove_user_model("chitrapathak-2")


def test_indicocr_gets_subfolder(isolated_user_registry: Path) -> None:
    del isolated_user_registry
    spec = registry.add_user_model(
        label="IndicOCR Extra",
        base="bodhan-ai/indic-ocr",
        family="indicocr",
    )
    assert spec.family == "indicocr"
    assert spec.subfolder == "weights/ocr"
