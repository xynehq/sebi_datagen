"""Model discovery, PEFT resolution, and Bengali dataset guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pipeline.bengali_data import BengaliDataError, ensure_source_layout, prepare
from pipeline.cli import main
from pipeline.discovery import (
    DiscoveryError,
    discover,
    inspect_path,
    record_peft_evaluation,
    register_model,
    select_from_catalog,
    unregister_model,
)


def _png(path: Path, text: str = "বাংলা") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)
    del text


def _adapter(path: Path, base: Path, *, weights: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base), "peft_type": "LORA"}),
        encoding="utf-8",
    )
    if weights:
        (path / "adapter_model.safetensors").write_bytes(b"not-a-real-weight")


def _base(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "test"}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"base")


def test_discover_adapter_in_search_root_and_hf_cache(tmp_path: Path, monkeypatch) -> None:
    adapter = tmp_path / "extra" / "guj_v1"
    base = tmp_path / "bases" / "chitrapathak2"
    _base(base)
    _adapter(adapter, base)
    lone = tmp_path / "extra" / "loose"
    lone.mkdir()
    (lone / "model.safetensors").write_bytes(b"no-config")
    cache = tmp_path / "hf" / "hub" / "models--org--demo" / "snapshots" / "rev1"
    _base(cache)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    found = discover([tmp_path / "extra"])
    paths = {item.path for item in found}
    assert str(adapter.resolve()) in paths
    assert str(cache.resolve()) in paths
    assert str(lone.resolve()) not in paths
    parsed = inspect_path(adapter)
    assert parsed is not None
    assert parsed.family == "peft"
    assert parsed.role == "adapter"
    assert parsed.base_model == str(base)
    assert parsed.languages == "unknown"


def test_explicit_path_is_not_replaced_and_noninteractive_fails(tmp_path: Path) -> None:
    first = tmp_path / "one.ckpt"
    second = tmp_path / "two.ckpt"
    first.write_bytes(b"ckpt")
    second.write_bytes(b"ckpt")
    catalog = [inspect_path(first)]
    assert catalog[0] is not None
    chosen = select_from_catalog(catalog, explicit=second)
    assert chosen.path == str(second.resolve())
    with pytest.raises(DiscoveryError):
        select_from_catalog(catalog, interactive=False)
    picked = select_from_catalog(catalog, choice="1")
    assert picked.name == "one.ckpt"


def test_registry_add_and_remove_keeps_files(tmp_path: Path) -> None:
    base = tmp_path / "models" / "cache" / "demo"
    _base(base)
    register_model(tmp_path, base, name="demo")
    roots = [tmp_path / "models" / "adapters", tmp_path / "models" / "cache"]
    found = discover(roots, include_hf_cache=False)
    assert any(item.path == str(base.resolve()) for item in found)
    unregister_model(tmp_path, base)
    assert base.is_dir()
    saved = json.loads((tmp_path / "models" / "registry.json").read_text(encoding="utf-8"))
    assert saved == []


def test_missing_base_is_reported_and_present_base_is_recorded(tmp_path: Path) -> None:
    missing = tmp_path / "adapters" / "guj_v1"
    _adapter(missing, tmp_path / "does-not-exist")
    model = inspect_path(missing)
    assert model is not None
    with pytest.raises(DiscoveryError, match="base model"):
        record_peft_evaluation(model, tmp_path / "out.json")
    base = tmp_path / "base"
    _base(base)
    ready = tmp_path / "adapters" / "ready"
    _adapter(ready, base)
    recorded = record_peft_evaluation(inspect_path(ready), tmp_path / "reports" / "model.json")
    assert recorded["parseq"] is False
    assert recorded["weights_modified"] is False
    assert recorded["base_source"] == "adapter"
    assert Path(str(recorded["adapter"])).is_dir()


def test_relative_adapter_base_uses_the_configured_directory(tmp_path: Path) -> None:
    adapter = tmp_path / "lora"
    _adapter(adapter, Path("checkpoints/data/chitrapathak2"))
    official = tmp_path / "chitrapathak2"
    _base(official)
    recorded = record_peft_evaluation(
        inspect_path(adapter),
        tmp_path / "model.json",
        base_override=official,
    )
    assert recorded["base_source"] == "configured"
    assert Path(str(recorded["base_model"])) == official.resolve()


def test_bengali_missing_source_creates_the_folder(tmp_path: Path) -> None:
    source = tmp_path / "data" / "bengali" / "source"
    with pytest.raises(BengaliDataError, match="ready"):
        prepare(tmp_path)
    assert (source / "images").is_dir()
    assert (source / "labels.jsonl").is_file()
    report = (tmp_path / "reports" / "bengali" / "dataset_audit.txt").read_text(encoding="utf-8")
    assert "eligible records: 0" in report
    assert "source folder created: True" in report
    assert not (tmp_path / "data" / "bengali" / "real10k" / "train.jsonl").exists()
    labels = source / "labels.jsonl"
    labels.write_text('{"image_filename": "keep.png", "expected_text": "বাংলা"}\n', encoding="utf-8")
    assert ensure_source_layout(source) is False
    assert labels.read_text(encoding="utf-8").startswith('{"image_filename": "keep.png"')


def test_bengali_rows_are_audited_and_isolated(tmp_path: Path) -> None:
    source = tmp_path / "data" / "bengali" / "source"
    good = source / "images" / "line.png"
    _png(good)
    bad = source / "images" / "bad.png"
    bad.write_bytes(b"not-an-image")
    frozen = tmp_path / "frozen" / "marathi100"
    frozen.mkdir(parents=True)
    (frozen / "validation.jsonl").write_text(
        json.dumps({"image_filename": "frozen.png", "expected_text": "x"}) + "\n",
        encoding="utf-8",
    )
    _png(source / "images" / "frozen.png")
    labels = [
        {"image_filename": "/images/line.png", "expected_text": "বাংলা ১"},
        {"image_filename": "/images/line.png", "expected_text": "বাংলা ১"},
        {"image_filename": "/images/missing.png", "expected_text": "বাংলা"},
        {"image_filename": "/images/bad.png", "expected_text": "বাংলা"},
        {"image_filename": "/images/empty.png", "expected_text": "  "},
        {"image_filename": "/images/frozen.png", "expected_text": "বাংলা"},
        "not-json",
    ]
    lines = []
    for row in labels:
        if isinstance(row, dict):
            lines.append(json.dumps(row, ensure_ascii=False))
        else:
            lines.append(row)
    (source / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BengaliDataError, match="not 2"):
        prepare(tmp_path, target=2, train_count=1, val_count=1)
    report = (tmp_path / "reports" / "bengali" / "dataset_audit.txt").read_text(encoding="utf-8")
    assert "duplicate image references: 1" in report
    assert "missing images: 1" in report
    assert "unreadable images: 1" in report
    assert "empty labels: 1" in report
    assert "invalid records: 1" in report
    assert "frozen rows excluded: 1" in report
    assert not (tmp_path / "data" / "gujarati").exists()


def test_bengali_split_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "data" / "bengali" / "source" / "images"
    rows = []
    for index in range(4):
        name = f"{index}.png"
        _png(source / name)
        rows.append({"image_filename": f"/images/{name}", "expected_text": f"বাংলা {index}"})
    (source.parent / "labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepare(tmp_path, target=4, train_count=3, val_count=1)
    train = (tmp_path / "data" / "bengali" / "real10k" / "train.jsonl").read_text(encoding="utf-8")
    val = (tmp_path / "data" / "bengali" / "real10k" / "val.jsonl").read_text(encoding="utf-8")
    assert "বাংলা" in train
    assert len(train.splitlines()) == 3
    assert len(val.splitlines()) == 1
    assert not set(train.splitlines()) & set(val.splitlines())


def test_cli_lists_models_and_requires_a_model(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    assert main(["--list-models", "--root", str(tmp_path), "--offline"]) == 0
    assert main(["--language", "gujarati", "--stage", "eval", "--root", str(tmp_path)]) == 2
    adapter = tmp_path / "guj_v1"
    _adapter(adapter, tmp_path / "missing-base")
    assert (
        main(
            [
                "--language",
                "gujarati",
                "--stage",
                "eval",
                "--model-path",
                str(adapter),
                "--root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert main(["--language", "bengali", "--stage", "dataset", "--root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "No complete models" in captured.out
    assert "base model" in captured.err
