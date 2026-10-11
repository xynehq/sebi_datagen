"""JSONL manifests are resumable and do not silently overwrite."""

from __future__ import annotations

from pathlib import Path

import pytest

from indic_ocr_pipeline.errors import ImmutableFieldError, ManifestConflictError
from indic_ocr_pipeline.manifests import JsonlManifest

SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
OTHER = "a" * 64


def test_append_resume_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    manifest = JsonlManifest(path, "document_id")
    record = {
        "document_id": "MHGR_000001",
        "sha256": SHA,
        "status": "pending",
        "original_filename": "आदेश.pdf",
    }
    assert manifest.append(record) == "created"
    assert manifest.append(dict(record)) == "skipped"
    stored = path.read_text(encoding="utf-8")
    assert "आदेश.pdf" in stored
    assert "\\u" not in stored
    with pytest.raises(ManifestConflictError):
        manifest.append({**record, "sha256": OTHER})
    assert path.read_text(encoding="utf-8") == stored
    assert manifest.get("MHGR_000001") == record


def test_status_transitions_and_frozen_complete_rows(tmp_path: Path) -> None:
    manifest = JsonlManifest(tmp_path / "pages.jsonl", "page_id")
    manifest.append({"page_id": "MHGR_000001_P0001", "status": "pending", "sha256": SHA})
    with pytest.raises(ImmutableFieldError):
        manifest.update("MHGR_000001_P0001", {"status": "complete"})
    assert manifest.update("MHGR_000001_P0001", {"status": "running"}) == "updated"
    assert manifest.update("MHGR_000001_P0001", {"status": "complete"}) == "updated"
    frozen = (tmp_path / "pages.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ImmutableFieldError):
        manifest.update("MHGR_000001_P0001", {"status": "failed"})
    assert (tmp_path / "pages.jsonl").read_text(encoding="utf-8") == frozen


def test_immutable_field_is_rejected_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    manifest = JsonlManifest(path, "document_id")
    manifest.append({"document_id": "MHGR_000001", "status": "running", "sha256": SHA})
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ImmutableFieldError):
        manifest.update("MHGR_000001", {"sha256": OTHER})
    assert path.read_text(encoding="utf-8") == before


def test_split_and_gold_text_can_be_filled_once(tmp_path: Path) -> None:
    manifest = JsonlManifest(tmp_path / "gold.jsonl", "sample_id")
    manifest.append({"sample_id": "SAMPLE_1", "gold_text": "एक", "split": None, "final_text": None})
    assert manifest.update("SAMPLE_1", {"split": "train"}) == "updated"
    with pytest.raises(ImmutableFieldError):
        manifest.update("SAMPLE_1", {"split": "val"})
    with pytest.raises(ImmutableFieldError):
        manifest.update("SAMPLE_1", {"gold_text": "दोन"})
    assert manifest.get("SAMPLE_1")["split"] == "train"
    assert manifest.get("SAMPLE_1")["gold_text"] == "एक"


def test_duplicate_ids_already_in_the_file_are_an_error(tmp_path: Path) -> None:
    path = tmp_path / "crops.jsonl"
    path.write_text(
        '{"crop_id":"same","status":"pending"}\n{"crop_id":"same","status":"pending"}\n',
        encoding="utf-8",
    )
    manifest = JsonlManifest(path, "crop_id")
    with pytest.raises(ManifestConflictError):
        manifest.load()


def test_malformed_json_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ManifestConflictError):
        JsonlManifest(path, "document_id").load()
