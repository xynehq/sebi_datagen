"""Document ingestion records source PDFs and does not modify them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from indic_ocr_pipeline.config import _build_settings, load_settings
from indic_ocr_pipeline.errors import ConfigError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.pdf.ingest import (
    commit_document,
    discover_pdfs,
    ingest_documents,
)
from indic_ocr_pipeline.schemas import DocumentRecord, StageStatus
from indic_ocr_pipeline.utils.hashing import sha256_bytes, sha256_file

SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _manifest(tmp_path: Path) -> JsonlManifest:
    return JsonlManifest(tmp_path / "manifests" / "documents.jsonl", "document_id")


def _write_pdf(directory: Path, name: str, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def _pdf_bytes(marker: bytes) -> bytes:
    return b"%PDF-1.4\n" + marker + b"\n%%EOF\n"


def test_single_pdf_is_ingested_with_the_file_hash(tmp_path: Path) -> None:
    documents = tmp_path / "data" / "documents"
    payload = _pdf_bytes(b"one")
    path = _write_pdf(documents, "order.pdf", payload)
    before = path.read_bytes()
    manifest = _manifest(tmp_path)
    report = ingest_documents(
        documents,
        manifest,
        repo_root=tmp_path,
        clock=lambda: "2026-10-07T10:00:00+00:00",
    )
    assert report.discovered == 1
    assert report.ingested == 1
    assert report.skipped == 0
    assert report.errors == 0
    record = DocumentRecord.from_dict(manifest.load()[0])
    assert record.document_id == "MHGR_000001"
    assert record.sha256 == sha256_bytes(payload)
    assert record.file_size == len(payload)
    assert record.original_filename == "order.pdf"
    assert record.source_path == "data/documents/order.pdf"
    assert record.status is StageStatus.COMPLETE
    assert record.department is None
    assert record.document_type is None
    assert record.document_date is None
    assert record.page_count is None
    assert path.read_bytes() == before


def test_three_pdfs_receive_stable_ids_and_a_second_run_skips(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    payloads = {
        "file_a.pdf": _pdf_bytes(b"a"),
        "file_b.pdf": _pdf_bytes(b"b"),
        "file_c.pdf": _pdf_bytes(b"c"),
    }
    for name, payload in payloads.items():
        _write_pdf(documents, name, payload)
    manifest = _manifest(tmp_path)
    first = ingest_documents(documents, manifest, clock=lambda: "2026-10-07T10:00:00+00:00")
    assert [row.document_id for row in first.rows] == [
        "MHGR_000001",
        "MHGR_000002",
        "MHGR_000003",
    ]
    stored = manifest.path.read_text(encoding="utf-8")
    assert [json.loads(line)["sha256"] for line in stored.splitlines()] == [
        sha256_bytes(payloads["file_a.pdf"]),
        sha256_bytes(payloads["file_b.pdf"]),
        sha256_bytes(payloads["file_c.pdf"]),
    ]
    second = ingest_documents(documents, manifest, clock=lambda: "2026-10-07T11:00:00+00:00")
    assert second.ingested == 0
    assert second.skipped == 3
    assert second.errors == 0
    assert manifest.path.read_text(encoding="utf-8") == stored
    assert len(manifest.load()) == 3


def test_renamed_pdf_keeps_the_same_document_id(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    payload = _pdf_bytes(b"same")
    original = _write_pdf(documents, "first.pdf", payload)
    manifest = _manifest(tmp_path)
    ingest_documents(documents, manifest, clock=lambda: "2026-10-07T10:00:00+00:00")
    original.rename(documents / "renamed.pdf")
    report = ingest_documents(documents, manifest, clock=lambda: "2026-10-07T11:00:00+00:00")
    assert report.skipped == 1
    assert report.rows[0].document_id == "MHGR_000001"
    records = manifest.load()
    assert len(records) == 1
    assert records[0]["sha256"] == sha256_bytes(payload)
    assert records[0]["original_filename"] == "first.pdf"


def test_changed_bytes_receive_a_new_document_id(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    path = _write_pdf(documents, "order.pdf", _pdf_bytes(b"before"))
    manifest = _manifest(tmp_path)
    ingest_documents(documents, manifest, clock=lambda: "2026-10-07T10:00:00+00:00")
    path.write_bytes(_pdf_bytes(b"after"))
    report = ingest_documents(documents, manifest, clock=lambda: "2026-10-07T11:00:00+00:00")
    assert report.ingested == 1
    assert report.rows[0].document_id == "MHGR_000002"
    records = manifest.load()
    assert len(records) == 2
    assert records[0]["sha256"] != records[1]["sha256"]
    assert records[1]["original_filename"] == "order.pdf"


def test_identical_payload_is_skipped_and_a_conflict_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = DocumentRecord(
        document_id="MHGR_000001",
        original_filename="order.pdf",
        source_path="data/documents/order.pdf",
        sha256=SHA,
        file_size=3,
        status=StageStatus.COMPLETE,
        ingested_at="2026-10-07T10:00:00+00:00",
    )
    assert commit_document(manifest, record) == "created"
    assert commit_document(manifest, record) == "skipped"
    before = manifest.path.read_text(encoding="utf-8")
    changed = DocumentRecord(
        document_id="MHGR_000001",
        original_filename="other.pdf",
        source_path="data/documents/other.pdf",
        sha256=SHA,
        file_size=3,
        status=StageStatus.COMPLETE,
        ingested_at="2026-10-07T10:00:00+00:00",
    )
    with pytest.raises(ManifestConflictError, match="can resume"):
        commit_document(manifest, changed)
    assert manifest.path.read_text(encoding="utf-8") == before


def test_missing_directory_and_non_pdf_files(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="does not exist"):
        discover_pdfs(tmp_path / "missing")
    documents = tmp_path / "documents"
    documents.mkdir()
    _write_pdf(documents, "NOTES.PDF", _pdf_bytes(b"upper"))
    (documents / "readme.txt").write_text("ignore", encoding="utf-8")
    (documents / ".gitkeep").write_text("\n", encoding="utf-8")
    nested = documents / "nested"
    nested.mkdir()
    _write_pdf(nested, "hidden.pdf", _pdf_bytes(b"nested"))
    found = discover_pdfs(documents)
    assert [path.name for path in found] == ["NOTES.PDF"]
    report = ingest_documents(documents, _manifest(tmp_path))
    assert report.discovered == 1
    assert report.ingested == 1
    assert report.rows[0].filename == "NOTES.PDF"


def test_hashing_reads_the_file_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"0123456789" * 40
    path = _write_pdf(tmp_path, "large.pdf", payload)
    read_sizes: list[int] = []
    real_open = Path.open

    def tracking_open(self: Path, *args: object, **kwargs: object) -> object:
        handle = real_open(self, *args, **kwargs)  # type: ignore[arg-type]
        original_read = handle.read

        def read(size: int = -1) -> bytes:
            chunk = original_read(size)
            if chunk:
                read_sizes.append(len(chunk))
            return chunk

        handle.read = read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    assert sha256_file(path, chunk_size=64) == sha256_bytes(payload)
    assert read_sizes
    assert max(read_sizes) <= 64


def test_manifest_stays_valid_jsonl(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    _write_pdf(documents, "a.pdf", _pdf_bytes(b"a"))
    _write_pdf(documents, "b.pdf", _pdf_bytes(b"b"))
    manifest = _manifest(tmp_path)
    ingest_documents(documents, manifest, clock=lambda: "2026-10-07T10:00:00+00:00")
    ingest_documents(documents, manifest, clock=lambda: "2026-10-07T11:00:00+00:00")
    lines = manifest.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["document_id"] == "MHGR_000001"
    assert "gold_text" not in parsed[0]


def test_malformed_document_record_is_not_written(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = DocumentRecord(
        document_id="MHGR_000001",
        original_filename="order.pdf",
        source_path="data/documents/order.pdf",
        sha256=SHA,
        file_size=3,
        status=StageStatus.COMPLETE,
        ingested_at="2026-10-07T10:00:00+00:00",
    )
    record.sha256 = "not-a-hash"
    with pytest.raises(SchemaError):
        commit_document(manifest, record)
    assert not manifest.path.exists()


def test_network_configuration_remains_rejected(repo_root: Path) -> None:
    settings = load_settings(repo_root, environ={})
    assert settings.safety.allow_network is False
    config_dir = repo_root / "configs"

    def load(name: str) -> dict[str, object]:
        with (config_dir / name).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        assert isinstance(loaded, dict)
        return loaded

    pipeline = load("pipeline.yaml")
    assert isinstance(pipeline["safety"], dict)
    pipeline["safety"]["allow_network"] = True
    with pytest.raises(ConfigError):
        _build_settings(
            root=repo_root,
            pipeline=pipeline,
            rendering=load("rendering.yaml"),
            detection=load("detection.yaml"),
            cropping=load("cropping.yaml"),
            ocr=load("ocr.yaml"),
            scoring=load("scoring.yaml"),
            review=load("review.yaml"),
            dataset=load("dataset.yaml"),
        )
