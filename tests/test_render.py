"""PDF pages render to PNG without modifying the source or running OCR."""

from __future__ import annotations

import json
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from indic_ocr_pipeline.ids import make_page_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.rendering.pages import (
    RenderConflictError,
    page_png_path,
    render_documents,
)
from indic_ocr_pipeline.schemas import DocumentRecord, StageStatus
from indic_ocr_pipeline.utils.hashing import sha256_bytes, sha256_file


def _documents(tmp_path: Path) -> JsonlManifest:
    return JsonlManifest(tmp_path / "documents.jsonl", "document_id")


def _pages(tmp_path: Path) -> JsonlManifest:
    return JsonlManifest(tmp_path / "pages.jsonl", "page_id")


def _write_pdf(path: Path, sizes: list[tuple[float, float]]) -> bytes:
    pdf = pdfium.PdfDocument.new()
    try:
        for width, height in sizes:
            pdf.new_page(width, height)
        pdf.save(path)
    finally:
        pdf.close()
    return path.read_bytes()


def _document_row(tmp_path: Path, name: str, sizes: list[tuple[float, float]]) -> dict[str, object]:
    source = tmp_path / "raw" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    payload = _write_pdf(source, sizes)
    record = DocumentRecord(
        document_id="MHGR_000001",
        original_filename=name,
        source_path=source_path(tmp_path, source),
        sha256=sha256_bytes(payload),
        file_size=len(payload),
        status=StageStatus.COMPLETE,
        ingested_at="2026-10-07T10:00:00+00:00",
    )
    return record.to_dict()


def source_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def test_one_page_pdf_renders_a_readable_png(tmp_path: Path) -> None:
    row = _document_row(tmp_path, "one.pdf", [(72, 72)])
    source = tmp_path / "raw" / "one.pdf"
    before = source.read_bytes()
    before_hash = sha256_file(source)
    pages = _pages(tmp_path)
    report = render_documents(
        [row],
        pages,
        tmp_path / "rendered",
        repo_root=tmp_path,
        dpi=200,
        clock=lambda: "2026-10-07T12:00:00+00:00",
    )
    assert report.pages_discovered == 1
    assert report.pages_rendered == 1
    assert report.errors == 0
    image = page_png_path(tmp_path / "rendered", "MHGR_000001", 1)
    assert image.is_file()
    with Image.open(image) as handle:
        handle.verify()
    with Image.open(image) as handle:
        assert handle.size == (200, 200)
        assert handle.format == "PNG"
    record = pages.load()[0]
    assert record["page_id"] == make_page_id("MHGR_000001", 1)
    assert record["page_number"] == 1
    assert record["width"] == 200
    assert record["height"] == 200
    assert record["sha256"] == sha256_file(image)
    assert record["dpi"] == 200
    assert "gold_text" not in record
    assert record["pdf_text_layer"] is None
    assert source.read_bytes() == before
    assert sha256_file(source) == before_hash


def test_multipage_pdf_renders_every_page_with_stable_ids(tmp_path: Path) -> None:
    row = _document_row(tmp_path, "many.pdf", [(72, 36), (36, 72), (72, 72)])
    pages = _pages(tmp_path)
    first = render_documents(
        [row],
        pages,
        tmp_path / "rendered",
        repo_root=tmp_path,
        dpi=200,
        clock=lambda: "2026-10-07T12:00:00+00:00",
    )
    assert first.pages_rendered == 3
    stored = pages.path.read_text(encoding="utf-8")
    ids = [json.loads(line)["page_id"] for line in stored.splitlines()]
    assert ids == [
        "MHGR_000001_P0001",
        "MHGR_000001_P0002",
        "MHGR_000001_P0003",
    ]
    second = render_documents(
        [row],
        pages,
        tmp_path / "rendered",
        repo_root=tmp_path,
        dpi=200,
        clock=lambda: "2026-10-07T13:00:00+00:00",
    )
    assert second.pages_rendered == 0
    assert second.pages_skipped == 3
    assert pages.path.read_text(encoding="utf-8") == stored
    sizes = []
    for number in (1, 2, 3):
        image = page_png_path(tmp_path / "rendered", "MHGR_000001", number)
        with Image.open(image) as handle:
            sizes.append(handle.size)
    assert sizes == [(200, 100), (100, 200), (200, 200)]


def test_same_payload_skips_and_different_payload_conflicts(tmp_path: Path) -> None:
    row = _document_row(tmp_path, "one.pdf", [(72, 72)])
    pages = _pages(tmp_path)
    rendered = tmp_path / "rendered"
    render_documents([row], pages, rendered, repo_root=tmp_path, dpi=200)
    image = page_png_path(rendered, "MHGR_000001", 1)
    original_png = image.read_bytes()
    original_manifest = pages.path.read_text(encoding="utf-8")
    again = render_documents([row], pages, rendered, repo_root=tmp_path, dpi=200)
    assert again.pages_skipped == 1
    assert pages.path.read_text(encoding="utf-8") == original_manifest

    corrupted = json.loads(original_manifest)
    corrupted["sha256"] = "a" * 64
    corrupted["width"] = 10
    pages.path.write_text(json.dumps(corrupted) + "\n", encoding="utf-8")
    conflict = render_documents([row], pages, rendered, repo_root=tmp_path, dpi=200)
    assert conflict.errors == 1
    assert "different payload" in conflict.details[0].message
    assert image.read_bytes() == original_png
    assert pages.path.read_text(encoding="utf-8") == json.dumps(corrupted) + "\n"


def test_missing_png_is_restored_when_the_render_matches(tmp_path: Path) -> None:
    row = _document_row(tmp_path, "one.pdf", [(72, 72)])
    pages = _pages(tmp_path)
    rendered = tmp_path / "rendered"
    render_documents(
        [row],
        pages,
        rendered,
        repo_root=tmp_path,
        dpi=200,
        clock=lambda: "2026-10-07T12:00:00+00:00",
    )
    image = page_png_path(rendered, "MHGR_000001", 1)
    original = image.read_bytes()
    manifest_text = pages.path.read_text(encoding="utf-8")
    image.unlink()
    report = render_documents(
        [row],
        pages,
        rendered,
        repo_root=tmp_path,
        dpi=200,
        clock=lambda: "2026-10-07T13:00:00+00:00",
    )
    assert report.pages_rendered == 1
    assert image.read_bytes() == original
    assert pages.path.read_text(encoding="utf-8") == manifest_text
    assert sha256_file(image) == json.loads(manifest_text)["sha256"]


def test_missing_and_corrupt_pdfs_do_not_write_page_rows(tmp_path: Path) -> None:
    missing = _document_row(tmp_path, "gone.pdf", [(72, 72)])
    (tmp_path / "raw" / "gone.pdf").unlink()
    pages = _pages(tmp_path)
    missing_report = render_documents(
        [missing], pages, tmp_path / "rendered", repo_root=tmp_path, dpi=200
    )
    assert missing_report.errors == 1
    assert "missing" in missing_report.details[0].message
    assert not pages.path.exists()

    corrupt_path = tmp_path / "raw" / "bad.pdf"
    corrupt_path.write_bytes(b"this is not a pdf")
    corrupt = dict(missing)
    corrupt["source_path"] = source_path(tmp_path, corrupt_path)
    corrupt["sha256"] = sha256_bytes(corrupt_path.read_bytes())
    corrupt["file_size"] = corrupt_path.stat().st_size
    corrupt_report = render_documents(
        [corrupt], pages, tmp_path / "rendered", repo_root=tmp_path, dpi=200
    )
    assert corrupt_report.errors == 1
    assert corrupt_report.pages_rendered == 0
    assert not pages.path.exists()


def test_failed_manifest_append_does_not_record_the_failed_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _document_row(tmp_path, "many.pdf", [(72, 72), (72, 36)])
    pages = _pages(tmp_path)
    original_append = pages.append

    def fail_second(record: dict[str, object]) -> str:
        if record["page_number"] == 2:
            raise RenderConflictError("simulated manifest failure")
        return original_append(record)

    monkeypatch.setattr(pages, "append", fail_second)
    report = render_documents(
        [row], pages, tmp_path / "rendered", repo_root=tmp_path, dpi=200
    )
    assert report.pages_rendered == 1
    assert report.errors == 1
    records = pages.load()
    assert [record["page_number"] for record in records] == [1]
    assert "gold_text" not in records[0]


def test_manifest_is_jsonl_and_rerun_is_idempotent(tmp_path: Path) -> None:
    row = _document_row(tmp_path, "one.pdf", [(72, 72)])
    pages = _pages(tmp_path)
    render_documents([row], pages, tmp_path / "rendered", repo_root=tmp_path, dpi=200)
    render_documents([row], pages, tmp_path / "rendered", repo_root=tmp_path, dpi=200)
    lines = pages.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["document_id"] == "MHGR_000001"
    assert parsed["page_id"] == "MHGR_000001_P0001"


def test_renderer_does_not_reference_ocr_or_network() -> None:
    source = Path(__file__).resolve().parents[1] / "src/indic_ocr_pipeline/rendering/pages.py"
    text = source.read_text(encoding="utf-8").lower()
    for banned in ("tesseract", "indicocr", "chitrapathak", "easyocr", "http://", "https://"):
        assert banned not in text
