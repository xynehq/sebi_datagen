"""Render ingested PDFs to one PNG per page.

Pages are rendered one at a time. The source PDF is only read.
No text layer is extracted and no OCR is run.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium

from indic_ocr_pipeline.errors import DatasetError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.ids import make_page_id
from indic_ocr_pipeline.manifests import JsonlManifest, open_manifest
from indic_ocr_pipeline.pdf.ingest import source_path_for
from indic_ocr_pipeline.schemas import (
    PDF_TEXT_LAYER_ROLE,
    DocumentRecord,
    PageRecord,
    StageStatus,
)
from indic_ocr_pipeline.utils.hashing import sha256_bytes, sha256_file
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PDF_POINTS_PER_INCH = 72
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RESUME_NOTE = "Existing manifest rows were kept. Rendering can resume."


class RenderError(DatasetError):
    """A page could not be rendered."""


class RenderConflictError(RenderError):
    """A page id already exists with a different payload or image hash."""


@dataclass
class DocumentRenderStats:
    """Rendering counters for one source document."""

    document_id: str
    pages: int = 0
    rendered: int = 0
    skipped: int = 0
    errors: int = 0
    message: str = ""


@dataclass
class RenderReport:
    """Counters for one rendering run."""

    documents: int = 0
    pages_discovered: int = 0
    pages_rendered: int = 0
    pages_skipped: int = 0
    errors: int = 0
    details: list[DocumentRenderStats] = field(default_factory=list)

    def add(self, stats: DocumentRenderStats) -> None:
        self.details.append(stats)
        self.documents += 1
        self.pages_discovered += stats.pages
        self.pages_rendered += stats.rendered
        self.pages_skipped += stats.skipped
        self.errors += stats.errors


def page_png_path(rendered_pages_dir: Path, document_id: str, page_number: int) -> Path:
    """Return the PNG path for a 1-based page number."""
    return rendered_pages_dir / document_id / f"page_{page_number:04d}.png"


def resolve_existing_path(stored: str, repo_root: Path | None) -> Path:
    """Resolve a manifest path against the repository root when it is relative."""
    path = Path(stored)
    if path.is_absolute() or repo_root is None:
        return path
    return repo_root / path


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def png_size(path: Path) -> tuple[int, int]:
    """Return width and height from a PNG header."""
    with path.open("rb") as handle:
        header = handle.read(24)
        handle.seek(-12, os.SEEK_END)
        tail = handle.read(12)
    if len(header) < 24 or header[:8] != PNG_SIGNATURE or b"IEND" not in tail:
        raise RenderError(f"{path} is not a complete PNG. {RESUME_NOTE}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width < 1 or height < 1:
        raise RenderError(f"{path} has invalid PNG dimensions. {RESUME_NOTE}")
    return width, height


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def rasterize_page(page: pdfium.PdfPage, dpi: int) -> tuple[bytes, int, int]:
    """Render one PDF page to PNG bytes, then release the bitmap."""
    scale = dpi / PDF_POINTS_PER_INCH
    bitmap = page.render(scale=scale, rotation=0, crop=(0, 0, 0, 0))
    image = None
    try:
        image = bitmap.to_pil()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        width, height = image.size
    finally:
        if image is not None:
            image.close()
        bitmap.close()
    if width < 1 or height < 1 or not payload.startswith(PNG_SIGNATURE):
        raise RenderError("renderer produced an empty page image")
    return payload, width, height


def commit_page(manifest: JsonlManifest, record: PageRecord) -> str:
    """Append a validated page row. A different payload for the same id conflicts."""
    payload = record.to_dict()
    try:
        return manifest.append(payload)
    except ManifestConflictError as exc:
        raise RenderConflictError(
            f"{exc}. page_id={record.page_id} document_id={record.document_id}. {RESUME_NOTE}"
        ) from exc


def render_documents(
    document_rows: list[Mapping[str, object]],
    manifest: JsonlManifest,
    rendered_pages_dir: Path,
    *,
    repo_root: Path | None,
    dpi: int,
    clock: Callable[[], str] | None = None,
) -> RenderReport:
    """Render every page of every ingested document.

    ``document_rows`` come from the document manifest. This function does not
    allocate document IDs and does not modify source PDFs.
    """
    report = RenderReport()
    existing_pages = {row["page_id"]: row for row in manifest.load() if "page_id" in row}
    timestamp = (clock or utc_now)()
    ordered = sorted(
        document_rows,
        key=lambda row: str(row.get("document_id", "")),
    )
    for raw in ordered:
        stats = _render_document(
            raw,
            manifest=manifest,
            rendered_pages_dir=rendered_pages_dir,
            repo_root=repo_root,
            dpi=dpi,
            timestamp=timestamp,
            existing_pages=existing_pages,
        )
        report.add(stats)
    return report


def _render_document(
    raw: Mapping[str, object],
    *,
    manifest: JsonlManifest,
    rendered_pages_dir: Path,
    repo_root: Path | None,
    dpi: int,
    timestamp: str,
    existing_pages: dict[str, dict[str, object]],
) -> DocumentRenderStats:
    try:
        document = DocumentRecord.from_dict(dict(raw))
    except (SchemaError, TypeError) as exc:
        document_id = str(raw.get("document_id") or "unknown")
        return DocumentRenderStats(document_id=document_id, errors=1, message=str(exc))
    stats = DocumentRenderStats(document_id=document.document_id)
    source = resolve_existing_path(document.source_path, repo_root)
    try:
        _require_unchanged_source(source, document)
        pdf = pdfium.PdfDocument(str(source))
    except (OSError, pdfium.PdfiumError, RenderError) as exc:
        stats.errors = 1
        stats.message = f"{exc} {RESUME_NOTE}"
        return stats
    try:
        stats.pages = len(pdf)
        if stats.pages < 1:
            raise RenderError(f"{document.document_id} has no pages. {RESUME_NOTE}")
        for page_number in range(1, stats.pages + 1):
            page_id = make_page_id(document.document_id, page_number)
            image_path = page_png_path(
                rendered_pages_dir, document.document_id, page_number
            )
            stored_path = source_path_for(image_path, repo_root)
            existing = existing_pages.get(page_id)
            if existing is not None and image_path.is_file():
                _require_matching_artifact(existing, image_path, stored_path, dpi)
                stats.skipped += 1
                continue
            page = pdf[page_number - 1]
            try:
                outcome = _render_one_page(
                    page,
                    document=document,
                    page_number=page_number,
                    manifest=manifest,
                    rendered_pages_dir=rendered_pages_dir,
                    repo_root=repo_root,
                    dpi=dpi,
                    timestamp=timestamp,
                    existing_pages=existing_pages,
                )
            finally:
                page.close()
            if outcome == "skipped":
                stats.skipped += 1
            else:
                stats.rendered += 1
    except (OSError, pdfium.PdfiumError, RenderError, SchemaError) as exc:
        stats.errors += 1
        stats.message = str(exc)
    finally:
        pdf.close()
    return stats


def _require_unchanged_source(source: Path, document: DocumentRecord) -> None:
    if not source.is_file():
        raise RenderError(
            f"source PDF is missing: {document.source_path} "
            f"document_id={document.document_id}. {RESUME_NOTE}"
        )
    digest = sha256_file(source)
    if digest != document.sha256:
        raise RenderError(
            f"source PDF hash changed for {document.document_id}: "
            f"manifest={document.sha256} file={digest}. {RESUME_NOTE}"
        )


def _render_one_page(
    page: pdfium.PdfPage,
    *,
    document: DocumentRecord,
    page_number: int,
    manifest: JsonlManifest,
    rendered_pages_dir: Path,
    repo_root: Path | None,
    dpi: int,
    timestamp: str,
    existing_pages: dict[str, dict[str, object]],
) -> str:
    page_id = make_page_id(document.document_id, page_number)
    image_path = page_png_path(rendered_pages_dir, document.document_id, page_number)
    stored_path = source_path_for(image_path, repo_root)
    existing = existing_pages.get(page_id)
    if existing is not None:
        return _resume_existing_page(
            page,
            existing=existing,
            image_path=image_path,
            stored_path=stored_path,
            dpi=dpi,
        )
    if image_path.is_file():
        return _adopt_existing_image(
            page,
            document=document,
            page_number=page_number,
            page_id=page_id,
            image_path=image_path,
            stored_path=stored_path,
            manifest=manifest,
            repo_root=repo_root,
            dpi=dpi,
            timestamp=timestamp,
            existing_pages=existing_pages,
        )
    payload, width, height = rasterize_page(page, dpi)
    write_bytes_atomic(image_path, payload)
    return _record_new_page(
        document=document,
        page_number=page_number,
        page_id=page_id,
        image_path=image_path,
        stored_path=stored_path,
        width=width,
        height=height,
        dpi=dpi,
        manifest=manifest,
        timestamp=timestamp,
        existing_pages=existing_pages,
    )


def _resume_existing_page(
    page: pdfium.PdfPage,
    *,
    existing: Mapping[str, object],
    image_path: Path,
    stored_path: str,
    dpi: int,
) -> str:
    page_id = str(existing.get("page_id"))
    if image_path.is_file():
        _require_matching_artifact(existing, image_path, stored_path, dpi)
        return "skipped"
    payload, width, height = rasterize_page(page, dpi)
    digest = _sha256_bytes(payload)
    if digest != existing.get("sha256") or width != existing.get("width"):
        raise RenderConflictError(
            f"re-rendered {page_id} does not match the manifest "
            f"and the PNG is missing. {RESUME_NOTE}"
        )
    if height != existing.get("height") or dpi != existing.get("dpi"):
        raise RenderConflictError(
            f"re-rendered {page_id} does not match the recorded page geometry. {RESUME_NOTE}"
        )
    write_bytes_atomic(image_path, payload)
    if sha256_file(image_path) != digest:
        raise RenderError(f"restored PNG hash mismatch for {page_id}. {RESUME_NOTE}")
    return "rendered"


def _adopt_existing_image(
    page: pdfium.PdfPage,
    *,
    document: DocumentRecord,
    page_number: int,
    page_id: str,
    image_path: Path,
    stored_path: str,
    manifest: JsonlManifest,
    repo_root: Path | None,
    dpi: int,
    timestamp: str,
    existing_pages: dict[str, dict[str, object]],
) -> str:
    del repo_root
    payload, width, height = rasterize_page(page, dpi)
    if _sha256_bytes(payload) != sha256_file(image_path):
        raise RenderConflictError(
            f"{page_id} image exists without a manifest row and does not match "
            f"a fresh render. {RESUME_NOTE}"
        )
    return _record_new_page(
        document=document,
        page_number=page_number,
        page_id=page_id,
        image_path=image_path,
        stored_path=stored_path,
        width=width,
        height=height,
        dpi=dpi,
        manifest=manifest,
        timestamp=timestamp,
        existing_pages=existing_pages,
    )


def _record_new_page(
    *,
    document: DocumentRecord,
    page_number: int,
    page_id: str,
    image_path: Path,
    stored_path: str,
    width: int,
    height: int,
    dpi: int,
    manifest: JsonlManifest,
    timestamp: str,
    existing_pages: dict[str, dict[str, object]],
) -> str:
    file_width, file_height = png_size(image_path)
    if (file_width, file_height) != (width, height) or image_path.stat().st_size < 1:
        raise RenderError(f"{page_id} PNG failed validation. {RESUME_NOTE}")
    digest = sha256_file(image_path)
    record = PageRecord(
        document_id=document.document_id,
        page_id=page_id,
        page_number=page_number,
        image_path=stored_path,
        width=width,
        height=height,
        dpi=dpi,
        sha256=digest,
        pdf_text_layer=None,
        pdf_text_layer_role=PDF_TEXT_LAYER_ROLE,
        status=StageStatus.COMPLETE,
        rendered_at=timestamp,
    )
    outcome = commit_page(manifest, record)
    existing_pages[page_id] = record.to_dict()
    return "skipped" if outcome == "skipped" else "rendered"


def _require_matching_artifact(
    existing: Mapping[str, object],
    image_path: Path,
    stored_path: str,
    dpi: int,
) -> None:
    page_id = str(existing.get("page_id"))
    digest = sha256_file(image_path)
    width, height = png_size(image_path)
    expected = {
        "sha256": digest,
        "width": width,
        "height": height,
        "dpi": dpi,
        "image_path": stored_path,
        "status": StageStatus.COMPLETE.value,
    }
    mismatches = [
        key for key, value in expected.items() if existing.get(key) != value
    ]
    if mismatches:
        raise RenderConflictError(
            f"{page_id} already exists with a different payload ({', '.join(mismatches)}). "
            f"{RESUME_NOTE}"
        )


def _sha256_bytes(payload: bytes) -> str:
    return sha256_bytes(payload)


def format_report(report: RenderReport) -> str:
    """Return the operator-facing rendering summary."""
    lines = [
        "Phase 3: PDF rendering",
        "",
        f"Documents discovered from manifest: {report.documents}",
        "",
    ]
    for stats in report.details:
        lines.append(stats.document_id)
        lines.append(f"  pages: {stats.pages}")
        lines.append(f"  rendered: {stats.rendered}")
        lines.append(f"  skipped: {stats.skipped}")
        if stats.message:
            lines.append(f"  error: {stats.message}")
        lines.append("")
    lines.extend(
        [
            "Rendering complete.",
            f"  documents: {report.documents}",
            f"  pages discovered: {report.pages_discovered}",
            f"  pages rendered: {report.pages_rendered}",
            f"  pages skipped: {report.pages_skipped}",
            f"  errors: {report.errors}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Render pages for every document in the document manifest."""
    from indic_ocr_pipeline.config import load_settings

    parser = argparse.ArgumentParser(
        description="Render ingested PDF pages to PNG images. Source PDFs are not modified."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    started = datetime.now(UTC)
    try:
        settings = load_settings()
        documents = open_manifest(settings.manifests_dir, "documents")
        pages = open_manifest(settings.manifests_dir, "pages")
        report = render_documents(
            documents.load(),
            pages,
            settings.rendered_pages_dir,
            repo_root=settings.repo_root,
            dpi=settings.render.dpi,
        )
    except (RenderError, ManifestConflictError, SchemaError, pdfium.PdfiumError) as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    duration = (datetime.now(UTC) - started).total_seconds()
    print(format_report(report), flush=True)
    logger.info(
        "rendering finished",
        extra={
            "processed": report.pages_discovered,
            "completed": report.pages_rendered,
            "skipped": report.pages_skipped,
            "failed": report.errors,
            "duration_seconds": round(duration, 3),
        },
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
