"""Hash source PDFs and append document-manifest rows.

This stage reads PDF bytes and writes ``manifests/documents.jsonl``.
It does not render pages, read a text layer, or store a transcription.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from indic_ocr_pipeline.errors import DatasetError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.ids import DocumentIdAllocator
from indic_ocr_pipeline.manifests import JsonlManifest, open_manifest
from indic_ocr_pipeline.schemas import DocumentRecord, StageStatus
from indic_ocr_pipeline.utils.hashing import sha256_file
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PREPROCESS_NOTE = (
    "Input is full PDF documents. Each PDF is rendered to page images, "
    "then cut into line crops, before OCR runs."
)

RESUME_NOTE = "Existing manifest rows were kept. Ingestion can resume."


class IngestError(DatasetError):
    """A source PDF could not be ingested."""


@dataclass(frozen=True)
class IngestRow:
    """One discovered PDF and what ingestion did with it."""

    action: str
    filename: str
    document_id: str | None = None
    sha256: str | None = None
    message: str = ""


@dataclass
class IngestReport:
    """Counters for one ingestion run."""

    discovered: int = 0
    ingested: int = 0
    skipped: int = 0
    errors: int = 0
    rows: list[IngestRow] = field(default_factory=list)

    def add(self, row: IngestRow) -> None:
        self.rows.append(row)
        if row.action == "ingested":
            self.ingested += 1
        elif row.action == "skipped":
            self.skipped += 1
        elif row.action == "error":
            self.errors += 1


def discover_pdfs(documents_dir: Path) -> list[Path]:
    """Return top-level PDF files in ``documents_dir``, sorted by filename.

    Discovery is not recursive. ``.pdf`` and ``.PDF`` both match.
    Other files are ignored.
    """
    if not documents_dir.exists():
        raise IngestError(
            f"input directory does not exist: {documents_dir}. {RESUME_NOTE}"
        )
    if not documents_dir.is_dir():
        raise IngestError(
            f"input path is not a directory: {documents_dir}. {RESUME_NOTE}"
        )
    pdfs = [
        path
        for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    return sorted(pdfs, key=lambda path: (path.name.casefold(), path.name))


def source_path_for(path: Path, repo_root: Path | None) -> str:
    """Store a repository-relative path when the PDF lives inside the repo."""
    resolved = path.resolve()
    if repo_root is not None:
        try:
            return resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def commit_document(manifest: JsonlManifest, record: DocumentRecord) -> str:
    """Validate and append one document row.

    The same id and payload is a resume skip. A different payload raises
    ``ManifestConflictError`` and does not rewrite the row.
    """
    payload = record.to_dict()
    try:
        return manifest.append(payload)
    except ManifestConflictError as exc:
        raise ManifestConflictError(
            f"{exc}. filename={record.original_filename} "
            f"document_id={record.document_id} sha256={record.sha256}. {RESUME_NOTE}"
        ) from exc


def ingest_documents(
    documents_dir: Path,
    manifest: JsonlManifest,
    *,
    repo_root: Path | None = None,
    clock: Callable[[], str] | None = None,
    max_documents: int | None = None,
) -> IngestReport:
    """Ingest PDFs from ``documents_dir`` into ``manifest``.

    Source files are only read. Known SHA-256 values keep their document ID
    and are skipped. New hashes receive the next ID from ``DocumentIdAllocator``.
    ``max_documents`` keeps only the first PDFs in filename order.
    """
    pdfs = discover_pdfs(documents_dir)
    if max_documents is not None:
        pdfs = pdfs[:max_documents]
    report = IngestReport(discovered=len(pdfs))
    existing = manifest.load()
    allocator = DocumentIdAllocator.from_records(existing)
    known_hashes = {
        str(record["sha256"]): record
        for record in existing
        if isinstance(record.get("sha256"), str)
    }
    timestamp = (clock or utc_now)()
    for path in pdfs:
        _ingest_one(
            path,
            manifest=manifest,
            allocator=allocator,
            known_hashes=known_hashes,
            repo_root=repo_root,
            timestamp=timestamp,
            report=report,
        )
        if report.errors:
            break
    return report


def _ingest_one(
    path: Path,
    *,
    manifest: JsonlManifest,
    allocator: DocumentIdAllocator,
    known_hashes: dict[str, dict[str, object]],
    repo_root: Path | None,
    timestamp: str,
    report: IngestReport,
) -> None:
    filename = path.name
    try:
        if not path.is_file():
            raise IngestError(f"{filename} is not a regular file. {RESUME_NOTE}")
        size_before = path.stat().st_size
        digest = sha256_file(path)
        size_after = path.stat().st_size
        if size_before != size_after:
            raise IngestError(
                f"{filename} changed while it was being read. {RESUME_NOTE}"
            )
        known = known_hashes.get(digest)
        if known is not None:
            document_id = known.get("document_id")
            if not isinstance(document_id, str):
                raise IngestError(
                    f"{filename} matches sha256 {digest} but the manifest row has no "
                    f"document_id. {RESUME_NOTE}"
                )
            report.add(
                IngestRow(
                    action="skipped",
                    filename=filename,
                    document_id=document_id,
                    sha256=digest,
                    message="already ingested",
                )
            )
            return
        document_id = allocator.assign(digest)
        record = DocumentRecord(
            document_id=document_id,
            original_filename=filename,
            source_path=source_path_for(path, repo_root),
            sha256=digest,
            file_size=size_after,
            status=StageStatus.COMPLETE,
            ingested_at=timestamp,
        )
        outcome = commit_document(manifest, record)
        known_hashes[digest] = record.to_dict()
        action = "skipped" if outcome == "skipped" else "ingested"
        report.add(
            IngestRow(
                action=action,
                filename=filename,
                document_id=document_id,
                sha256=digest,
                message="already ingested" if action == "skipped" else "",
            )
        )
    except (OSError, SchemaError, ManifestConflictError, IngestError) as exc:
        report.add(
            IngestRow(
                action="error",
                filename=filename,
                message=str(exc),
            )
        )


def format_report(report: IngestReport) -> str:
    """Return the operator-facing ingestion summary."""
    lines = [
        "Discovering documents...",
        f"Found {report.discovered} PDF files.",
        "",
    ]
    for row in report.rows:
        if row.action == "ingested" or row.action == "skipped":
            label = "INGESTED" if row.action == "ingested" else "SKIP"
            suffix = "  already ingested" if row.action == "skipped" else ""
            lines.append(f"{label:<8}  {row.document_id}  {row.filename}{suffix}")
            continue
        detail = row.message or "ingestion failed"
        if RESUME_NOTE not in detail:
            detail = f"{detail} {RESUME_NOTE}"
        lines.append(f"ERROR  {row.filename}  {detail}")
    lines.extend(
        [
            "",
            "Ingestion complete.",
            f"discovered: {report.discovered}",
            f"ingested: {report.ingested}",
            f"skipped: {report.skipped}",
            f"errors: {report.errors}",
        ]
    )
    return "\n".join(lines)


def _document_cap(documents_dir: Path) -> int | None:
    """How many leading PDFs to ingest for the current dataset-size budget."""
    from indic_ocr_pipeline.limits import dataset_size_from_env, pages_to_keep, pdf_page_count

    size = dataset_size_from_env()
    if size is None:
        return None
    counts: list[int] = []
    for path in discover_pdfs(documents_dir):
        counts.append(pdf_page_count(path))
        plan = pages_to_keep(counts, size)
        if len(plan) < len(counts):
            break
    plan = pages_to_keep(counts, size)
    print(
        f"dataset-size {size}: ingesting {len(plan)} PDF(s), "
        f"{sum(plan)} page(s).",
        flush=True,
    )
    return len(plan)


def main(argv: list[str] | None = None) -> int:
    """Ingest PDFs from the configured documents directory."""
    from indic_ocr_pipeline.config import load_settings

    parser = argparse.ArgumentParser(
        description=(
            "Hash source PDFs into the document manifest. "
            "Source PDFs are not modified."
        )
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    print(PREPROCESS_NOTE, flush=True)
    logger.info(PREPROCESS_NOTE)
    started = datetime.now(UTC)
    try:
        settings = load_settings()
        manifest = open_manifest(settings.manifests_dir, "documents")
        report = ingest_documents(
            settings.documents_dir,
            manifest,
            repo_root=settings.repo_root,
            max_documents=_document_cap(settings.documents_dir),
        )
    except (IngestError, ManifestConflictError, SchemaError) as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    duration = (datetime.now(UTC) - started).total_seconds()
    print(format_report(report), flush=True)
    logger.info(
        "ingestion finished",
        extra={
            "processed": report.discovered,
            "completed": report.ingested,
            "skipped": report.skipped,
            "failed": report.errors,
            "duration_seconds": round(duration, 3),
        },
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
