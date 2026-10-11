"""Deterministic identifiers.

Document IDs are allocated once per SHA-256 and then never reused.
Page, region, line, crop, prediction, and sample IDs are pure functions of
their parents, so a rerun with the same inputs reproduces them.

IDs are hierarchical so a sample can be traced to its source document
without a lookup table. They are not a global counter of crops.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from indic_ocr_pipeline.errors import ManifestConflictError

DOCUMENT_PREFIX = "MHGR"
_SEQUENCE_WIDTH = 6
_PAGE_WIDTH = 4
_REGION_WIDTH = 4
_LINE_WIDTH = 4
_CROP_WIDTH = 2
_MAX_SEQUENCE = 10**_SEQUENCE_WIDTH - 1
_MAX_PAGE = 10**_PAGE_WIDTH - 1
_MAX_REGION = 10**_REGION_WIDTH - 1
_MAX_LINE = 10**_LINE_WIDTH - 1
_MAX_CROP_INDEX = 10**_CROP_WIDTH - 1

_DOCUMENT_RE = re.compile(rf"^{DOCUMENT_PREFIX}_(?P<seq>\d{{{_SEQUENCE_WIDTH}}})$")
_PAGE_RE = re.compile(
    rf"^(?P<document>{DOCUMENT_PREFIX}_\d{{{_SEQUENCE_WIDTH}}})_P(?P<page>\d{{{_PAGE_WIDTH}}})$"
)
_REGION_RE = re.compile(
    rf"^(?P<page>{DOCUMENT_PREFIX}_\d{{{_SEQUENCE_WIDTH}}}_P\d{{{_PAGE_WIDTH}}})_R(?P<index>\d{{{_REGION_WIDTH}}})$"
)
_LINE_RE = re.compile(
    rf"^(?P<page>{DOCUMENT_PREFIX}_\d{{{_SEQUENCE_WIDTH}}}_P\d{{{_PAGE_WIDTH}}})_L(?P<index>\d{{{_LINE_WIDTH}}})$"
)
_CROP_RE = re.compile(
    rf"^(?P<line>{DOCUMENT_PREFIX}_\d{{{_SEQUENCE_WIDTH}}}_P\d{{{_PAGE_WIDTH}}}_L\d{{{_LINE_WIDTH}}})"
    rf"_C_(?P<crop_type>[a-z][a-z0-9_]*)_(?P<index>\d{{{_CROP_WIDTH}}})$"
)
_BACKEND_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_index(value: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be in 1..{maximum}")


def format_document_id(sequence: int) -> str:
    """Return ``MHGR_######`` for a 1-based sequence number."""
    _require_index(sequence, _MAX_SEQUENCE, "sequence")
    return f"{DOCUMENT_PREFIX}_{sequence:0{_SEQUENCE_WIDTH}d}"


def parse_document_sequence(document_id: str) -> int:
    """Return the sequence number stored in a document ID."""
    match = _DOCUMENT_RE.fullmatch(document_id)
    if match is None:
        raise ValueError(f"invalid document_id: {document_id!r}")
    return int(match.group("seq"))


def validate_document_id(document_id: str) -> str:
    """Return ``document_id`` when it has the required form."""
    parse_document_sequence(document_id)
    return document_id


def make_page_id(document_id: str, page_number: int) -> str:
    """Return the page ID for a 1-based page number."""
    validate_document_id(document_id)
    _require_index(page_number, _MAX_PAGE, "page_number")
    return f"{document_id}_P{page_number:0{_PAGE_WIDTH}d}"


def validate_page_id(page_id: str) -> str:
    """Return ``page_id`` when it has the required form."""
    if _PAGE_RE.fullmatch(page_id) is None:
        raise ValueError(f"invalid page_id: {page_id!r}")
    return page_id


def make_region_id(page_id: str, region_index: int) -> str:
    """Return the region ID for a 1-based region index on a page."""
    validate_page_id(page_id)
    _require_index(region_index, _MAX_REGION, "region_index")
    return f"{page_id}_R{region_index:0{_REGION_WIDTH}d}"


def validate_region_id(region_id: str) -> str:
    """Return ``region_id`` when it has the required form."""
    if _REGION_RE.fullmatch(region_id) is None:
        raise ValueError(f"invalid region_id: {region_id!r}")
    return region_id


def make_line_id(page_id: str, line_index: int) -> str:
    """Return the parent line ID for a 1-based line index on a page."""
    validate_page_id(page_id)
    _require_index(line_index, _MAX_LINE, "line_index")
    return f"{page_id}_L{line_index:0{_LINE_WIDTH}d}"


def validate_line_id(line_id: str) -> str:
    """Return ``line_id`` when it has the required form."""
    if _LINE_RE.fullmatch(line_id) is None:
        raise ValueError(f"invalid line_id: {line_id!r}")
    return line_id


def make_crop_id(parent_line_id: str, crop_type: str, crop_index: int) -> str:
    """Return the crop ID for one crop of a parent line."""
    validate_line_id(parent_line_id)
    if not isinstance(crop_type, str) or re.fullmatch(r"[a-z][a-z0-9_]*", crop_type) is None:
        raise ValueError(f"invalid crop_type: {crop_type!r}")
    _require_index(crop_index, _MAX_CROP_INDEX, "crop_index")
    return f"{parent_line_id}_C_{crop_type}_{crop_index:0{_CROP_WIDTH}d}"


def validate_crop_id(crop_id: str) -> str:
    """Return ``crop_id`` when it has the required form."""
    if _CROP_RE.fullmatch(crop_id) is None:
        raise ValueError(f"invalid crop_id: {crop_id!r}")
    return crop_id


def make_sample_id(crop_id: str) -> str:
    """Return the gold sample ID for a crop.

    The sample ID is the crop ID with a prefix. The same crop always receives
    the same sample ID, including after a resume.
    """
    validate_crop_id(crop_id)
    return f"SAMPLE_{crop_id}"


def validate_sample_id(sample_id: str) -> str:
    """Return ``sample_id`` when it has the required form."""
    if not sample_id.startswith("SAMPLE_"):
        raise ValueError(f"invalid sample_id: {sample_id!r}")
    validate_crop_id(sample_id.removeprefix("SAMPLE_"))
    return sample_id


def make_prediction_id(crop_id: str, backend: str) -> str:
    """Return the prediction ID for one backend on one crop."""
    validate_crop_id(crop_id)
    if _BACKEND_RE.fullmatch(backend) is None:
        raise ValueError(f"invalid backend name: {backend!r}")
    return f"{crop_id}_OCR_{backend}"


def make_candidate_id(crop_id: str) -> str:
    """Return the candidate-comparison ID for a crop."""
    validate_crop_id(crop_id)
    return f"{crop_id}_CAND"


def make_review_id(crop_id: str) -> str:
    """Return the review-record ID for a crop."""
    validate_crop_id(crop_id)
    return f"{crop_id}_REV"


def make_verified_id(crop_id: str) -> str:
    """Return the verified-decision ID for a crop."""
    validate_crop_id(crop_id)
    return f"{crop_id}_VER"


def document_id_from_identifier(identifier: str) -> str:
    """Recover the document ID prefix from any pipeline identifier."""
    token = identifier.removeprefix("SAMPLE_")
    match = re.match(rf"^({DOCUMENT_PREFIX}_\d{{{_SEQUENCE_WIDTH}}})", token)
    if match is None:
        raise ValueError(f"identifier has no document_id: {identifier!r}")
    return match.group(1)


def parent_line_id_from_crop_id(crop_id: str) -> str:
    """Return the parent line ID embedded in a crop ID."""
    validate_crop_id(crop_id)
    marker = "_C_"
    return crop_id[: crop_id.rfind(marker)]


def _require_sha256(sha256: str) -> str:
    if _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return sha256


@dataclass
class DocumentIdAllocator:
    """Assign stable document IDs from source SHA-256 values.

    Existing manifest rows win. A SHA-256 that is already recorded keeps its
    document ID. A new SHA-256 receives the next unused sequence number.
    Sequence numbers are never reused, including when the manifest has gaps.
    """

    by_sha256: dict[str, str] = field(default_factory=dict)
    by_document_id: dict[str, str] = field(default_factory=dict)
    next_sequence: int = 1

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> DocumentIdAllocator:
        """Rebuild an allocator from document-manifest rows."""
        allocator = cls()
        highest = 0
        for record in records:
            sha256 = record.get("sha256")
            document_id = record.get("document_id")
            if not isinstance(sha256, str) or not isinstance(document_id, str):
                raise ManifestConflictError("document records require sha256 and document_id")
            _require_sha256(sha256)
            sequence = parse_document_sequence(document_id)
            previous_sha = allocator.by_document_id.get(document_id)
            previous_document = allocator.by_sha256.get(sha256)
            if previous_sha is not None and previous_sha != sha256:
                raise ManifestConflictError(
                    f"{document_id} is already assigned to a different sha256"
                )
            if previous_document is not None and previous_document != document_id:
                raise ManifestConflictError(
                    f"sha256 {sha256} is already assigned to {previous_document}"
                )
            allocator.by_sha256[sha256] = document_id
            allocator.by_document_id[document_id] = sha256
            highest = max(highest, sequence)
        allocator.next_sequence = highest + 1
        return allocator

    def assign(self, sha256: str) -> str:
        """Return the document ID for ``sha256``, allocating one if needed."""
        _require_sha256(sha256)
        existing = self.by_sha256.get(sha256)
        if existing is not None:
            return existing
        if self.next_sequence > _MAX_SEQUENCE:
            raise ManifestConflictError("document ID sequence is exhausted")
        document_id = format_document_id(self.next_sequence)
        self.next_sequence += 1
        self.by_sha256[sha256] = document_id
        self.by_document_id[document_id] = sha256
        return document_id
