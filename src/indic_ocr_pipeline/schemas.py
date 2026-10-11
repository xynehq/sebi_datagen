"""Manifest record schemas.

Three ideas stay separate in these records:

* the rendered page image is the visual source
* an OCR prediction is a candidate
* a gold label is text a person confirmed while looking at the crop

``auto_approved_candidate`` is a review-priority status. It is not gold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import StrEnum
from typing import Any

from indic_ocr_pipeline.errors import SchemaError
from indic_ocr_pipeline.ids import (
    document_id_from_identifier,
    make_crop_id,
    make_line_id,
    make_page_id,
    make_region_id,
    validate_crop_id,
    validate_document_id,
    validate_line_id,
    validate_page_id,
    validate_region_id,
    validate_sample_id,
)
from indic_ocr_pipeline.text_policy import has_replacement_character, is_storage_form
from indic_ocr_pipeline.utils.hashing import is_sha256_hex
from indic_ocr_pipeline.utils.jsonutil import assert_json_value, to_jsonable

ALLOWED_DPI = frozenset({150, 200, 250, 300, 400})
PDF_TEXT_LAYER_ROLE = "weak_diagnostic_not_ground_truth"
SCORE_IS_NOT_A_PROBABILITY = (
    "confidence_score and risk_score rank human review. "
    "They are not a calibrated probability that a transcription is correct."
)


class StageStatus(StrEnum):
    """Resumable status of one manifest row."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationStatus(StrEnum):
    """How a transcription was produced.

    Only ``human_verified`` and ``human_corrected`` may enter the gold dataset.
    """

    CANDIDATE = "candidate"
    AUTO_APPROVED_CANDIDATE = "auto_approved_candidate"
    HUMAN_VERIFIED = "human_verified"
    HUMAN_CORRECTED = "human_corrected"
    REJECTED = "rejected"


class QualityTier(StrEnum):
    """Dataset tier. Tiers are not merged. Only tier 5 is trainable gold."""

    TIER_0_SOURCE = "tier_0_source"
    TIER_1_CROP = "tier_1_crop"
    TIER_2_CANDIDATE = "tier_2_candidate"
    TIER_3_HIGH_CONFIDENCE_CANDIDATE = "tier_3_high_confidence_candidate"
    TIER_4_HUMAN_REVIEWED = "tier_4_human_reviewed"
    TIER_5_GOLD = "tier_5_gold"


class ReviewDecision(StrEnum):
    """Decision a reviewer records while looking at the crop image."""

    ACCEPT_CANDIDATE = "accept_candidate"
    CORRECT_TEXT = "correct_text"
    REJECT_CROP = "reject_crop"
    BAD_BOUNDARY = "bad_boundary"
    BAD_IMAGE = "bad_image"
    NON_TEXT = "non_text"
    UNCERTAIN = "uncertain"


class RegionType(StrEnum):
    """Coarse region class from detection. Detection does not read the text."""

    TEXT_BLOCK = "text_block"
    LINE = "line"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class CropType(StrEnum):
    """Which slice of a parent line a crop contains."""

    FULL = "full"
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    RANDOM = "random"
    MULTI_LINE = "multi_line"


class SplitName(StrEnum):
    """Document-level split name."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DisagreementType(StrEnum):
    """Why OCR candidates differ. ``none`` means the compared texts match."""

    NONE = "none"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    NUMERAL = "numeral"
    UNICODE_NORMALIZATION = "unicode_normalization"
    CHARACTER_SUBSTITUTION = "character_substitution"
    EMPTY = "empty"
    OTHER = "other"


class ScoreReason(StrEnum):
    """Interpretable reason codes. A score without these codes is incomplete."""

    ALL_BACKENDS_AGREE = "all_backends_agree"
    PARTIAL_AGREEMENT = "partial_agreement"
    BACKENDS_DISAGREE = "backends_disagree"
    TESSERACT_HIGH_CONFIDENCE = "tesseract_high_confidence"
    TESSERACT_LOW_CONFIDENCE = "tesseract_low_confidence"
    TESSERACT_CONFIDENCE_UNAVAILABLE = "tesseract_confidence_unavailable"
    MODEL_CONFIDENCE_UNAVAILABLE = "model_confidence_unavailable"
    CLEAN_CROP = "clean_crop"
    SUSPICIOUS_CROP_BOUNDARY = "suspicious_crop_boundary"
    LOW_IMAGE_QUALITY = "low_image_quality"
    VALID_UNICODE = "valid_unicode"
    SUSPICIOUS_UNICODE = "suspicious_unicode"
    EMPTY_OUTPUT = "empty_output"
    VERY_SHORT_OUTPUT = "very_short_output"
    NUMERIC_MISMATCH = "numeric_mismatch"
    PUNCTUATION_DISAGREEMENT = "punctuation_disagreement"
    NO_NUMERIC_CONFLICT = "no_numeric_conflict"


GOLD_VERIFICATION_STATUSES = frozenset(
    {
        VerificationStatus.HUMAN_VERIFIED,
        VerificationStatus.HUMAN_CORRECTED,
    }
)

CANDIDATE_VERIFICATION_STATUSES = frozenset(
    {
        VerificationStatus.CANDIDATE,
        VerificationStatus.AUTO_APPROVED_CANDIDATE,
    }
)


def admits_to_gold(status: VerificationStatus | str) -> bool:
    """Return whether a verification status may be copied into gold."""
    try:
        parsed = VerificationStatus(status)
    except ValueError:
        return False
    return parsed in GOLD_VERIFICATION_STATUSES


def decision_creates_gold(decision: ReviewDecision | str) -> bool:
    """Return whether a review decision can produce a gold sample."""
    try:
        parsed = ReviewDecision(decision)
    except ValueError:
        return False
    return parsed in {ReviewDecision.ACCEPT_CANDIDATE, ReviewDecision.CORRECT_TEXT}


@dataclass(frozen=True)
class BBox:
    """Inclusive-exclusive pixel box ``[x0, y0, x1, y1]`` in page coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaError(f"bbox.{name} must be an integer")
        if self.x0 < 0 or self.y0 < 0:
            raise SchemaError("bbox origin must be non-negative")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise SchemaError("bbox must have positive width and height")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    def contains(self, other: BBox) -> bool:
        """Return whether ``other`` lies entirely inside this box."""
        return (
            other.x0 >= self.x0
            and other.y0 >= self.y0
            and other.x1 <= self.x1
            and other.y1 <= self.y1
        )

    @classmethod
    def from_list(cls, value: object) -> BBox:
        """Build a box from ``[x0, y0, x1, y1]``."""
        if not isinstance(value, list | tuple) or len(value) != 4:
            raise SchemaError("bbox must be [x0, y0, x1, y1]")
        return cls(value[0], value[1], value[2], value[3])


def assert_crop_within_parent(crop: BBox, parent: BBox) -> None:
    """Require a crop box to stay inside its parent line box."""
    if not parent.contains(crop):
        raise SchemaError("crop bbox must lie inside the parent line bbox")


def assert_bbox_within_page(bbox: BBox, width: int, height: int) -> None:
    """Require a box to stay inside a rendered page."""
    if isinstance(width, bool) or isinstance(height, bool):
        raise SchemaError("page width and height must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise SchemaError("page width and height must be integers")
    if width < 1 or height < 1:
        raise SchemaError("page width and height must be positive")
    if bbox.x1 > width or bbox.y1 > height:
        raise SchemaError("bbox extends outside the rendered page")


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must be a string or null")
    return value


def _require_int(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{field_name} must be >= {minimum}")
    return value


def _optional_finite(value: object, field_name: str, *, low: float, high: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaError(f"{field_name} must be a number or null")
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise SchemaError(f"{field_name} must be between {low} and {high}")
    return number


def _require_sha(value: object, field_name: str) -> str:
    text = _require_str(value, field_name)
    if not is_sha256_hex(text):
        raise SchemaError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


def _optional_sha(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_sha(value, field_name)


def _parse_enum(enum_cls: type[StrEnum], value: object, field_name: str) -> StrEnum:
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must be a string")
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise SchemaError(f"{field_name} must be one of: {allowed}") from exc


def _optional_timestamp(value: object, field_name: str) -> str | None:
    text = _optional_str(value, field_name)
    if text is None or text == "":
        if text == "":
            raise SchemaError(f"{field_name} must be an ISO-8601 timestamp or null")
        return None
    _require_timestamp(text, field_name)
    return text


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_str(value, field_name)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError(f"{field_name} must be an ISO-8601 timestamp") from exc
    return text


def _require_mapping(data: object, kind: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaError(f"{kind} must be a JSON object")
    return data


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], kind: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise SchemaError(f"{kind} has unknown fields: {', '.join(extra)}")


def _require_fields(data: Mapping[str, Any], required: set[str], kind: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise SchemaError(f"{kind} is missing fields: {', '.join(missing)}")


def _metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SchemaError("metadata must be an object")
    try:
        assert_json_value(value, "metadata")
    except ValueError as exc:
        raise SchemaError(str(exc)) from exc
    return dict(value)


def _string_map(
    value: object, field_name: str, *, allow_null_values: bool
) -> dict[str, str | None]:
    if not isinstance(value, dict):
        raise SchemaError(f"{field_name} must be an object")
    parsed: dict[str, str | None] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise SchemaError(f"{field_name} keys must be non-empty strings")
        if item is None and allow_null_values:
            parsed[key] = None
            continue
        if not isinstance(item, str):
            raise SchemaError(f"{field_name}.{key} must be a string")
        parsed[key] = item
    return parsed


def _bool_map(value: object, field_name: str) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise SchemaError(f"{field_name} must be an object")
    parsed: dict[str, bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise SchemaError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, bool):
            raise SchemaError(f"{field_name}.{key} must be a boolean")
        parsed[key] = item
    return parsed


def _reason_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError("reasons must be a list")
    reasons: list[str] = []
    for item in value:
        parsed = _parse_enum(ScoreReason, item, "reasons")
        reasons.append(str(parsed.value))
    return reasons


def _token_confidences(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise SchemaError("token_confidences must be a list or null")
    scores: list[float] = []
    for index, item in enumerate(value):
        number = _optional_finite(item, f"token_confidences[{index}]", low=0.0, high=1.0)
        if number is None:
            raise SchemaError(f"token_confidences[{index}] must be a number")
        scores.append(number)
    return scores


class Record:
    """JSON round-trip for a dataclass record."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``SchemaError`` when the record breaks a policy."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON object. Validation runs before serialization."""
        self.validate()
        data: dict[str, Any] = {}
        for item in fields(self):  # type: ignore[arg-type]
            data[item.name] = to_jsonable(getattr(self, item.name))
        return data


def _ids_match_page(document_id: str, page_id: str, page_number: int) -> None:
    validate_document_id(document_id)
    expected = make_page_id(document_id, page_number)
    if page_id != expected:
        raise SchemaError(f"page_id must be {expected}")


@dataclass
class DocumentRecord(Record):
    """One source PDF. Ingestion must not modify the file bytes."""

    document_id: str
    original_filename: str
    source_path: str
    sha256: str
    file_size: int
    source_url: str | None = None
    download_timestamp: str | None = None
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    department: str | None = None
    document_type: str | None = None
    document_date: str | None = None
    status: StageStatus = StageStatus.PENDING
    error: str | None = None
    ingested_at: str | None = None

    def validate(self) -> None:
        validate_document_id(self.document_id)
        _require_str(self.original_filename, "original_filename")
        _require_str(self.source_path, "source_path")
        _require_sha(self.sha256, "sha256")
        _require_int(self.file_size, "file_size", minimum=0)
        _optional_str(self.source_url, "source_url")
        _optional_timestamp(self.download_timestamp, "download_timestamp")
        if self.page_count is not None:
            _require_int(self.page_count, "page_count", minimum=1)
        _metadata(self.metadata)
        _optional_str(self.department, "department")
        _optional_str(self.document_type, "document_type")
        _optional_str(self.document_date, "document_date")
        _parse_enum(StageStatus, self.status, "status")
        _optional_str(self.error, "error")
        _optional_timestamp(self.ingested_at, "ingested_at")

    @classmethod
    def from_dict(cls, data: object) -> DocumentRecord:
        raw = _require_mapping(data, "DocumentRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "DocumentRecord")
        _require_fields(
            raw,
            {"document_id", "original_filename", "source_path", "sha256", "file_size"},
            "DocumentRecord",
        )
        record = cls(
            document_id=str(raw["document_id"]),
            original_filename=str(raw["original_filename"]),
            source_path=str(raw["source_path"]),
            sha256=str(raw["sha256"]),
            file_size=raw["file_size"],
            source_url=raw.get("source_url"),
            download_timestamp=raw.get("download_timestamp"),
            page_count=raw.get("page_count"),
            metadata=raw.get("metadata") or {},
            department=raw.get("department"),
            document_type=raw.get("document_type"),
            document_date=raw.get("document_date"),
            status=_parse_enum(StageStatus, raw.get("status", StageStatus.PENDING), "status"),  # type: ignore[arg-type]
            error=raw.get("error"),
            ingested_at=raw.get("ingested_at"),
        )
        record.validate()
        return record


@dataclass
class PageRecord(Record):
    """One rendered page image. The image is the visual source of truth."""

    document_id: str
    page_id: str
    page_number: int
    image_path: str
    width: int
    height: int
    dpi: int
    sha256: str
    pdf_text_layer: str | None = None
    pdf_text_layer_role: str = PDF_TEXT_LAYER_ROLE
    status: StageStatus = StageStatus.PENDING
    error: str | None = None
    rendered_at: str | None = None

    def validate(self) -> None:
        _ids_match_page(self.document_id, self.page_id, self.page_number)
        _require_str(self.image_path, "image_path")
        _require_int(self.width, "width", minimum=1)
        _require_int(self.height, "height", minimum=1)
        if self.dpi not in ALLOWED_DPI:
            raise SchemaError(f"dpi must be one of {sorted(ALLOWED_DPI)}")
        _require_sha(self.sha256, "sha256")
        _optional_str(self.pdf_text_layer, "pdf_text_layer")
        if self.pdf_text_layer_role != PDF_TEXT_LAYER_ROLE:
            raise SchemaError(
                "pdf_text_layer_role must stay "
                f"{PDF_TEXT_LAYER_ROLE}; extracted PDF text is not ground truth"
            )
        _parse_enum(StageStatus, self.status, "status")
        _optional_str(self.error, "error")
        _optional_timestamp(self.rendered_at, "rendered_at")

    @classmethod
    def from_dict(cls, data: object) -> PageRecord:
        raw = _require_mapping(data, "PageRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "PageRecord")
        _require_fields(
            raw,
            {
                "document_id",
                "page_id",
                "page_number",
                "image_path",
                "width",
                "height",
                "dpi",
                "sha256",
            },
            "PageRecord",
        )
        record = cls(
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            page_number=raw["page_number"],
            image_path=str(raw["image_path"]),
            width=raw["width"],
            height=raw["height"],
            dpi=raw["dpi"],
            sha256=str(raw["sha256"]),
            pdf_text_layer=raw.get("pdf_text_layer"),
            pdf_text_layer_role=str(raw.get("pdf_text_layer_role", PDF_TEXT_LAYER_ROLE)),
            status=_parse_enum(StageStatus, raw.get("status", StageStatus.PENDING), "status"),  # type: ignore[arg-type]
            error=raw.get("error"),
            rendered_at=raw.get("rendered_at"),
        )
        record.validate()
        return record


@dataclass
class RegionRecord(Record):
    """A detected region. This record does not contain a transcription."""

    region_id: str
    document_id: str
    page_id: str
    page_number: int
    region_index: int
    region_type: RegionType
    bbox: BBox
    detection_method: str
    confidence: float | None = None
    parent_region_id: str | None = None
    line_id: str | None = None
    line_index: int | None = None
    reading_order: int = 1
    status: StageStatus = StageStatus.COMPLETE
    error: str | None = None

    def validate(self) -> None:
        _ids_match_page(self.document_id, self.page_id, self.page_number)
        expected_region = make_region_id(self.page_id, self.region_index)
        if self.region_id != expected_region:
            raise SchemaError(f"region_id must be {expected_region}")
        _parse_enum(RegionType, self.region_type, "region_type")
        if not isinstance(self.bbox, BBox):
            raise SchemaError("bbox must be a BBox")
        _require_str(self.detection_method, "detection_method")
        _optional_finite(self.confidence, "confidence", low=0.0, high=1.0)
        if self.parent_region_id is not None:
            validate_region_id(self.parent_region_id)
            if document_id_from_identifier(self.parent_region_id) != self.document_id:
                raise SchemaError("parent_region_id belongs to a different document")
        if self.region_type is RegionType.LINE:
            if self.line_index is None or self.line_id is None:
                raise SchemaError("line regions require line_id and line_index")
            expected_line = make_line_id(self.page_id, self.line_index)
            if self.line_id != expected_line:
                raise SchemaError(f"line_id must be {expected_line}")
        elif self.line_id is not None or self.line_index is not None:
            raise SchemaError("line_id is only valid when region_type is line")
        _require_int(self.reading_order, "reading_order", minimum=1)
        _parse_enum(StageStatus, self.status, "status")
        _optional_str(self.error, "error")

    @classmethod
    def from_dict(cls, data: object) -> RegionRecord:
        raw = _require_mapping(data, "RegionRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "RegionRecord")
        _require_fields(
            raw,
            {
                "region_id",
                "document_id",
                "page_id",
                "page_number",
                "region_index",
                "region_type",
                "bbox",
                "detection_method",
            },
            "RegionRecord",
        )
        record = cls(
            region_id=str(raw["region_id"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            page_number=raw["page_number"],
            region_index=raw["region_index"],
            region_type=_parse_enum(RegionType, raw["region_type"], "region_type"),  # type: ignore[arg-type]
            bbox=BBox.from_list(raw["bbox"]),
            detection_method=str(raw["detection_method"]),
            confidence=raw.get("confidence"),
            parent_region_id=raw.get("parent_region_id"),
            line_id=raw.get("line_id"),
            line_index=raw.get("line_index"),
            reading_order=raw.get("reading_order", 1),
            status=_parse_enum(StageStatus, raw.get("status", StageStatus.COMPLETE), "status"),  # type: ignore[arg-type]
            error=raw.get("error"),
        )
        record.validate()
        return record


@dataclass
class CropRecord(Record):
    """An image crop. Crop generation stores geometry and does not store a label."""

    crop_id: str
    document_id: str
    page_id: str
    page_number: int
    region_id: str
    parent_line_id: str
    crop_type: CropType
    crop_index: int
    image_path: str
    bbox: BBox
    parent_bbox: BBox
    random_seed: int | None = None
    sha256: str | None = None
    multi_line_enabled: bool = False
    status: StageStatus = StageStatus.PENDING
    error: str | None = None

    def validate(self) -> None:
        _ids_match_page(self.document_id, self.page_id, self.page_number)
        validate_region_id(self.region_id)
        validate_line_id(self.parent_line_id)
        if document_id_from_identifier(self.region_id) != self.document_id:
            raise SchemaError("region_id belongs to a different document")
        if document_id_from_identifier(self.parent_line_id) != self.document_id:
            raise SchemaError("parent_line_id belongs to a different document")
        if not self.region_id.startswith(self.page_id + "_R"):
            raise SchemaError("region_id must belong to page_id")
        if not self.parent_line_id.startswith(self.page_id + "_L"):
            raise SchemaError("parent_line_id must belong to page_id")
        expected = make_crop_id(self.parent_line_id, self.crop_type.value, self.crop_index)
        if self.crop_id != expected:
            raise SchemaError(f"crop_id must be {expected}")
        _require_str(self.image_path, "image_path")
        if not isinstance(self.bbox, BBox) or not isinstance(self.parent_bbox, BBox):
            raise SchemaError("bbox and parent_bbox must be BBox values")
        assert_crop_within_parent(self.bbox, self.parent_bbox)
        if self.crop_type is CropType.MULTI_LINE and not self.multi_line_enabled:
            raise SchemaError("multi_line crops require multi_line_enabled")
        if self.crop_type is not CropType.MULTI_LINE and self.multi_line_enabled:
            raise SchemaError("multi_line_enabled is only valid for multi_line crops")
        if self.random_seed is not None:
            _require_int(self.random_seed, "random_seed", minimum=0)
        _optional_sha(self.sha256, "sha256")
        if not isinstance(self.multi_line_enabled, bool):
            raise SchemaError("multi_line_enabled must be a boolean")
        _parse_enum(StageStatus, self.status, "status")
        _optional_str(self.error, "error")

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["x_start"] = self.bbox.x0
        data["y_start"] = self.bbox.y0
        data["x_end"] = self.bbox.x1
        data["y_end"] = self.bbox.y1
        return data

    @classmethod
    def from_dict(cls, data: object) -> CropRecord:
        raw = _require_mapping(data, "CropRecord")
        allowed = {item.name for item in fields(cls)} | {"x_start", "y_start", "x_end", "y_end"}
        _reject_unknown(raw, allowed, "CropRecord")
        _require_fields(
            raw,
            {
                "crop_id",
                "document_id",
                "page_id",
                "page_number",
                "region_id",
                "parent_line_id",
                "crop_type",
                "crop_index",
                "image_path",
                "bbox",
                "parent_bbox",
            },
            "CropRecord",
        )
        bbox = BBox.from_list(raw["bbox"])
        for name, expected in (
            ("x_start", bbox.x0),
            ("y_start", bbox.y0),
            ("x_end", bbox.x1),
            ("y_end", bbox.y1),
        ):
            if name in raw and raw[name] != expected:
                raise SchemaError(f"{name} does not match bbox")
        multi_line_enabled = raw.get("multi_line_enabled", False)
        if not isinstance(multi_line_enabled, bool):
            raise SchemaError("multi_line_enabled must be a boolean")
        record = cls(
            crop_id=str(raw["crop_id"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            page_number=raw["page_number"],
            region_id=str(raw["region_id"]),
            parent_line_id=str(raw["parent_line_id"]),
            crop_type=_parse_enum(CropType, raw["crop_type"], "crop_type"),  # type: ignore[arg-type]
            crop_index=raw["crop_index"],
            image_path=str(raw["image_path"]),
            bbox=bbox,
            parent_bbox=BBox.from_list(raw["parent_bbox"]),
            random_seed=raw.get("random_seed"),
            sha256=raw.get("sha256"),
            multi_line_enabled=multi_line_enabled,
            status=_parse_enum(StageStatus, raw.get("status", StageStatus.PENDING), "status"),  # type: ignore[arg-type]
            error=raw.get("error"),
        )
        record.validate()
        return record


@dataclass
class OCRPrediction:
    """One backend's candidate transcription of one image.

    ``confidence`` stays null when the backend does not expose one.
    This object is never a gold label.
    """

    backend: str
    backend_version: str
    text: str
    raw_text: str
    confidence: float | None = None
    token_confidences: list[float] | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_str(self.backend, "backend")
        _require_str(self.backend_version, "backend_version")
        if not isinstance(self.text, str) or not isinstance(self.raw_text, str):
            raise SchemaError("text and raw_text must be strings")
        _optional_finite(self.confidence, "confidence", low=0.0, high=1.0)
        _token_confidences(self.token_confidences)
        if self.latency_ms is not None:
            _optional_finite(self.latency_ms, "latency_ms", low=0.0, high=3_600_000.0)
        _optional_str(self.error, "error")
        _metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {item.name: to_jsonable(getattr(self, item.name)) for item in fields(self)}


@dataclass
class PredictionRecord(Record):
    """Stored OCR candidate for one crop and one backend."""

    prediction_id: str
    crop_id: str
    document_id: str
    page_id: str
    parent_line_id: str
    backend: str
    backend_version: str
    text: str
    raw_text: str
    image_sha256: str | None = None
    confidence: float | None = None
    token_confidences: list[float] | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: StageStatus = StageStatus.COMPLETE

    def validate(self) -> None:
        validate_crop_id(self.crop_id)
        validate_document_id(self.document_id)
        validate_page_id(self.page_id)
        validate_line_id(self.parent_line_id)
        if document_id_from_identifier(self.crop_id) != self.document_id:
            raise SchemaError("crop_id belongs to a different document")
        if not self.prediction_id.endswith(f"_OCR_{self.backend}"):
            raise SchemaError("prediction_id must end with the backend name")
        if not self.prediction_id.startswith(self.crop_id):
            raise SchemaError("prediction_id must start with crop_id")
        OCRPrediction(
            backend=self.backend,
            backend_version=self.backend_version,
            text=self.text,
            raw_text=self.raw_text,
            confidence=self.confidence,
            token_confidences=self.token_confidences,
            latency_ms=self.latency_ms,
            error=self.error,
            metadata=self.metadata,
        ).validate()
        _optional_sha(self.image_sha256, "image_sha256")
        _parse_enum(StageStatus, self.status, "status")

    @classmethod
    def from_dict(cls, data: object) -> PredictionRecord:
        raw = _require_mapping(data, "PredictionRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "PredictionRecord")
        _require_fields(
            raw,
            {
                "prediction_id",
                "crop_id",
                "document_id",
                "page_id",
                "parent_line_id",
                "backend",
                "backend_version",
                "text",
                "raw_text",
            },
            "PredictionRecord",
        )
        record = cls(
            prediction_id=str(raw["prediction_id"]),
            crop_id=str(raw["crop_id"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            parent_line_id=str(raw["parent_line_id"]),
            backend=str(raw["backend"]),
            backend_version=str(raw["backend_version"]),
            text=str(raw["text"]),
            raw_text=str(raw["raw_text"]),
            image_sha256=raw.get("image_sha256"),
            confidence=raw.get("confidence"),
            token_confidences=raw.get("token_confidences"),
            latency_ms=raw.get("latency_ms"),
            error=raw.get("error"),
            metadata=raw.get("metadata") or {},
            status=_parse_enum(StageStatus, raw.get("status", StageStatus.COMPLETE), "status"),  # type: ignore[arg-type]
        )
        record.validate()
        return record


@dataclass
class CandidateRecord(Record):
    """Agreement and ranking for one crop.

    ``consensus_text`` is the text proposed to a reviewer. It is not gold.
    ``confidence_score`` ranks review. It is not a probability of correctness.
    """

    candidate_id: str
    crop_id: str
    document_id: str
    page_id: str
    parent_line_id: str
    texts_by_backend: dict[str, str | None]
    exact_agreement_count: int
    backends_present: int
    pairwise_agreement: dict[str, bool]
    normalized_agreement: dict[str, bool]
    consensus_text: str | None
    disagreement_type: DisagreementType | None
    confidence_score: float | None
    risk_score: float | None
    reasons: list[str]
    verification_status: VerificationStatus = VerificationStatus.CANDIDATE
    tier: QualityTier = QualityTier.TIER_2_CANDIDATE

    def validate(self) -> None:
        validate_crop_id(self.crop_id)
        if self.candidate_id != f"{self.crop_id}_CAND":
            raise SchemaError("candidate_id must be the crop ID plus _CAND")
        if document_id_from_identifier(self.crop_id) != self.document_id:
            raise SchemaError("crop_id belongs to a different document")
        validate_page_id(self.page_id)
        validate_line_id(self.parent_line_id)
        if not isinstance(self.texts_by_backend, dict):
            raise SchemaError("texts_by_backend must be an object")
        _require_int(self.exact_agreement_count, "exact_agreement_count", minimum=0)
        _require_int(self.backends_present, "backends_present", minimum=0)
        if self.exact_agreement_count > self.backends_present:
            raise SchemaError("exact_agreement_count cannot exceed backends_present")
        _bool_map(self.pairwise_agreement, "pairwise_agreement")
        _bool_map(self.normalized_agreement, "normalized_agreement")
        _optional_str(self.consensus_text, "consensus_text")
        if self.disagreement_type is not None:
            _parse_enum(DisagreementType, self.disagreement_type, "disagreement_type")
        _optional_finite(self.confidence_score, "confidence_score", low=0.0, high=1.0)
        _optional_finite(self.risk_score, "risk_score", low=0.0, high=1.0)
        _reason_list(self.reasons)
        if self.verification_status not in CANDIDATE_VERIFICATION_STATUSES:
            raise SchemaError(
                "candidate records stay candidate or auto_approved_candidate; "
                "human decisions belong on verified and gold records"
            )
        expected_tier = (
            QualityTier.TIER_3_HIGH_CONFIDENCE_CANDIDATE
            if self.verification_status is VerificationStatus.AUTO_APPROVED_CANDIDATE
            else QualityTier.TIER_2_CANDIDATE
        )
        if self.tier is not expected_tier:
            raise SchemaError(f"tier must be {expected_tier.value} for this verification_status")

    @classmethod
    def from_dict(cls, data: object) -> CandidateRecord:
        raw = _require_mapping(data, "CandidateRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "CandidateRecord")
        _require_fields(
            raw,
            {
                "candidate_id",
                "crop_id",
                "document_id",
                "page_id",
                "parent_line_id",
                "texts_by_backend",
                "exact_agreement_count",
                "backends_present",
                "pairwise_agreement",
                "normalized_agreement",
                "consensus_text",
                "disagreement_type",
                "confidence_score",
                "risk_score",
                "reasons",
            },
            "CandidateRecord",
        )
        disagreement = raw.get("disagreement_type")
        record = cls(
            candidate_id=str(raw["candidate_id"]),
            crop_id=str(raw["crop_id"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            parent_line_id=str(raw["parent_line_id"]),
            texts_by_backend=_string_map(
                raw["texts_by_backend"], "texts_by_backend", allow_null_values=True
            ),
            exact_agreement_count=raw["exact_agreement_count"],
            backends_present=raw["backends_present"],
            pairwise_agreement=_bool_map(raw["pairwise_agreement"], "pairwise_agreement"),
            normalized_agreement=_bool_map(raw["normalized_agreement"], "normalized_agreement"),
            consensus_text=raw.get("consensus_text"),
            disagreement_type=(
                None
                if disagreement is None
                else _parse_enum(DisagreementType, disagreement, "disagreement_type")  # type: ignore[arg-type]
            ),
            confidence_score=raw.get("confidence_score"),
            risk_score=raw.get("risk_score"),
            reasons=list(raw["reasons"]),
            verification_status=_parse_enum(  # type: ignore[arg-type]
                VerificationStatus,
                raw.get("verification_status", VerificationStatus.CANDIDATE),
                "verification_status",
            ),
            tier=_parse_enum(  # type: ignore[arg-type]
                QualityTier,
                raw.get("tier", QualityTier.TIER_2_CANDIDATE),
                "tier",
            ),
        )
        record.validate()
        return record


@dataclass
class ReviewRecord(Record):
    """One human review task. The reviewer may type text that no OCR proposed."""

    review_id: str
    crop_id: str
    candidate_id: str
    document_id: str
    page_id: str
    parent_line_id: str
    priority_rank: int
    risk_score: float | None
    audit_sample: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    decision: ReviewDecision | None = None
    final_text: str | None = None
    selected_candidate_text: str | None = None
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    notes: str | None = None
    status: StageStatus = StageStatus.PENDING

    def validate(self) -> None:
        validate_crop_id(self.crop_id)
        if self.review_id != f"{self.crop_id}_REV":
            raise SchemaError("review_id must be the crop ID plus _REV")
        if self.candidate_id != f"{self.crop_id}_CAND":
            raise SchemaError("candidate_id must be the crop ID plus _CAND")
        if document_id_from_identifier(self.crop_id) != self.document_id:
            raise SchemaError("crop_id belongs to a different document")
        validate_page_id(self.page_id)
        validate_line_id(self.parent_line_id)
        _require_int(self.priority_rank, "priority_rank", minimum=1)
        _optional_finite(self.risk_score, "risk_score", low=0.0, high=1.0)
        if not isinstance(self.audit_sample, bool):
            raise SchemaError("audit_sample must be a boolean")
        self.diagnostics = _metadata(self.diagnostics)
        if self.decision is not None:
            _parse_enum(ReviewDecision, self.decision, "decision")
        _optional_str(self.final_text, "final_text")
        _optional_str(self.selected_candidate_text, "selected_candidate_text")
        if self.final_text == "":
            raise SchemaError("final_text must be null or non-empty")
        _optional_str(self.reviewer_id, "reviewer_id")
        _optional_timestamp(self.reviewed_at, "reviewed_at")
        _optional_str(self.notes, "notes")
        _parse_enum(StageStatus, self.status, "status")
        if self.decision is ReviewDecision.ACCEPT_CANDIDATE:
            if not self.final_text or self.final_text != self.selected_candidate_text:
                raise SchemaError(
                    "accept_candidate requires final_text equal to the selected candidate"
                )
        if self.decision is ReviewDecision.CORRECT_TEXT:
            if not self.final_text:
                raise SchemaError("correct_text requires final_text typed by the reviewer")
        if self.decision in {
            ReviewDecision.REJECT_CROP,
            ReviewDecision.BAD_BOUNDARY,
            ReviewDecision.BAD_IMAGE,
            ReviewDecision.NON_TEXT,
            ReviewDecision.UNCERTAIN,
        }:
            if self.final_text is not None:
                raise SchemaError(f"{self.decision.value} does not store final_text")
        if self.decision is not None and self.reviewed_at is None:
            raise SchemaError("a recorded decision requires reviewed_at")

    @classmethod
    def from_dict(cls, data: object) -> ReviewRecord:
        raw = _require_mapping(data, "ReviewRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "ReviewRecord")
        _require_fields(
            raw,
            {
                "review_id",
                "crop_id",
                "candidate_id",
                "document_id",
                "page_id",
                "parent_line_id",
                "priority_rank",
                "risk_score",
                "audit_sample",
            },
            "ReviewRecord",
        )
        decision = raw.get("decision")
        record = cls(
            review_id=str(raw["review_id"]),
            crop_id=str(raw["crop_id"]),
            candidate_id=str(raw["candidate_id"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            parent_line_id=str(raw["parent_line_id"]),
            priority_rank=raw["priority_rank"],
            risk_score=raw.get("risk_score"),
            audit_sample=raw["audit_sample"],
            diagnostics=raw.get("diagnostics") or {},
            decision=(
                None
                if decision is None
                else _parse_enum(ReviewDecision, decision, "decision")  # type: ignore[arg-type]
            ),
            final_text=raw.get("final_text"),
            selected_candidate_text=raw.get("selected_candidate_text"),
            reviewer_id=raw.get("reviewer_id"),
            reviewed_at=raw.get("reviewed_at"),
            notes=raw.get("notes"),
            status=_parse_enum(StageStatus, raw.get("status", StageStatus.PENDING), "status"),  # type: ignore[arg-type]
        )
        record.validate()
        return record


@dataclass
class VerifiedRecord(Record):
    """A completed human decision. Rejected rows stay here and stay out of gold."""

    verified_id: str
    crop_id: str
    document_id: str
    page_id: str
    parent_line_id: str
    verification_status: VerificationStatus
    final_text: str | None
    review_decision: ReviewDecision
    original_candidate_texts: dict[str, str | None]
    source_sha256: str
    verification_timestamp: str
    reviewer_id: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tier: QualityTier = QualityTier.TIER_4_HUMAN_REVIEWED

    def validate(self) -> None:
        validate_crop_id(self.crop_id)
        if self.verified_id != f"{self.crop_id}_VER":
            raise SchemaError("verified_id must be the crop ID plus _VER")
        if document_id_from_identifier(self.crop_id) != self.document_id:
            raise SchemaError("crop_id belongs to a different document")
        validate_page_id(self.page_id)
        validate_line_id(self.parent_line_id)
        if self.verification_status not in {
            VerificationStatus.HUMAN_VERIFIED,
            VerificationStatus.HUMAN_CORRECTED,
            VerificationStatus.REJECTED,
        }:
            raise SchemaError("verified records require a human decision status")
        if self.tier is not QualityTier.TIER_4_HUMAN_REVIEWED:
            raise SchemaError("verified records stay at tier 4 until the gold builder copies them")
        _parse_enum(ReviewDecision, self.review_decision, "review_decision")
        _string_map(
            self.original_candidate_texts,
            "original_candidate_texts",
            allow_null_values=True,
        )
        _require_sha(self.source_sha256, "source_sha256")
        _require_timestamp(self.verification_timestamp, "verification_timestamp")
        _optional_str(self.reviewer_id, "reviewer_id")
        _optional_str(self.notes, "notes")
        self.metadata = _metadata(self.metadata)
        if admits_to_gold(self.verification_status):
            stored = self.final_text is not None and is_storage_form(self.final_text)
            if not stored or not self.final_text:
                raise SchemaError("accepted text must be non-empty storage-form Unicode")
            if has_replacement_character(self.final_text):
                raise SchemaError("accepted text contains U+FFFD")
            if not decision_creates_gold(self.review_decision):
                raise SchemaError("gold-eligible status requires accept_candidate or correct_text")
        else:
            if self.final_text is not None:
                raise SchemaError("rejected verification does not store final_text")
        if self.verification_status is VerificationStatus.HUMAN_VERIFIED:
            if self.review_decision is not ReviewDecision.ACCEPT_CANDIDATE:
                raise SchemaError("human_verified requires accept_candidate")
        if self.verification_status is VerificationStatus.HUMAN_CORRECTED:
            if self.review_decision is not ReviewDecision.CORRECT_TEXT:
                raise SchemaError("human_corrected requires correct_text")
        if self.verification_status is VerificationStatus.REJECTED:
            if self.review_decision not in {
                ReviewDecision.REJECT_CROP,
                ReviewDecision.BAD_BOUNDARY,
                ReviewDecision.BAD_IMAGE,
                ReviewDecision.NON_TEXT,
                ReviewDecision.UNCERTAIN,
            }:
                raise SchemaError("rejected status requires a non-gold review decision")

    @classmethod
    def from_dict(cls, data: object) -> VerifiedRecord:
        raw = _require_mapping(data, "VerifiedRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "VerifiedRecord")
        _require_fields(
            raw,
            {
                "verified_id",
                "crop_id",
                "document_id",
                "page_id",
                "parent_line_id",
                "verification_status",
                "final_text",
                "review_decision",
                "original_candidate_texts",
                "source_sha256",
                "verification_timestamp",
            },
            "VerifiedRecord",
        )
        record = cls(
            verified_id=str(raw["verified_id"]),
            crop_id=str(raw["crop_id"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            parent_line_id=str(raw["parent_line_id"]),
            verification_status=_parse_enum(  # type: ignore[arg-type]
                VerificationStatus, raw["verification_status"], "verification_status"
            ),
            final_text=raw.get("final_text"),
            review_decision=_parse_enum(ReviewDecision, raw["review_decision"], "review_decision"),  # type: ignore[arg-type]
            original_candidate_texts=_string_map(
                raw["original_candidate_texts"],
                "original_candidate_texts",
                allow_null_values=True,
            ),
            source_sha256=str(raw["source_sha256"]),
            verification_timestamp=str(raw["verification_timestamp"]),
            reviewer_id=raw.get("reviewer_id"),
            notes=raw.get("notes"),
            metadata=raw.get("metadata") or {},
            tier=_parse_enum(  # type: ignore[arg-type]
                QualityTier,
                raw.get("tier", QualityTier.TIER_4_HUMAN_REVIEWED),
                "tier",
            ),
        )
        record.validate()
        return record


@dataclass
class GoldRecord(Record):
    """One trainable sample. The label was confirmed by a person looking at the crop."""

    sample_id: str
    image_path: str
    gold_text: str
    document_id: str
    page_id: str
    page_number: int
    region_id: str
    parent_line_id: str
    crop_id: str
    crop_type: CropType
    bbox: BBox
    verification_status: VerificationStatus
    source_sha256: str
    original_candidate_texts: dict[str, str | None]
    verification_timestamp: str
    reviewer_id: str | None = None
    split: SplitName | None = None
    consensus_text: str | None = None
    tier: QualityTier = QualityTier.TIER_5_GOLD

    def validate(self) -> None:
        validate_sample_id(self.sample_id)
        validate_crop_id(self.crop_id)
        if self.sample_id != f"SAMPLE_{self.crop_id}":
            raise SchemaError("sample_id must be SAMPLE_ plus crop_id")
        _ids_match_page(self.document_id, self.page_id, self.page_number)
        validate_region_id(self.region_id)
        validate_line_id(self.parent_line_id)
        if document_id_from_identifier(self.crop_id) != self.document_id:
            raise SchemaError("crop_id belongs to a different document")
        if not self.region_id.startswith(self.page_id + "_R"):
            raise SchemaError("region_id must belong to page_id")
        if not self.parent_line_id.startswith(self.page_id + "_L"):
            raise SchemaError("parent_line_id must belong to page_id")
        expected_crop_prefix = f"{self.parent_line_id}_C_{self.crop_type.value}_"
        if not self.crop_id.startswith(expected_crop_prefix):
            raise SchemaError("crop_id must match parent_line_id and crop_type")
        _require_str(self.image_path, "image_path")
        if not isinstance(self.gold_text, str) or not self.gold_text.strip():
            raise SchemaError("gold_text must be non-empty")
        if not is_storage_form(self.gold_text):
            raise SchemaError(
                "gold_text must already be NFC with surrounding whitespace removed; "
                "the record will not rewrite it"
            )
        if has_replacement_character(self.gold_text):
            raise SchemaError("gold_text contains U+FFFD")
        if not isinstance(self.bbox, BBox):
            raise SchemaError("bbox must be a BBox")
        if not admits_to_gold(self.verification_status):
            raise SchemaError(
                "gold verification_status must be human_verified or human_corrected"
            )
        if self.tier is not QualityTier.TIER_5_GOLD:
            raise SchemaError("gold records must be tier_5_gold")
        _require_sha(self.source_sha256, "source_sha256")
        _string_map(
            self.original_candidate_texts,
            "original_candidate_texts",
            allow_null_values=True,
        )
        _require_timestamp(self.verification_timestamp, "verification_timestamp")
        _optional_str(self.reviewer_id, "reviewer_id")
        if self.split is not None:
            _parse_enum(SplitName, self.split, "split")
        _optional_str(self.consensus_text, "consensus_text")

    @classmethod
    def from_dict(cls, data: object) -> GoldRecord:
        raw = _require_mapping(data, "GoldRecord")
        allowed = {item.name for item in fields(cls)}
        _reject_unknown(raw, allowed, "GoldRecord")
        _require_fields(
            raw,
            {
                "sample_id",
                "image_path",
                "gold_text",
                "document_id",
                "page_id",
                "page_number",
                "region_id",
                "parent_line_id",
                "crop_id",
                "crop_type",
                "bbox",
                "verification_status",
                "source_sha256",
                "original_candidate_texts",
                "verification_timestamp",
            },
            "GoldRecord",
        )
        split = raw.get("split")
        record = cls(
            sample_id=str(raw["sample_id"]),
            image_path=str(raw["image_path"]),
            gold_text=str(raw["gold_text"]),
            document_id=str(raw["document_id"]),
            page_id=str(raw["page_id"]),
            page_number=raw["page_number"],
            region_id=str(raw["region_id"]),
            parent_line_id=str(raw["parent_line_id"]),
            crop_id=str(raw["crop_id"]),
            crop_type=_parse_enum(CropType, raw["crop_type"], "crop_type"),  # type: ignore[arg-type]
            bbox=BBox.from_list(raw["bbox"]),
            verification_status=_parse_enum(  # type: ignore[arg-type]
                VerificationStatus, raw["verification_status"], "verification_status"
            ),
            source_sha256=str(raw["source_sha256"]),
            original_candidate_texts=_string_map(
                raw["original_candidate_texts"],
                "original_candidate_texts",
                allow_null_values=True,
            ),
            verification_timestamp=str(raw["verification_timestamp"]),
            reviewer_id=raw.get("reviewer_id"),
            split=None if split is None else _parse_enum(SplitName, split, "split"),  # type: ignore[arg-type]
            consensus_text=raw.get("consensus_text"),
            tier=_parse_enum(QualityTier, raw.get("tier", QualityTier.TIER_5_GOLD), "tier"),  # type: ignore[arg-type]
        )
        record.validate()
        return record


def assert_partial_label_matches_crop(
    crop_type: CropType,
    gold_text: str,
    full_line_gold_text: str | None,
) -> None:
    """Refuse a partial crop whose label is a copy of the full parent line.

    When the full line has no gold label yet, this check does not guess the
    visible substring. The reviewer still has to confirm the crop text.
    """
    if crop_type is CropType.FULL or crop_type is CropType.MULTI_LINE:
        return
    if full_line_gold_text is None:
        return
    if gold_text == full_line_gold_text:
        raise SchemaError(
            "partial crop gold_text copies the full parent line; "
            "the label must be the text visible inside this crop"
        )


__all__ = [
    "ALLOWED_DPI",
    "BBox",
    "CANDIDATE_VERIFICATION_STATUSES",
    "CandidateRecord",
    "CropRecord",
    "CropType",
    "DisagreementType",
    "DocumentRecord",
    "GOLD_VERIFICATION_STATUSES",
    "GoldRecord",
    "OCRPrediction",
    "PDF_TEXT_LAYER_ROLE",
    "PageRecord",
    "PredictionRecord",
    "QualityTier",
    "RegionRecord",
    "RegionType",
    "ReviewDecision",
    "ReviewRecord",
    "SCORE_IS_NOT_A_PROBABILITY",
    "ScoreReason",
    "SplitName",
    "StageStatus",
    "VerificationStatus",
    "VerifiedRecord",
    "admits_to_gold",
    "assert_bbox_within_page",
    "assert_crop_within_parent",
    "assert_partial_label_matches_crop",
    "decision_creates_gold",
]
