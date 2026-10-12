"""YAML configuration.

Paths, thresholds, seeds, and model locations live in ``configs/``.
Environment variables may point at model weights that are already installed.
The loader refuses a configuration that enables network access.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from indic_ocr_pipeline.errors import ConfigError
from indic_ocr_pipeline.schemas import ALLOWED_DPI, GOLD_VERIFICATION_STATUSES, PDF_TEXT_LAYER_ROLE

KNOWN_BACKENDS = ("tesseract", "indicocr", "chitrapathak")
_WEIGHT_KEYS = (
    "agreement",
    "tesseract_confidence",
    "image_quality",
    "crop_geometry",
    "unicode_sanity",
)
_QUALITY_GATE_KEYS = (
    "fail_on_document_leakage",
    "fail_on_parent_line_leakage",
    "fail_on_invalid_unicode",
    "fail_on_missing_images",
    "fail_on_missing_labels",
    "fail_on_duplicate_sample_ids",
    "fail_on_conflicting_gold_labels",
    "fail_on_unverified_candidates",
    "fail_on_rejected_samples",
    "fail_on_missing_provenance",
)


@dataclass(frozen=True)
class SafetySettings:
    overwrite_existing_outputs: bool
    allow_network: bool


@dataclass(frozen=True)
class RenderSettings:
    dpi: int
    allowed_dpi: tuple[int, ...]
    overwrite: bool
    image_format: str
    extract_pdf_text_layer: bool
    pdf_text_layer_role: str


@dataclass(frozen=True)
class DetectionSettings:
    method: str
    min_line_height_px: int
    max_line_height_px: int
    min_line_width_px: int
    binarization: str
    adaptive_block_size: int
    adaptive_c: int
    morph_kernel_width: int
    morph_kernel_height: int
    uncertain_region_type: str
    intra_line_gap_px: int
    block_gap_y_px: int
    max_block_gap_px: int
    spacing_split_ratio: float
    column_gap_px: int
    min_region_width_px: int
    min_region_height_px: int
    min_region_ink_px: int
    padding_px: int
    row_ink_tolerance: int
    absolute_ink_gray: int
    ink_max_gray: int
    rule_min_length_px: int
    rule_min_length_fraction: float
    table_min_h_rules: int
    table_min_v_rules: int
    table_max_rule_gap_px: int
    frame_page_fraction: float
    border_margin_px: int
    line_split_gap_px: int
    line_padding_px: int
    min_physical_line_width_px: int
    min_physical_line_height_px: int
    min_physical_line_ink_px: int
    rule_band_max_height_px: int
    rule_band_min_width_fraction: float
    rule_band_min_row_fill: float
    rule_band_busy_fill: float
    rule_band_max_busy_rows: int
    rule_suppress_max_height_px: int
    rule_suppress_row_fill: float
    line_row_empty_fraction: float
    matra_band_max_height_px: int
    matra_merge_gap_px: int
    vertical_rule_min_region_px: int
    vertical_rule_min_run_px: int
    vertical_rule_min_fraction: float
    tall_line_px: int
    line_region_index_base: int
    debug_images: bool


@dataclass(frozen=True)
class FractionRange:
    min_width_fraction: float
    max_width_fraction: float


@dataclass(frozen=True)
class CroppingSettings:
    full_line: bool
    partial: bool
    multi_line: bool
    random_seed: int
    max_partial_per_line: int
    left: FractionRange
    right: FractionRange
    middle: FractionRange
    random: FractionRange


@dataclass(frozen=True)
class BackendSettings:
    name: str
    cmd: str | None
    lang: str | None
    psm: int | None
    model_path: str | None
    device: str
    batch_size: int
    dtype: str
    max_image_size: int
    concurrency: int


@dataclass(frozen=True)
class OCRSettings:
    enabled_backends: tuple[str, ...]
    device: str
    dtype: str
    batch_size: int
    max_image_size: int
    concurrency: int
    tesseract: BackendSettings
    indicocr: BackendSettings
    chitrapathak: BackendSettings


@dataclass(frozen=True)
class ScoringSettings:
    weights: dict[str, float]
    high_confidence_threshold: float
    score_meaning: str


@dataclass(frozen=True)
class ReviewSettings:
    high_confidence_random_audit: float
    audit_min_samples: int
    audit_max_samples: int
    random_seed: int
    reviewer_id: str
    calibration_batch_size: int
    calibration_tier_quota: dict[str, int | None]
    host: str
    port: int


@dataclass(frozen=True)
class SplitSettings:
    train: float
    val: float
    test: float
    seed: int


@dataclass(frozen=True)
class LabelLengthSettings:
    max_characters: int
    short_below: int
    short_fraction: float
    long_min_characters: int


@dataclass(frozen=True)
class DatasetSettings:
    split: SplitSettings
    export_formats: tuple[str, ...]
    gold_statuses: tuple[str, ...]
    department_share_warning: float
    document_share_warning: float
    partial_crop_ratio_low: float
    partial_crop_ratio_high: float
    quality_gate: dict[str, bool]
    label_length: LabelLengthSettings


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    documents_dir: Path
    rendered_pages_dir: Path
    regions_dir: Path
    crops_full_dir: Path
    crops_partial_dir: Path
    crops_review_dir: Path
    predictions_dir: Path
    scoring_dir: Path
    review_dir: Path
    verified_accepted_dir: Path
    verified_corrected_dir: Path
    verified_rejected_dir: Path
    gold_dir: Path
    splits_dir: Path
    manifests_dir: Path
    reports_dir: Path
    safety: SafetySettings
    logging_level: str
    random_seed: int
    render: RenderSettings
    detection: DetectionSettings
    cropping: CroppingSettings
    ocr: OCRSettings
    scoring: ScoringSettings
    review: ReviewSettings
    dataset: DatasetSettings


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents until ``pyproject.toml`` and ``configs/pipeline.yaml`` exist."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        marker = (parent / "pyproject.toml").is_file()
        pipeline = (parent / "configs" / "pipeline.yaml").is_file()
        if marker and pipeline:
            return parent
    raise ConfigError("could not find the repository root from the working directory")


def load_settings(
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate every config file."""
    root = (repo_root or find_repo_root()).resolve()
    config_dir = root / "configs"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    rendering = _load_yaml(config_dir / "rendering.yaml")
    detection = _load_yaml(config_dir / "detection.yaml")
    cropping = _load_yaml(config_dir / "cropping.yaml")
    ocr = _load_yaml(config_dir / "ocr.yaml")
    scoring = _load_yaml(config_dir / "scoring.yaml")
    review = _load_yaml(config_dir / "review.yaml")
    dataset = _load_yaml(config_dir / "dataset.yaml")
    env = os.environ if environ is None else environ
    _apply_environment(ocr, env)
    return _build_settings(
        root=root,
        pipeline=pipeline,
        rendering=rendering,
        detection=detection,
        cropping=cropping,
        ocr=ocr,
        scoring=scoring,
        review=review,
        dataset=dataset,
    )


def _apply_environment(ocr: dict[str, Any], environ: Mapping[str, str]) -> None:
    indicocr_path = environ.get("INDICOCR_MODEL_PATH", "").strip()
    chitrapathak_path = environ.get("CHITRAPATHAK_MODEL_PATH", "").strip()
    tesseract_cmd = environ.get("TESSERACT_CMD", "").strip()
    device = (
        environ.get("INDIC_OCR_DEVICE", "").strip()
        or environ.get("MARATHI_OCR_DEVICE", "").strip()
    )
    if indicocr_path:
        _mapping(ocr.get("indicocr"), "ocr.indicocr")["model_path"] = indicocr_path
    if chitrapathak_path:
        _mapping(ocr.get("chitrapathak"), "ocr.chitrapathak")["model_path"] = chitrapathak_path
    if tesseract_cmd:
        _mapping(ocr.get("tesseract"), "ocr.tesseract")["cmd"] = tesseract_cmd
    if device:
        ocr["device"] = device
        _mapping(ocr.get("indicocr"), "ocr.indicocr")["device"] = device
        _mapping(ocr.get("chitrapathak"), "ocr.chitrapathak")["device"] = device


def _build_settings(
    *,
    root: Path,
    pipeline: dict[str, Any],
    rendering: dict[str, Any],
    detection: dict[str, Any],
    cropping: dict[str, Any],
    ocr: dict[str, Any],
    scoring: dict[str, Any],
    review: dict[str, Any],
    dataset: dict[str, Any],
) -> Settings:
    _reject_unknown(
        pipeline,
        {
            "source",
            "paths",
            "safety",
            "logging",
            "random_seed",
            "execution",
            "runtime",
            "calibration",
            "training",
            "evaluation",
            "gpu",
            "seed",
            "ocr",
        },
        "pipeline.yaml",
    )
    source = _mapping(pipeline.get("source"), "pipeline.source")
    paths = _mapping(pipeline.get("paths"), "pipeline.paths")
    safety = _mapping(pipeline.get("safety"), "pipeline.safety")
    logging_block = _mapping(pipeline.get("logging"), "pipeline.logging")
    _reject_unknown(source, {"documents_dir"}, "pipeline.source")
    _reject_unknown(
        paths,
        {
            "rendered_pages_dir",
            "regions_dir",
            "crops_full_dir",
            "crops_partial_dir",
            "crops_review_dir",
            "predictions_dir",
            "scoring_dir",
            "review_dir",
            "verified_accepted_dir",
            "verified_corrected_dir",
            "verified_rejected_dir",
            "gold_dir",
            "splits_dir",
            "manifests_dir",
            "reports_dir",
        },
        "pipeline.paths",
    )
    _reject_unknown(safety, {"overwrite_existing_outputs", "allow_network"}, "pipeline.safety")
    _reject_unknown(logging_block, {"level"}, "pipeline.logging")
    allow_network = _bool(safety.get("allow_network"), "pipeline.safety.allow_network")
    if allow_network:
        raise ConfigError(
            "pipeline.safety.allow_network must be false; this pipeline stays offline"
        )
    logging_level = _str(logging_block.get("level"), "pipeline.logging.level")
    if logging_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("pipeline.logging.level must be DEBUG, INFO, WARNING, or ERROR")

    def resolve(key: str, block: Mapping[str, Any] = paths) -> Path:
        return _resolve(root, _str(block.get(key), f"path.{key}"))

    return Settings(
        repo_root=root,
        documents_dir=_documents_dir(root, source),
        rendered_pages_dir=resolve("rendered_pages_dir"),
        regions_dir=resolve("regions_dir"),
        crops_full_dir=resolve("crops_full_dir"),
        crops_partial_dir=resolve("crops_partial_dir"),
        crops_review_dir=resolve("crops_review_dir"),
        predictions_dir=resolve("predictions_dir"),
        scoring_dir=resolve("scoring_dir"),
        review_dir=resolve("review_dir"),
        verified_accepted_dir=resolve("verified_accepted_dir"),
        verified_corrected_dir=resolve("verified_corrected_dir"),
        verified_rejected_dir=resolve("verified_rejected_dir"),
        gold_dir=resolve("gold_dir"),
        splits_dir=resolve("splits_dir"),
        manifests_dir=resolve("manifests_dir"),
        reports_dir=resolve("reports_dir"),
        safety=SafetySettings(
            overwrite_existing_outputs=_bool(
                safety.get("overwrite_existing_outputs"),
                "pipeline.safety.overwrite_existing_outputs",
            ),
            allow_network=False,
        ),
        logging_level=logging_level,
        random_seed=_int(pipeline.get("random_seed"), "pipeline.random_seed", minimum=0),
        render=_render_settings(rendering),
        detection=_detection_settings(detection),
        cropping=_cropping_settings(cropping),
        ocr=_ocr_settings(ocr),
        scoring=_scoring_settings(scoring),
        review=_review_settings(review),
        dataset=_dataset_settings(dataset),
    )


def _render_settings(raw: dict[str, Any]) -> RenderSettings:
    _reject_unknown(
        raw,
        {
            "dpi",
            "allowed_dpi",
            "overwrite",
            "image_format",
            "extract_pdf_text_layer",
            "pdf_text_layer_role",
        },
        "rendering.yaml",
    )
    allowed = raw.get("allowed_dpi")
    integers = isinstance(allowed, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in allowed
    )
    if not integers:
        raise ConfigError("rendering.allowed_dpi must be a list of integers")
    if set(allowed) != ALLOWED_DPI:
        raise ConfigError("rendering.allowed_dpi must be exactly 150, 200, 250, 300, and 400")
    dpi = _int(raw.get("dpi"), "rendering.dpi", minimum=1)
    if dpi not in ALLOWED_DPI:
        raise ConfigError("rendering.dpi must be one of the allowed DPI values")
    image_format = _str(raw.get("image_format"), "rendering.image_format")
    if image_format != "png":
        raise ConfigError("rendering.image_format must be png")
    role = _str(raw.get("pdf_text_layer_role"), "rendering.pdf_text_layer_role")
    if role != PDF_TEXT_LAYER_ROLE:
        raise ConfigError(
            "rendering.pdf_text_layer_role cannot treat extracted PDF text as ground truth"
        )
    return RenderSettings(
        dpi=dpi,
        allowed_dpi=tuple(sorted(allowed)),
        overwrite=_bool(raw.get("overwrite"), "rendering.overwrite"),
        image_format=image_format,
        extract_pdf_text_layer=_bool(
            raw.get("extract_pdf_text_layer"), "rendering.extract_pdf_text_layer"
        ),
        pdf_text_layer_role=role,
    )


def _detection_settings(raw: dict[str, Any]) -> DetectionSettings:
    _reject_unknown(
        raw,
        {
            "method",
            "min_line_height_px",
            "max_line_height_px",
            "min_line_width_px",
            "binarization",
            "adaptive_block_size",
            "adaptive_c",
            "morph_kernel_width",
            "morph_kernel_height",
            "uncertain_region_type",
            "intra_line_gap_px",
            "block_gap_y_px",
            "max_block_gap_px",
            "spacing_split_ratio",
            "column_gap_px",
            "min_region_width_px",
            "min_region_height_px",
            "min_region_ink_px",
            "padding_px",
            "row_ink_tolerance",
            "absolute_ink_gray",
            "ink_max_gray",
            "rule_min_length_px",
            "rule_min_length_fraction",
            "table_min_h_rules",
            "table_min_v_rules",
            "table_max_rule_gap_px",
            "frame_page_fraction",
            "border_margin_px",
            "line_split_gap_px",
            "line_padding_px",
            "min_physical_line_width_px",
            "min_physical_line_height_px",
            "min_physical_line_ink_px",
            "rule_band_max_height_px",
            "rule_band_min_width_fraction",
            "rule_band_min_row_fill",
            "rule_band_busy_fill",
            "rule_band_max_busy_rows",
            "rule_suppress_max_height_px",
            "rule_suppress_row_fill",
            "line_row_empty_fraction",
            "matra_band_max_height_px",
            "matra_merge_gap_px",
            "vertical_rule_min_region_px",
            "vertical_rule_min_run_px",
            "vertical_rule_min_fraction",
            "tall_line_px",
            "line_region_index_base",
            "debug_images",
        },
        "detection.yaml",
    )
    minimum_height = _int(raw.get("min_line_height_px"), "detection.min_line_height_px", minimum=1)
    maximum_height = _int(raw.get("max_line_height_px"), "detection.max_line_height_px", minimum=1)
    if minimum_height >= maximum_height:
        raise ConfigError("detection line height range is empty")
    block_size = _int(raw.get("adaptive_block_size"), "detection.adaptive_block_size", minimum=3)
    if block_size % 2 == 0:
        raise ConfigError("detection.adaptive_block_size must be odd")
    uncertain = _str(raw.get("uncertain_region_type"), "detection.uncertain_region_type")
    if uncertain != "unknown":
        raise ConfigError("detection.uncertain_region_type must be unknown")
    binarization = _str(raw.get("binarization"), "detection.binarization")
    if binarization not in {"adaptive", "otsu"}:
        raise ConfigError("detection.binarization must be adaptive or otsu")
    block_gap = _int(raw.get("block_gap_y_px"), "detection.block_gap_y_px", minimum=1)
    max_block_gap = _int(raw.get("max_block_gap_px"), "detection.max_block_gap_px", minimum=1)
    if max_block_gap < block_gap:
        raise ConfigError("detection.max_block_gap_px must be >= block_gap_y_px")
    return DetectionSettings(
        method=_str(raw.get("method"), "detection.method"),
        min_line_height_px=minimum_height,
        max_line_height_px=maximum_height,
        min_line_width_px=_int(
            raw.get("min_line_width_px"), "detection.min_line_width_px", minimum=1
        ),
        binarization=binarization,
        adaptive_block_size=block_size,
        adaptive_c=_int(raw.get("adaptive_c"), "detection.adaptive_c", minimum=0),
        morph_kernel_width=_int(
            raw.get("morph_kernel_width"), "detection.morph_kernel_width", minimum=1
        ),
        morph_kernel_height=_int(
            raw.get("morph_kernel_height"), "detection.morph_kernel_height", minimum=1
        ),
        uncertain_region_type=uncertain,
        intra_line_gap_px=_int(
            raw.get("intra_line_gap_px"), "detection.intra_line_gap_px", minimum=1
        ),
        block_gap_y_px=block_gap,
        max_block_gap_px=max_block_gap,
        spacing_split_ratio=_ratio(
            raw.get("spacing_split_ratio"), "detection.spacing_split_ratio"
        ),
        column_gap_px=_int(raw.get("column_gap_px"), "detection.column_gap_px", minimum=1),
        min_region_width_px=_int(
            raw.get("min_region_width_px"), "detection.min_region_width_px", minimum=1
        ),
        min_region_height_px=_int(
            raw.get("min_region_height_px"), "detection.min_region_height_px", minimum=1
        ),
        min_region_ink_px=_int(
            raw.get("min_region_ink_px"), "detection.min_region_ink_px", minimum=1
        ),
        padding_px=_int(raw.get("padding_px"), "detection.padding_px", minimum=0),
        row_ink_tolerance=_int(
            raw.get("row_ink_tolerance"), "detection.row_ink_tolerance", minimum=0
        ),
        absolute_ink_gray=_int(
            raw.get("absolute_ink_gray"), "detection.absolute_ink_gray", minimum=0
        ),
        ink_max_gray=_int(raw.get("ink_max_gray"), "detection.ink_max_gray", minimum=1),
        rule_min_length_px=_int(
            raw.get("rule_min_length_px"), "detection.rule_min_length_px", minimum=1
        ),
        rule_min_length_fraction=_unit_fraction(
            raw.get("rule_min_length_fraction"), "detection.rule_min_length_fraction"
        ),
        table_min_h_rules=_int(
            raw.get("table_min_h_rules"), "detection.table_min_h_rules", minimum=2
        ),
        table_min_v_rules=_int(
            raw.get("table_min_v_rules"), "detection.table_min_v_rules", minimum=2
        ),
        table_max_rule_gap_px=_int(
            raw.get("table_max_rule_gap_px"), "detection.table_max_rule_gap_px", minimum=1
        ),
        frame_page_fraction=_unit_fraction(
            raw.get("frame_page_fraction"), "detection.frame_page_fraction"
        ),
        border_margin_px=_int(
            raw.get("border_margin_px"), "detection.border_margin_px", minimum=0
        ),
        line_split_gap_px=_int(
            raw.get("line_split_gap_px"), "detection.line_split_gap_px", minimum=1
        ),
        line_padding_px=_int(raw.get("line_padding_px"), "detection.line_padding_px", minimum=0),
        min_physical_line_width_px=_int(
            raw.get("min_physical_line_width_px"),
            "detection.min_physical_line_width_px",
            minimum=1,
        ),
        min_physical_line_height_px=_int(
            raw.get("min_physical_line_height_px"),
            "detection.min_physical_line_height_px",
            minimum=1,
        ),
        min_physical_line_ink_px=_int(
            raw.get("min_physical_line_ink_px"),
            "detection.min_physical_line_ink_px",
            minimum=1,
        ),
        rule_band_max_height_px=_int(
            raw.get("rule_band_max_height_px"),
            "detection.rule_band_max_height_px",
            minimum=1,
        ),
        rule_band_min_width_fraction=_unit_fraction(
            raw.get("rule_band_min_width_fraction"),
            "detection.rule_band_min_width_fraction",
        ),
        rule_band_min_row_fill=_unit_fraction(
            raw.get("rule_band_min_row_fill"),
            "detection.rule_band_min_row_fill",
        ),
        rule_band_busy_fill=_unit_fraction(
            raw.get("rule_band_busy_fill"),
            "detection.rule_band_busy_fill",
        ),
        rule_band_max_busy_rows=_int(
            raw.get("rule_band_max_busy_rows"),
            "detection.rule_band_max_busy_rows",
            minimum=1,
        ),
        rule_suppress_max_height_px=_int(
            raw.get("rule_suppress_max_height_px"),
            "detection.rule_suppress_max_height_px",
            minimum=1,
        ),
        rule_suppress_row_fill=_unit_fraction(
            raw.get("rule_suppress_row_fill"),
            "detection.rule_suppress_row_fill",
        ),
        line_row_empty_fraction=_unit_fraction(
            raw.get("line_row_empty_fraction"),
            "detection.line_row_empty_fraction",
        ),
        matra_band_max_height_px=_int(
            raw.get("matra_band_max_height_px"),
            "detection.matra_band_max_height_px",
            minimum=1,
        ),
        matra_merge_gap_px=_int(
            raw.get("matra_merge_gap_px"),
            "detection.matra_merge_gap_px",
            minimum=0,
        ),
        vertical_rule_min_region_px=_int(
            raw.get("vertical_rule_min_region_px"),
            "detection.vertical_rule_min_region_px",
            minimum=1,
        ),
        vertical_rule_min_run_px=_int(
            raw.get("vertical_rule_min_run_px"),
            "detection.vertical_rule_min_run_px",
            minimum=1,
        ),
        vertical_rule_min_fraction=_unit_fraction(
            raw.get("vertical_rule_min_fraction"),
            "detection.vertical_rule_min_fraction",
        ),
        tall_line_px=_int(raw.get("tall_line_px"), "detection.tall_line_px", minimum=1),
        line_region_index_base=_int(
            raw.get("line_region_index_base"),
            "detection.line_region_index_base",
            minimum=1,
        ),
        debug_images=_bool(raw.get("debug_images"), "detection.debug_images"),
    )


def _cropping_settings(raw: dict[str, Any]) -> CroppingSettings:
    _reject_unknown(
        raw,
        {
            "full_line",
            "partial",
            "multi_line",
            "random_seed",
            "max_partial_per_line",
            "left",
            "right",
            "middle",
            "random",
        },
        "cropping.yaml",
    )
    full_line = _bool(raw.get("full_line"), "cropping.full_line")
    if not full_line:
        raise ConfigError("cropping.full_line must stay enabled")
    return CroppingSettings(
        full_line=True,
        partial=_bool(raw.get("partial"), "cropping.partial"),
        multi_line=_bool(raw.get("multi_line"), "cropping.multi_line"),
        random_seed=_int(raw.get("random_seed"), "cropping.random_seed", minimum=0),
        max_partial_per_line=_int(
            raw.get("max_partial_per_line"), "cropping.max_partial_per_line", minimum=0
        ),
        left=_fraction_range(raw.get("left"), "cropping.left"),
        right=_fraction_range(raw.get("right"), "cropping.right"),
        middle=_fraction_range(raw.get("middle"), "cropping.middle"),
        random=_fraction_range(raw.get("random"), "cropping.random"),
    )


def _ocr_settings(raw: dict[str, Any]) -> OCRSettings:
    _reject_unknown(
        raw,
        {
            "enabled_backends",
            "device",
            "dtype",
            "batch_size",
            "max_image_size",
            "concurrency",
            "tesseract",
            "indicocr",
            "chitrapathak",
        },
        "ocr.yaml",
    )
    enabled = raw.get("enabled_backends")
    names = isinstance(enabled, list) and bool(enabled)
    names = names and all(isinstance(item, str) for item in enabled)
    if not names:
        raise ConfigError("ocr.enabled_backends must be a non-empty list of backend names")
    if len(set(enabled)) != len(enabled):
        raise ConfigError("ocr.enabled_backends contains a duplicate")
    unknown = sorted(set(enabled) - set(KNOWN_BACKENDS))
    if unknown:
        raise ConfigError(f"ocr.enabled_backends has unknown backends: {', '.join(unknown)}")
    return OCRSettings(
        enabled_backends=tuple(enabled),
        device=_str(raw.get("device"), "ocr.device"),
        dtype=_str(raw.get("dtype"), "ocr.dtype"),
        batch_size=_int(raw.get("batch_size"), "ocr.batch_size", minimum=1),
        max_image_size=_int(raw.get("max_image_size"), "ocr.max_image_size", minimum=1),
        concurrency=_int(raw.get("concurrency"), "ocr.concurrency", minimum=1),
        tesseract=_backend_settings(raw.get("tesseract"), "tesseract", command=True),
        indicocr=_backend_settings(raw.get("indicocr"), "indicocr", command=False),
        chitrapathak=_backend_settings(raw.get("chitrapathak"), "chitrapathak", command=False),
    )


def _backend_settings(value: object, name: str, *, command: bool) -> BackendSettings:
    raw = _mapping(value, f"ocr.{name}")
    allowed = {"device", "batch_size", "dtype", "max_image_size", "concurrency"}
    if command:
        allowed |= {"cmd", "lang", "psm"}
    else:
        allowed |= {"model_path"}
    _reject_unknown(raw, allowed, f"ocr.{name}")
    model_path = None if command else _optional_str(raw.get("model_path"), f"ocr.{name}.model_path")
    return BackendSettings(
        name=name,
        cmd=_str(raw.get("cmd"), f"ocr.{name}.cmd") if command else None,
        lang=_str(raw.get("lang"), f"ocr.{name}.lang") if command else None,
        psm=_int(raw.get("psm"), f"ocr.{name}.psm", minimum=0) if command else None,
        model_path=model_path,
        device=_str(raw.get("device"), f"ocr.{name}.device"),
        batch_size=_int(raw.get("batch_size"), f"ocr.{name}.batch_size", minimum=1),
        dtype=_str(raw.get("dtype"), f"ocr.{name}.dtype"),
        max_image_size=_int(raw.get("max_image_size"), f"ocr.{name}.max_image_size", minimum=1),
        concurrency=_int(raw.get("concurrency"), f"ocr.{name}.concurrency", minimum=1),
    )


def _scoring_settings(raw: dict[str, Any]) -> ScoringSettings:
    _reject_unknown(
        raw,
        {"weights", "high_confidence_threshold", "score_meaning"},
        "scoring.yaml",
    )
    weights_raw = _mapping(raw.get("weights"), "scoring.weights")
    _reject_unknown(weights_raw, set(_WEIGHT_KEYS), "scoring.weights")
    weights = {
        key: _probability(weights_raw.get(key), f"scoring.weights.{key}") for key in _WEIGHT_KEYS
    }
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ConfigError("scoring.weights must sum to 1")
    meaning = _str(raw.get("score_meaning"), "scoring.score_meaning")
    if meaning != "ranking_not_probability":
        raise ConfigError("scoring.score_meaning must be ranking_not_probability")
    return ScoringSettings(
        weights=weights,
        high_confidence_threshold=_probability(
            raw.get("high_confidence_threshold"), "scoring.high_confidence_threshold"
        ),
        score_meaning=meaning,
    )


def _review_settings(raw: dict[str, Any]) -> ReviewSettings:
    _reject_unknown(
        raw,
        {
            "high_confidence_random_audit",
            "audit_min_samples",
            "audit_max_samples",
            "random_seed",
            "reviewer_id",
            "calibration_batch_size",
            "calibration_tier_quota",
            "host",
            "port",
        },
        "review.yaml",
    )
    minimum = _int(raw.get("audit_min_samples"), "review.audit_min_samples", minimum=1)
    maximum = _int(raw.get("audit_max_samples"), "review.audit_max_samples", minimum=1)
    if minimum > maximum:
        raise ConfigError("review.audit_min_samples cannot exceed audit_max_samples")
    fraction = _probability(
        raw.get("high_confidence_random_audit"), "review.high_confidence_random_audit"
    )
    if fraction <= 0:
        raise ConfigError("review.high_confidence_random_audit must be greater than 0")
    quota_raw = _mapping(raw.get("calibration_tier_quota"), "review.calibration_tier_quota")
    _reject_unknown(quota_raw, {"P0", "P1", "P2", "P3"}, "review.calibration_tier_quota")
    quota: dict[str, int | None] = {}
    for key in ("P0", "P1", "P2", "P3"):
        value = quota_raw.get(key)
        if value is None:
            quota[key] = None
        else:
            quota[key] = _int(value, f"review.calibration_tier_quota.{key}", minimum=0)
    host = _str(raw.get("host"), "review.host")
    if host not in {"127.0.0.1", "localhost"}:
        raise ConfigError("review.host must be 127.0.0.1 or localhost")
    return ReviewSettings(
        high_confidence_random_audit=fraction,
        audit_min_samples=minimum,
        audit_max_samples=maximum,
        random_seed=_int(raw.get("random_seed"), "review.random_seed", minimum=0),
        reviewer_id=_str(raw.get("reviewer_id"), "review.reviewer_id"),
        calibration_batch_size=_int(
            raw.get("calibration_batch_size"), "review.calibration_batch_size", minimum=1
        ),
        calibration_tier_quota=quota,
        host=host,
        port=_int(raw.get("port"), "review.port", minimum=1),
    )


def _dataset_settings(raw: dict[str, Any]) -> DatasetSettings:
    _reject_unknown(
        raw,
        {
            "split",
            "export_formats",
            "gold_statuses",
            "diversity_warnings",
            "quality_gate",
            "label_length",
        },
        "dataset.yaml",
    )
    split_raw = _mapping(raw.get("split"), "dataset.split")
    _reject_unknown(split_raw, {"train", "val", "test", "seed"}, "dataset.split")
    train = _probability(split_raw.get("train"), "dataset.split.train")
    val = _probability(split_raw.get("val"), "dataset.split.val")
    test = _probability(split_raw.get("test"), "dataset.split.test")
    if train <= 0 or val <= 0 or test <= 0:
        raise ConfigError("dataset.split ratios must all be greater than 0")
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ConfigError("dataset.split ratios must sum to 1")
    formats = raw.get("export_formats")
    if formats != ["provenance", "simple"]:
        raise ConfigError("dataset.export_formats must be provenance and simple")
    statuses = raw.get("gold_statuses")
    if not isinstance(statuses, list) or not all(isinstance(item, str) for item in statuses):
        raise ConfigError("dataset.gold_statuses must be a list of strings")
    if set(statuses) != {status.value for status in GOLD_VERIFICATION_STATUSES}:
        raise ConfigError("dataset.gold_statuses must be human_verified and human_corrected")
    warnings = _mapping(raw.get("diversity_warnings"), "dataset.diversity_warnings")
    _reject_unknown(
        warnings,
        {
            "department_share",
            "document_share",
            "partial_crop_ratio_low",
            "partial_crop_ratio_high",
        },
        "dataset.diversity_warnings",
    )
    low = _probability(warnings.get("partial_crop_ratio_low"), "dataset.partial_crop_ratio_low")
    high = _probability(warnings.get("partial_crop_ratio_high"), "dataset.partial_crop_ratio_high")
    if low >= high:
        raise ConfigError("partial crop ratio warning range is empty")
    gate_raw = _mapping(raw.get("quality_gate"), "dataset.quality_gate")
    _reject_unknown(gate_raw, set(_QUALITY_GATE_KEYS), "dataset.quality_gate")
    gate = {
        key: _bool(gate_raw.get(key), f"dataset.quality_gate.{key}")
        for key in _QUALITY_GATE_KEYS
    }
    disabled = [key for key, value in gate.items() if not value]
    if disabled:
        raise ConfigError("quality gate checks cannot be disabled: " + ", ".join(disabled))
    label_length = _label_length_settings(raw.get("label_length"))
    return DatasetSettings(
        split=SplitSettings(
            train=train,
            val=val,
            test=test,
            seed=_int(split_raw.get("seed"), "dataset.split.seed", minimum=0),
        ),
        export_formats=("provenance", "simple"),
        gold_statuses=("human_corrected", "human_verified"),
        department_share_warning=_probability(
            warnings.get("department_share"), "dataset.department_share"
        ),
        document_share_warning=_probability(
            warnings.get("document_share"), "dataset.document_share"
        ),
        partial_crop_ratio_low=low,
        partial_crop_ratio_high=high,
        quality_gate=gate,
        label_length=label_length,
    )


def _label_length_settings(value: object) -> LabelLengthSettings:
    """Accept the shipped defaults when dataset.yaml omits the block."""
    if value is None:
        value = {
            "max_characters": 10,
            "short_below": 5,
            "short_fraction": 0.95,
            "long_min_characters": 8,
        }
    raw = _mapping(value, "dataset.label_length")
    _reject_unknown(
        raw,
        {"max_characters", "short_below", "short_fraction", "long_min_characters"},
        "dataset.label_length",
    )
    short_below = _int(raw.get("short_below"), "dataset.label_length.short_below", minimum=2)
    long_min = _int(
        raw.get("long_min_characters"),
        "dataset.label_length.long_min_characters",
        minimum=1,
    )
    maximum = _int(raw.get("max_characters"), "dataset.label_length.max_characters", minimum=1)
    fraction = _unit_fraction(raw.get("short_fraction"), "dataset.label_length.short_fraction")
    if long_min <= short_below:
        raise ConfigError("dataset.label_length long labels must start above the short limit")
    if maximum < long_min:
        raise ConfigError("dataset.label_length max_characters must cover the long labels")
    return LabelLengthSettings(
        max_characters=maximum,
        short_below=short_below,
        short_fraction=fraction,
        long_min_characters=long_min,
    )


def _fraction_range(value: object, field_name: str) -> FractionRange:
    raw = _mapping(value, field_name)
    _reject_unknown(raw, {"min_width_fraction", "max_width_fraction"}, field_name)
    minimum = _unit_fraction(raw.get("min_width_fraction"), f"{field_name}.min_width_fraction")
    maximum = _unit_fraction(raw.get("max_width_fraction"), f"{field_name}.max_width_fraction")
    if minimum >= maximum:
        raise ConfigError(f"{field_name} width range is empty")
    return FractionRange(min_width_fraction=minimum, max_width_fraction=maximum)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name} must be a mapping")
    return loaded


def _documents_dir(root: Path, source: Mapping[str, Any]) -> Path:
    override = os.environ.get("OCR_INPUT_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else (root / path).resolve()
    return _resolve(root, _str(source.get("documents_dir"), "path.documents_dir"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a mapping")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ConfigError(f"{field_name} has unknown fields: {', '.join(extra)}")


def _str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string or null")
    return value


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean")
    return value


def _int(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer")
    if value < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}")
    return value


def _ratio(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{field_name} must be a number")
    number = float(value)
    if number < 1:
        raise ConfigError(f"{field_name} must be >= 1")
    return number


def _probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{field_name} must be a number")
    number = float(value)
    if number < 0 or number > 1:
        raise ConfigError(f"{field_name} must be between 0 and 1")
    return number


def _unit_fraction(value: object, field_name: str) -> float:
    number = _probability(value, field_name)
    if number <= 0 or number > 1:
        raise ConfigError(f"{field_name} must be in (0, 1]")
    return number
