"""Stage graph for the unified OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    module: str
    depends_on: tuple[str, ...]
    sections: tuple[str, ...]
    files: tuple[str, ...]
    outputs: tuple[str, ...]
    recovery: str


STAGES: dict[str, Stage] = {
    "preflight": Stage(
        "preflight",
        "Preflight",
        "pipeline.preflight",
        (),
        ("runtime",),
        ("configs/pipeline.yaml",),
        (),
        "Install the missing dependency, then run ./run_pipeline.sh",
    ),
    "ingest": Stage(
        "ingest",
        "Ingestion",
        "indic_ocr_pipeline.pdf.ingest",
        (),
        ("source", "paths"),
        (),
        ("data/output/documents.jsonl",),
        "Put source PDFs in the documents directory, then run ./run_pipeline.sh",
    ),
    "render": Stage(
        "render",
        "Rendering",
        "indic_ocr_pipeline.rendering.pages",
        ("ingest",),
        ("paths",),
        ("configs/rendering.yaml",),
        ("data/output/pages.jsonl",),
        "Fix rendering, then run ./run_pipeline.sh",
    ),
    "detect": Stage(
        "detect",
        "Region detection",
        "indic_ocr_pipeline.detection.regions",
        ("render",),
        ("paths",),
        ("configs/detection.yaml",),
        ("data/output/regions.jsonl",),
        "Fix region detection, then run ./run_pipeline.sh",
    ),
    "crop": Stage(
        "crop",
        "Line detection",
        "indic_ocr_pipeline.detection.lines",
        ("detect",),
        ("paths",),
        ("configs/detection.yaml", "configs/cropping.yaml"),
        ("data/output/lines.jsonl", "data/output/crops.jsonl"),
        "Fix line detection, then run ./run_pipeline.sh",
    ),
    "ocr": Stage(
        "ocr",
        "OCR",
        "indic_ocr_pipeline.ocr.run",
        ("crop",),
        ("ocr", "paths"),
        ("configs/ocr.yaml",),
        ("data/output/predictions.jsonl",),
        "Fix the OCR backend, then run ./run_pipeline.sh",
    ),
    "compare": Stage(
        "compare",
        "Comparison",
        "pipeline.dataset_flow",
        ("ocr",),
        ("ocr",),
        (),
        ("data/output/consensus.jsonl",),
        "Fix comparison, then run ./run_pipeline.sh",
    ),
    "calibrate": Stage(
        "calibrate",
        "Review / calibration",
        "pipeline.dataset_flow",
        ("compare",),
        ("calibration", "seed"),
        (),
        ("data/review/calibration/sample.jsonl",),
        "Fix calibration, then run ./run_pipeline.sh",
    ),
    "export": Stage(
        "export",
        "Dataset export",
        "pipeline.dataset_flow",
        ("calibrate",),
        ("paths",),
        ("configs/dataset.yaml",),
        ("data/datasets/train.jsonl", "data/datasets/val.jsonl"),
        "Record human gold decisions, then run ./run_pipeline.sh",
    ),
    "train": Stage(
        "train",
        "Training",
        "IndicPhotoOCR.recognition.finetune",
        ("export",),
        ("training", "paths"),
        ("configs/training.yaml", "configs/dataset.yaml"),
        ("training/complete.json",),
        (
            "Approve human gold, pass --checkpoint with a local file or a "
            "Hugging Face link, then run ./run_pipeline.sh"
        ),
    ),
    "eval_checkpoints": Stage(
        "eval_checkpoints",
        "Checkpoint evaluation",
        "evaluation.harness",
        ("train",),
        ("evaluation", "training"),
        ("configs/evaluation.yaml",),
        ("evaluation/raw_scores.json",),
        "Fix checkpoint evaluation, then run ./run_pipeline.sh",
    ),
    "select": Stage(
        "select",
        "Best checkpoint selection",
        "training.selection",
        ("eval_checkpoints",),
        ("evaluation", "training"),
        ("configs/evaluation.yaml",),
        ("evaluation/checkpoint_scoreboard.json", "BEST_CHECKPOINT"),
        "Selection uses the validation split only. Run ./run_pipeline.sh",
    ),
    "frozen": Stage(
        "frozen",
        "Frozen-100 evaluation",
        "evaluation",
        ("select",),
        ("evaluation", "paths"),
        ("configs/evaluation.yaml",),
        ("evaluation/frozen100/metrics.json",),
        "Restore data/eval_data, then run ./run_pipeline.sh",
    ),
    "report": Stage(
        "report",
        "Final report",
        "pipeline.report",
        ("frozen",),
        ("paths",),
        (),
        ("reports/final_report.html",),
        "Run ./run_pipeline.sh to rewrite the report",
    ),
}

ORDER: tuple[str, ...] = tuple(STAGES)

DATASET_GROUP = (
    "ingest",
    "render",
    "detect",
    "crop",
    "ocr",
    "compare",
    "calibrate",
    "export",
)
TRAIN_GROUP = ("train",)
EVAL_GROUP = ("eval_checkpoints", "select", "frozen", "report")

ALIASES = {
    "training": "train",
    "evaluation": "eval_checkpoints",
    "eval": "eval_checkpoints",
    "lines": "crop",
    "crops": "crop",
    "review": "calibrate",
    "calibration": "calibrate",
    "dataset": "ingest",
}


def descendants(name: str) -> tuple[str, ...]:
    """Stages that depend on ``name``, directly or further downstream."""
    found: list[str] = []
    for stage in ORDER:
        depends = STAGES[stage].depends_on
        if name in depends or any(item in depends for item in found):
            if stage != name:
                found.append(stage)
    return tuple(found)


def resolve_stage_name(name: str) -> str:
    key = name.strip().lower()
    if key in STAGES:
        return key
    if key in ALIASES:
        return ALIASES[key]
    known = ", ".join(ORDER)
    raise KeyError(f"unknown stage {name!r}. Known stages: {known}")
