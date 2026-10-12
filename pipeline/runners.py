"""Stage runners for production and the end-to-end fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.dataset import assert_training_rows_safe, load_frozen_ids
from training.selection import select_best

from pipeline.dataset_flow import (
    build_consensus,
    count_summary,
    export_datasets,
    read_jsonl,
    write_calibration,
    write_jsonl,
)
from pipeline.errors import StageFailed
from pipeline.gold import ensure_invalidated_archive
from pipeline.report import write_report

FIXTURE_CROPS: list[dict[str, Any]] = [
    {
        "crop_id": "A1",
        "document_id": "DOC_A",
        "page_id": "DOC_A_P1",
        "parent_line_id": "DOC_A_L1",
        "texts": {"indicocr": "अ", "chitrapathak": "अ", "tesseract": "अ"},
        "bad_boundary": False,
    },
    {
        "crop_id": "A2",
        "document_id": "DOC_A",
        "page_id": "DOC_A_P1",
        "parent_line_id": "DOC_A_L2",
        "texts": {"indicocr": "ब", "chitrapathak": "ब", "tesseract": "क"},
        "bad_boundary": False,
    },
    {
        "crop_id": "B1",
        "document_id": "DOC_B",
        "page_id": "DOC_B_P1",
        "parent_line_id": "DOC_B_L1",
        "texts": {"indicocr": "ड", "chitrapathak": "ई", "tesseract": "फ"},
        "bad_boundary": False,
    },
    {
        "crop_id": "B2",
        "document_id": "DOC_B",
        "page_id": "DOC_B_P1",
        "parent_line_id": "DOC_B_L2",
        "texts": {"indicocr": "", "chitrapathak": "ग", "tesseract": "ग"},
        "bad_boundary": False,
    },
    {
        "crop_id": "B3",
        "document_id": "DOC_B",
        "page_id": "DOC_B_P1",
        "parent_line_id": "DOC_B_L3",
        "texts": {"indicocr": "ह", "chitrapathak": "ह", "tesseract": "ह"},
        "bad_boundary": True,
    },
]


class Ctx:
    def __init__(
        self,
        root: Path,
        run_dir: Path,
        config: dict[str, Any],
        mode: str,
        stage: str,
    ) -> None:
        self.root = root
        self.run_dir = run_dir
        self.config = config
        self.mode = mode
        self.stage = stage


def run_stage(ctx: Ctx) -> dict[str, Any]:
    if ctx.config.get("execution", {}).get("fail_stage") == ctx.stage:
        raise StageFailed(ctx.stage, f"injected failure in {ctx.stage}", module="pipeline.runners")
    handler = _HANDLERS[ctx.stage]
    try:
        return handler(ctx)
    except StageFailed:
        raise
    except Exception as exc:
        raise StageFailed(ctx.stage, str(exc), module=handler.__module__) from exc


def _documents(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "fixture":
        rows = [
            {"document_id": "DOC_A", "source_path": "data/raw/documents/a.pdf"},
            {"document_id": "DOC_B", "source_path": "data/raw/documents/b.pdf"},
        ]
        write_jsonl(ctx.root / "manifests" / "documents.jsonl", rows)
        return {"messages": ["2 documents"], "row_counts": {"documents": 2}}
    from indic_ocr_pipeline.pdf.ingest import main as ingest_main

    _delegate(ctx, ingest_main)
    count = len(read_jsonl(ctx.root / "manifests" / "documents.jsonl"))
    return {"messages": [f"{count} documents"], "row_counts": {"documents": count}}


def _pages(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "fixture":
        rows = [
            {"page_id": "DOC_A_P1", "document_id": "DOC_A"},
            {"page_id": "DOC_B_P1", "document_id": "DOC_B"},
        ]
        write_jsonl(ctx.root / "manifests" / "pages.jsonl", rows)
        return {"messages": ["2 pages"], "row_counts": {"pages": 2}}
    from indic_ocr_pipeline.rendering.pages import main as render_main

    _delegate(ctx, render_main)
    count = len(read_jsonl(ctx.root / "manifests" / "pages.jsonl"))
    return {"messages": [f"{count} pages"], "row_counts": {"pages": count}}


def _regions(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "fixture":
        rows = [
            {"region_id": "DOC_A_R1", "document_id": "DOC_A", "page_id": "DOC_A_P1"},
            {"region_id": "DOC_B_R1", "document_id": "DOC_B", "page_id": "DOC_B_P1"},
        ]
        write_jsonl(ctx.root / "manifests" / "regions.jsonl", rows)
        return {"messages": ["2 regions"], "row_counts": {"regions": 2}}
    from indic_ocr_pipeline.detection.regions import main as detect_main

    _delegate(ctx, detect_main)
    count = len(read_jsonl(ctx.root / "manifests" / "regions.jsonl"))
    return {"messages": [f"{count} regions"], "row_counts": {"regions": count}}


def _crops(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "fixture":
        crops = []
        lines = []
        for spec in FIXTURE_CROPS:
            image = ctx.root / "data" / "crops" / f"{spec['crop_id']}.png"
            _write_png(image, _line_text(spec))
            crops.append(
                {
                    "crop_id": spec["crop_id"],
                    "document_id": spec["document_id"],
                    "page_id": spec["page_id"],
                    "parent_line_id": spec["parent_line_id"],
                    "image_path": str(image.relative_to(ctx.root)),
                    "status": "complete",
                    "bad_boundary": spec["bad_boundary"],
                    "crop_type": "full",
                }
            )
            lines.append({"line_id": spec["parent_line_id"], "document_id": spec["document_id"]})
        write_jsonl(ctx.root / "manifests" / "crops.jsonl", crops)
        write_jsonl(ctx.root / "manifests" / "lines.jsonl", lines)
        return {
            "messages": [f"{len(crops)} crops"],
            "row_counts": {"crops": len(crops), "lines": len(lines)},
        }
    from indic_ocr_pipeline.detection.lines import main as lines_main

    _delegate(ctx, lines_main)
    count = len(read_jsonl(ctx.root / "manifests" / "crops.jsonl"))
    return {"messages": [f"{count} crops"], "row_counts": {"crops": count}}


def _ocr(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "fixture":
        crops = {row["crop_id"]: row for row in read_jsonl(ctx.root / "manifests" / "crops.jsonl")}
        specs = {spec["crop_id"]: spec for spec in FIXTURE_CROPS}
        predictions = []
        for crop_id, crop in crops.items():
            texts = specs[crop_id]["texts"]
            for backend, text in texts.items():
                predictions.append(
                    {
                        "prediction_id": f"{crop_id}_OCR_{backend}",
                        "crop_id": crop_id,
                        "document_id": crop["document_id"],
                        "page_id": crop["page_id"],
                        "parent_line_id": crop["parent_line_id"],
                        "backend": backend,
                        "text": text,
                        "status": "complete",
                    }
                )
        write_jsonl(ctx.root / "manifests" / "predictions.jsonl", predictions)
        return {
            "messages": [f"{len(predictions)} predictions"],
            "row_counts": {"predictions": len(predictions)},
        }
    from indic_ocr_pipeline.ocr.run import main as ocr_main

    _prepare_ocr_models(ctx)
    _delegate(ctx, ocr_main)
    count = len(read_jsonl(ctx.root / "manifests" / "predictions.jsonl"))
    return {"messages": [f"{count} predictions"], "row_counts": {"predictions": count}}


def _compare(ctx: Ctx) -> dict[str, Any]:
    from pipeline.budget import select_crop_ids

    crops = read_jsonl(ctx.root / "manifests" / "crops.jsonl")
    limit = ctx.config.get("dataset_size")
    crop_ids = set(select_crop_ids(crops, int(limit))) if limit else None
    records = build_consensus(ctx.root, crop_ids)
    crops = len(records)
    predictions = crops * 3
    return {
        "messages": count_summary(crops, predictions, len(records)),
        "row_counts": {"crops": crops, "predictions": predictions, "candidates": len(records)},
    }


def _calibrate(ctx: Ctx) -> dict[str, Any]:
    chosen = write_calibration(ctx.root, ctx.config)
    return {
        "messages": [f"{len(chosen)} calibration rows"],
        "row_counts": {"calibration": len(chosen)},
    }


def _export(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "fixture":
        _seed_fixture_reviews(ctx.root)
    counts = export_datasets(ctx.root, ctx.config)
    return {
        "messages": [f"{counts['train']} train", f"{counts['val']} validation"],
        "row_counts": counts,
    }


def _train(ctx: Ctx) -> dict[str, Any]:
    train_rows = read_jsonl(ctx.root / "data" / "datasets" / "train.jsonl")
    val_rows = read_jsonl(ctx.root / "data" / "datasets" / "val.jsonl")
    if not train_rows and not val_rows:
        raise StageFailed(
            "train",
            "No approved training rows. Consensus is not gold, and invalidated "
            "Phase 10 reviews cannot enter training.",
            module="pipeline.runners",
        )
    frozen_dir = ctx.root / ctx.config["paths"]["frozen_benchmark"]
    frozen_ids = load_frozen_ids(frozen_dir)
    allow_auto = bool(ctx.config.get("include_auto_accepted"))
    assert_training_rows_safe(train_rows, frozen_ids, frozen_dir, allow_auto=allow_auto)
    assert_training_rows_safe(val_rows, frozen_ids, frozen_dir)
    training_cfg = ctx.config.get("training") or {}
    backend = str(training_cfg.get("backend", "parseq"))
    schedule = _training_schedule(train_rows, training_cfg)
    if ctx.mode == "fixture" or backend == "fixture":
        _write_fixture_checkpoints(ctx, val_rows or train_rows, schedule)
        return {
            "messages": [
                "fixture checkpoints written",
                _validation_message(schedule),
            ],
            "row_counts": {"checkpoints": 2, "validation_steps": len(schedule["validation_steps"])},
        }
    if not train_rows:
        raise StageFailed(
            "train",
            "Training split is empty.",
            module="IndicPhotoOCR.recognition.finetune",
        )
    if not val_rows:
        raise StageFailed(
            "train",
            "Validation split is empty. Checkpoint selection needs the development "
            "set and cannot use frozen-100.",
            module="IndicPhotoOCR.recognition.finetune",
        )
    checkpoint = _resolve_base_checkpoint(ctx)
    from IndicPhotoOCR.recognition.finetune import finetune

    output = ctx.run_dir / "checkpoints" / "finetuned.ckpt"
    output.parent.mkdir(parents=True, exist_ok=True)
    device = training_cfg.get("device")
    if device in {None, "", "auto"}:
        device = None
    trained = finetune(
        checkpoint=checkpoint,
        image_dir=str(ctx.root),
        labels=str(ctx.root / "data" / "datasets" / "train.jsonl"),
        output=str(output),
        val_labels=str(ctx.root / "data" / "datasets" / "val.jsonl"),
        val_split=0.0,
        epochs=int(training_cfg.get("epochs", 1)),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        device=device,
        seed=int(ctx.config.get("seed", 3407)),
        limit_train_batches=1 if training_cfg.get("steps") in {1, "1"} else None,
    )
    _write_training_record(
        ctx,
        {
            "backend": "parseq",
            "checkpoint_dir": str(output.parent),
            "val_check_interval": schedule["val_check_interval"],
            "validation_steps": schedule["validation_steps"],
            **trained,
        },
    )
    return {
        "messages": [
            "completed",
            _validation_message(schedule),
        ],
        "row_counts": {"validation_steps": len(schedule["validation_steps"])},
    }


def _eval(ctx: Ctx) -> dict[str, Any]:
    from evaluation.harness import score_corpus

    val_rows = read_jsonl(ctx.root / "data" / "datasets" / "val.jsonl")
    if not val_rows:
        raise StageFailed(
            "eval_checkpoints",
            "Validation split is empty.",
            module="evaluation.harness",
        )
    frozen_ids = load_frozen_ids(ctx.root / ctx.config["paths"]["frozen_benchmark"])
    checkpoint_root = _checkpoint_root(ctx)
    if ctx.mode != "fixture":
        _materialize_checkpoint_predictions(ctx, checkpoint_root, val_rows)
    scored = []
    directories = (checkpoint_root / "checkpoints").glob("*")
    for directory in sorted(path for path in directories if path.is_dir()):
        sidecar = directory / "predictions.jsonl"
        if not sidecar.is_file():
            continue
        predicted = {row["sample_id"]: row.get("prediction", "") for row in read_jsonl(sidecar)}
        sample_ids = [str(row["sample_id"]) for row in val_rows]
        if set(sample_ids) & frozen_ids:
            raise StageFailed(
                "eval_checkpoints",
                "validation split contains frozen-100 ids",
                module="evaluation.harness",
            )
        metrics = score_corpus(
            [str(row["expected_text"]) for row in val_rows],
            [str(predicted.get(row["sample_id"], "")) for row in val_rows],
        )
        metrics["mode"] = "piecewise"
        scored.append(
            {
                "checkpoint": str(directory.name),
                "metrics": metrics,
                "sample_ids": sample_ids,
                "split": "val",
            }
        )
    if not scored:
        raise StageFailed(
            "eval_checkpoints",
            "No checkpoint predictions found.",
            module="evaluation.harness",
        )
    path = ctx.run_dir / "evaluation" / "raw_scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "messages": [f"{len(scored)} checkpoints scored"],
        "row_counts": {"checkpoints": len(scored)},
    }


def _select(ctx: Ctx) -> dict[str, Any]:
    raw_path = ctx.run_dir / "evaluation" / "raw_scores.json"
    if not raw_path.is_file():
        recorded = _checkpoint_root(ctx)
        raw_path = recorded / "evaluation" / "raw_scores.json"
    scored = json.loads(raw_path.read_text(encoding="utf-8"))
    frozen_ids = load_frozen_ids(ctx.root / ctx.config["paths"]["frozen_benchmark"])
    board = select_best(scored, frozen_ids=frozen_ids, primary_metric="cer")
    out = ctx.run_dir / "evaluation" / "checkpoint_scoreboard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(board, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ctx.run_dir / "BEST_CHECKPOINT").write_text(board["selected"] + "\n", encoding="utf-8")
    return {
        "messages": [f"best checkpoint selected ({board['selected']})"],
        "row_counts": {"checkpoints": len(board["rows"])},
    }


def _frozen(ctx: Ctx) -> dict[str, Any]:
    best = (ctx.run_dir / "BEST_CHECKPOINT").read_text(encoding="utf-8").strip()
    board_path = ctx.run_dir / "evaluation" / "checkpoint_scoreboard.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))
    frozen_dir = ctx.root / ctx.config["paths"]["frozen_benchmark"]
    frozen_ids = load_frozen_ids(frozen_dir)
    for row in board["rows"]:
        leaked = set(row.get("sample_ids") or []) & frozen_ids
        if leaked or row.get("frozen_used"):
            raise StageFailed("frozen", "scoreboard includes frozen-100", module="evaluation")
    rows = read_jsonl(frozen_dir / "validation.jsonl")
    checkpoint_file = _checkpoint_file(ctx, best)
    if checkpoint_file is not None:
        device_name = str((ctx.config.get("evaluation") or {}).get("device", "auto"))
        metrics = _score_frozen_checkpoint(checkpoint_file, frozen_dir, device_name)
    elif ctx.mode == "fixture":
        metrics = {
            "benchmark": "frozen100",
            "used_for_selection": False,
            "checkpoint": best,
            "rows": len(rows),
            "note": "Fixture checkpoints are score sidecars. The recogniser was not loaded.",
        }
    else:
        raise StageFailed(
            "frozen",
            f"Selected checkpoint {best} is not a local .ckpt file.",
            module="evaluation.inference",
        )
    metrics["benchmark"] = "frozen100"
    metrics["used_for_selection"] = False
    metrics["checkpoint"] = best
    out = ctx.run_dir / "evaluation" / "frozen100" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"messages": ["frozen-100 complete"], "row_counts": {"frozen": len(rows)}}


def _report(ctx: Ctx) -> dict[str, Any]:
    board_path = ctx.run_dir / "evaluation" / "checkpoint_scoreboard.json"
    frozen_path = ctx.run_dir / "evaluation" / "frozen100" / "metrics.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    summary = {
        "run_id": ctx.run_dir.name,
        "counts": {
            "crops": len(read_jsonl(ctx.root / "manifests" / "crops.jsonl")),
            "predictions": len(read_jsonl(ctx.root / "manifests" / "predictions.jsonl")),
            "candidates": len(read_jsonl(ctx.root / "manifests" / "consensus.jsonl")),
            "train": len(read_jsonl(ctx.root / "data" / "datasets" / "train.jsonl")),
            "val": len(read_jsonl(ctx.root / "data" / "datasets" / "val.jsonl")),
        },
        "scoreboard": board,
        "frozen": {
            "cer": frozen.get("cer"),
            "wer": frozen.get("wer"),
            "exact_match_rate": frozen.get("exact_match_rate"),
            "used_for_selection": False,
            "checkpoint": frozen.get("checkpoint"),
        },
    }
    path = write_report(ctx.run_dir, summary)
    return {"messages": [str(path)], "row_counts": {}}


def _delegate(ctx: Ctx, main: Any) -> None:
    code = main([])
    if code != 0:
        raise StageFailed(ctx.stage, f"exit code {code}", module=getattr(main, "__module__", ""))


def _line_text(spec: dict[str, Any]) -> str:
    """The pixels of a fixture crop. Majority text, otherwise the first vote."""
    if spec.get("line_text"):
        return str(spec["line_text"])
    votes = [str(spec["texts"][name]) for name in ("indicocr", "chitrapathak", "tesseract")]
    counts: dict[str, int] = {}
    for vote in votes:
        if vote.strip():
            counts[vote] = counts.get(vote, 0) + 1
    if not counts:
        raise ValueError(f"fixture crop {spec['crop_id']} has no visible text")
    return max(counts, key=lambda vote: (counts[vote], vote))


def _write_png(path: Path, text: str) -> None:
    from pipeline.line_image import write_text_image

    write_text_image(path, text)


def _seed_fixture_reviews(root: Path) -> None:
    archive = root / "data" / "review" / "archive" / "invalidated_phase10" / "invalidated_ids.json"
    if archive.is_file():
        return
    historical = {
        "crop_id": "OLD_CROP",
        "document_id": "DOC_A",
        "page_id": "DOC_A_P1",
        "parent_line_id": "DOC_A_L0",
        "image_path": "data/crops/OLD_CROP.png",
        "reviewer_id": "local_reviewer",
        "verification_timestamp": "2026-10-01T00:00:00+00:00",
        "verification_status": "human_verified",
        "review_decision": "accept_candidate",
        "final_text": "जुने",
        "metadata": {"decision_label": "human_verified", "reviewer_app_version": "phase8-review-1"},
        "original_candidate_texts": {"indicocr": "जुने", "chitrapathak": "जुने", "tesseract": "जुने"},
    }
    verified = root / "manifests" / "verified.jsonl"
    write_jsonl(verified, [historical])
    ensure_invalidated_archive(root)
    fresh = [
        _human("A1", "DOC_A", "DOC_A_P1", "DOC_A_L1", "human_verified", "अ"),
        _human("B1", "DOC_B", "DOC_B_P1", "DOC_B_L1", "human_corrected", "योग्य"),
    ]
    write_jsonl(verified, [historical, *fresh])


def _human(
    crop_id: str,
    document_id: str,
    page_id: str,
    line_id: str,
    decision: str,
    text: str,
) -> dict[str, Any]:
    return {
        "crop_id": crop_id,
        "document_id": document_id,
        "page_id": page_id,
        "parent_line_id": line_id,
        "image_path": f"data/crops/{crop_id}.png",
        "reviewer_id": "reviewer_1",
        "verification_timestamp": "2026-10-08T12:00:00+00:00",
        "review_origin": "human",
        "verification_status": decision,
        "decision": decision,
        "final_text": text,
        "gold_text": text,
        "metadata": {"decision_label": decision, "review_origin": "human"},
        "original_candidate_texts": {"indicocr": text, "chitrapathak": text, "tesseract": text},
    }


def _validation_message(schedule: dict[str, Any]) -> str:
    steps = schedule["validation_steps"]
    if len(steps) >= 2:
        return "validation at 30% and 70% of training, then once after training"
    if len(steps) == 1:
        return "validation once during training, then once after training"
    return "validation once after training"


def _training_schedule(
    train_rows: list[dict[str, Any]],
    training_cfg: dict[str, Any],
) -> dict[str, Any]:
    from training.schedule import resolve_training_schedule

    configured = training_cfg.get("steps")
    return resolve_training_schedule(
        train_count=len(train_rows),
        batch_size=int(training_cfg.get("batch_size", 8)),
        epochs=int(training_cfg.get("epochs", 1)),
        steps=int(configured) if configured else None,
        val_checks=int(training_cfg.get("val_checks", 10)),
    )


def _write_training_record(ctx: Ctx, payload: dict[str, Any]) -> None:
    marker = ctx.run_dir / "training" / "complete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_fixture_checkpoints(
    ctx: Ctx,
    rows: list[dict[str, Any]],
    schedule: dict[str, Any],
) -> None:
    root = ctx.run_dir / "checkpoints"
    for name, mutate in (("ckpt_low", lambda text: text), ("ckpt_high", lambda text: text + "X")):
        predictions = [
            {"sample_id": row["sample_id"], "prediction": mutate(str(row["expected_text"]))}
            for row in rows
        ]
        write_jsonl(root / name / "predictions.jsonl", predictions)
    _write_training_record(
        ctx,
        {
            "backend": "fixture",
            "frozen_used_for_selection": False,
            **schedule,
        },
    )


def _image_path(root: Path, row: dict[str, Any]) -> Path:
    raw = row.get("image_filename") or row.get("crop_path") or row.get("image_path")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return root / path


def _materialize_checkpoint_predictions(
    ctx: Ctx,
    checkpoint_root: Path,
    val_rows: list[dict[str, Any]],
) -> None:
    """Run each local .ckpt on the validation images and store one sidecar."""
    from evaluation.inference import load_model, recognise_canonical, resolve_device

    device_name = str((ctx.config.get("evaluation") or {}).get("device", "auto"))
    device = resolve_device(device_name).device
    folder = checkpoint_root / "checkpoints"
    if not folder.is_dir():
        return
    for ckpt in sorted(folder.glob("*.ckpt")):
        sidecar_dir = folder / ckpt.stem
        if (sidecar_dir / "predictions.jsonl").is_file():
            continue
        model = load_model(str(ckpt), device)
        predictions = []
        for row in val_rows:
            image = _image_path(ctx.root, row)
            result = recognise_canonical(model, device, str(image))
            predictions.append(
                {"sample_id": row["sample_id"], "prediction": result.prediction}
            )
        write_jsonl(sidecar_dir / "predictions.jsonl", predictions)


def _checkpoint_file(ctx: Ctx, name: str) -> Path | None:
    folder = _checkpoint_root(ctx) / "checkpoints"
    direct = folder / name
    if direct.is_file():
        return direct
    suffixed = folder / f"{name}.ckpt"
    if suffixed.is_file():
        return suffixed
    return None


def _score_frozen_checkpoint(
    checkpoint: Path,
    frozen_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    from evaluation.canonical import pipeline_fingerprint
    from evaluation.dataset import load_validation_dataset
    from evaluation.harness import score_corpus
    from evaluation.inference import load_model, recognise_canonical, resolve_device

    device = resolve_device(device_name).device
    model = load_model(str(checkpoint), device)
    samples = load_validation_dataset(frozen_dir)
    expected = [sample.expected_text for sample in samples]
    predicted = [
        recognise_canonical(model, device, sample.image_path).prediction
        for sample in samples
    ]
    metrics = score_corpus(expected, predicted)
    metrics["rows"] = len(samples)
    metrics["mode"] = "piecewise"
    metrics["pipeline"] = pipeline_fingerprint()
    return metrics


def _resolve_base_checkpoint(ctx: Ctx) -> str:
    from pipeline.weights import WeightError, resolve_weight

    training = ctx.config.get("training") or {}
    spec = str(training.get("base_checkpoint") or training.get("model_root") or "")
    try:
        return str(resolve_weight(spec, ctx.root / "models" / "cache", expect="file"))
    except WeightError as exc:
        raise StageFailed(
            "train",
            f"{exc}. Pass --checkpoint with a local file or a Hugging Face link.",
            module="IndicPhotoOCR.recognition.finetune",
        ) from exc


def _prepare_ocr_models(ctx: Ctx) -> None:
    import os

    from pipeline.registry import default_models, resolve_models, write_resolved
    from pipeline.weights import WeightError

    models = ctx.config.get("models") or {}
    env_names = {
        "indicocr": "INDICOCR_MODEL_PATH",
        "chitrapathak": "CHITRAPATHAK_MODEL_PATH",
    }
    cache = ctx.root / "models" / "cache"
    slots = [dict(slot) for slot in (ctx.config.get("ocr_models") or default_models())]
    for name in ("indicocr", "chitrapathak"):
        override = models.get(name)
        if not isinstance(override, str) or not override.strip():
            continue
        text = override.strip()
        for slot in slots:
            if slot.get("name") != name:
                continue
            if Path(text).expanduser().exists():
                slot["path"] = text
            else:
                slot["hf"] = text
    prompt = str((ctx.config.get("language_profile") or {}).get("prompt") or "")
    try:
        resolved = resolve_models(slots, cache, prompt=prompt, environ=dict(os.environ))
    except WeightError as exc:
        raise StageFailed("ocr", str(exc), module="indic_ocr_pipeline.ocr.run") from exc
    for item in resolved:
        env_name = env_names.get(item.name)
        if env_name and item.path and item.revision != "unconfigured":
            os.environ[env_name] = item.path
    record = cache / "resolved.json"
    write_resolved(record, resolved)
    os.environ["OCR_RESOLVED"] = str(record)
    if ctx.config.get("ocr_models"):
        os.environ["OCR_BACKENDS"] = ",".join(item.name for item in resolved)


def _checkpoint_root(ctx: Ctx) -> Path:
    profile = ctx.config.get("language_profile") or {}
    language = str(ctx.config.get("language") or "")
    runs_dir = profile.get("runs_dir") if isinstance(profile, dict) else None
    runs_root = ctx.root / "runs"
    if language and language != "marathi" and isinstance(runs_dir, str) and runs_dir:
        runs_root = ctx.root / runs_dir
    state_path = runs_root / "pipeline_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        recorded = (state.get("stages") or {}).get("train", {}).get("artifact_root")
        if recorded and (Path(recorded) / "checkpoints").is_dir():
            return Path(recorded)
    return ctx.run_dir


_HANDLERS = {
    "ingest": _documents,
    "render": _pages,
    "detect": _regions,
    "crop": _crops,
    "ocr": _ocr,
    "compare": _compare,
    "calibrate": _calibrate,
    "export": _export,
    "train": _train,
    "eval_checkpoints": _eval,
    "select": _select,
    "frozen": _frozen,
    "report": _report,
}
