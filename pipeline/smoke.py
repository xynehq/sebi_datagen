"""Score a few real lines without touching production reviews.

The labels written here are human-verified only inside ``runs/live_smoke``.
They are a test mark requested for this run. They are not production gold.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from evaluation.core_metrics import sample_core_metrics
from evaluation.harness import score_corpus
from training.dataset import assert_training_rows_safe, load_frozen_ids

SMOKE_LINES = (
    ("01.png", "मराठी भाषा संवर्धन पुरस्कार,"),
    ("02.png", "सन २०१८ जाहीर करण्याबाबत."),
    ("04.png", "मराठी भाषा विभाग"),
)


def run_smoke(
    repo: Path,
    *,
    adapter: Path | None = None,
    official_base: Path | None = None,
    checkpoint: Path | None = None,
    pdf: Path | None = None,
) -> int:
    """Train on two real lines and print CER for PARSeq and the LoRA adapter."""
    _use_training_interpreter()
    workspace = repo / "runs" / "live_smoke"
    if workspace.exists():
        shutil.rmtree(workspace)
    images = workspace / "images"
    images.mkdir(parents=True)
    crops = repo / "runs" / "smoke" / "crops"
    rows = _copy_lines(crops, images)
    frozen_dir = repo / "data" / "eval_data"
    assert_training_rows_safe(rows, load_frozen_ids(frozen_dir), frozen_dir)
    _write_jsonl(workspace / "reviews" / "human_verified.jsonl", rows)
    train, val = rows[:2], rows[2:]
    _write_jsonl(workspace / "data" / "datasets" / "train.jsonl", train)
    _write_jsonl(workspace / "data" / "datasets" / "val.jsonl", val)
    page = _render_pdf(repo, workspace, pdf)
    scores: dict[str, Any] = {
        "workspace": str(workspace),
        "labels": "human_verified inside runs/live_smoke only",
        "pdf_page": str(page) if page else "",
        "models": {},
    }
    ckpt = _checkpoint(checkpoint)
    if ckpt is not None:
        try:
            scores["models"]["parseq"] = _train_and_score(workspace, images, ckpt, train, rows)
        except Exception as exc:
            scores["models"]["parseq_error"] = str(exc)
            print(f"PARSeq training failed: {exc}", file=sys.stderr)
    else:
        print("PARSeq training skipped: pass --base-checkpoint with a .ckpt file.")
    base = _official_base(official_base)
    if adapter is not None and base is not None:
        try:
            scores["models"]["chitrapathak_lora"] = _score_lora(adapter, base, images, rows)
        except Exception as exc:
            scores["models"]["chitrapathak_lora_error"] = str(exc)
            print(f"LoRA scoring failed: {exc}", file=sys.stderr)
    elif adapter is not None:
        print("LoRA scoring skipped: pass --chitrapathak with the official base directory.")
    if not any(name in scores["models"] for name in ("parseq", "chitrapathak_lora")):
        print("No scores were computed.", file=sys.stderr)
        return 2
    _log_wandb(scores)
    destination = workspace / "scores.json"
    destination.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_scores(scores)
    print(f"scores: {destination}")
    return 0


def _use_training_interpreter() -> None:
    if os.environ.get("INDIC_OCR_SMOKE_READY") == "1":
        return
    try:
        import pytorch_lightning  # noqa: F401
        import wandb  # noqa: F401
    except Exception:
        other = os.environ.get("MARATHI_OCR_PYTHON", "").strip()
        if other and Path(other).is_file():
            os.environ["INDIC_OCR_SMOKE_READY"] = "1"
            os.execv(other, [other, "-m", "indic_ocr_pipeline", *sys.argv[1:]])
    os.environ["INDIC_OCR_SMOKE_READY"] = "1"


def _copy_lines(crops: Path, images: Path) -> list[dict[str, Any]]:
    rows = []
    for index, (name, text) in enumerate(SMOKE_LINES, start=1):
        source = crops / name
        if not source.is_file():
            raise FileNotFoundError(f"real smoke crop is missing: {source}")
        shutil.copyfile(source, images / name)
        rows.append(
            {
                "sample_id": f"smoke-{index:02d}",
                "image_filename": name,
                "expected_text": text,
                "verification_method": "human",
                "verification_status": "human_verified",
                "gold": True,
                "reviewer_id": "test-review",
                "smoke_test": True,
            }
        )
    return rows


def _render_pdf(repo: Path, workspace: Path, pdf: Path | None) -> Path | None:
    source = _find_pdf(repo, pdf)
    if source is None:
        print("PDF render skipped: no PDF was found.")
        return None
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(source)
    try:
        bitmap = document[0].render(scale=2)
        image = bitmap.to_pil()
    finally:
        document.close()
    page = workspace / "page.png"
    image.save(page)
    print(f"rendered {source.name} page 1 to {page}")
    return page


def _find_pdf(repo: Path, pdf: Path | None) -> Path | None:
    if pdf is not None:
        candidate = pdf.expanduser()
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            return candidate
        if candidate.is_dir():
            found = sorted(candidate.glob("*.pdf"), key=lambda path: path.stat().st_size)
            return found[0] if found else None
        return None
    folder = repo / "data" / "raw" / "documents"
    if not folder.is_dir():
        return None
    found = sorted(folder.glob("*.pdf"), key=lambda path: path.stat().st_size)
    return found[0] if found else None


def _as_path(value: Path | str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _checkpoint(path: Path | str | None) -> Path | None:
    chosen = _as_path(path)
    if chosen is not None:
        return chosen
    return _as_path(os.environ.get("CHECKPOINT", "").strip())


def _official_base(path: Path | str | None) -> Path | None:
    chosen = _as_path(path)
    if chosen is not None:
        return chosen
    return _as_path(os.environ.get("CHITRAPATHAK_MODEL_PATH", "").strip())


def _train_and_score(
    workspace: Path,
    images: Path,
    checkpoint: Path,
    train: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from IndicPhotoOCR.recognition.finetune import finetune

    output = workspace / "checkpoints" / "finetuned.ckpt"
    output.parent.mkdir(parents=True, exist_ok=True)
    trained = finetune(
        checkpoint=str(checkpoint),
        image_dir=str(images),
        labels=str(workspace / "data" / "datasets" / "train.jsonl"),
        output=str(output),
        val_labels=str(workspace / "data" / "datasets" / "val.jsonl"),
        val_split=0.0,
        epochs=1,
        batch_size=1,
        device=None,
        seed=3407,
        limit_train_batches=1,
    )
    _release_torch()
    base_metrics = _score_parseq(checkpoint, images, rows)
    tuned_metrics = _score_parseq(output, images, rows)
    _release_torch()
    return {
        "train_samples": len(train),
        "val_samples": len(rows) - len(train),
        "base_checkpoint": str(checkpoint),
        "finetuned_checkpoint": str(output),
        "training": {
            key: trained.get(key)
            for key in ("num_train_samples", "num_val_samples", "epochs")
        },
        "base": base_metrics,
        "finetuned": tuned_metrics,
    }


def _score_parseq(checkpoint: Path, images: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from evaluation.inference import load_model, recognise_canonical, resolve_device

    device = resolve_device("auto").device
    model = load_model(str(checkpoint), device)
    predictions = []
    details = []
    for row in rows:
        image = images / str(row["image_filename"])
        prediction = recognise_canonical(model, device, str(image)).prediction
        predictions.append(prediction)
        sample = sample_core_metrics(str(row["expected_text"]), prediction)
        details.append(
            {
                "sample_id": row["sample_id"],
                "expected_text": row["expected_text"],
                "prediction": prediction,
                "cer": sample.cer,
            }
        )
    del model
    _release_torch()
    metrics = score_corpus([str(row["expected_text"]) for row in rows], predictions)
    return {
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "exact_match_rate": metrics["exact_match_rate"],
        "lines": details,
    }


def _score_lora(
    adapter: Path,
    base: Path,
    images: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from indic_ocr_pipeline.config import BackendSettings
    from models.chitrapathak.backend import ChitrapathakBackend
    from pipeline.discovery import inspect_path, resolve_adapter_base

    model = inspect_path(adapter.expanduser())
    if model is None:
        raise FileNotFoundError(f"{adapter} is not a complete model")
    resolved, source = resolve_adapter_base(model, base)
    settings = BackendSettings(
        name="chitrapathak",
        cmd=None,
        lang=None,
        psm=None,
        model_path=str(resolved),
        device="auto",
        batch_size=1,
        dtype="bfloat16",
        max_image_size=4096,
        concurrency=1,
    )
    backend = ChitrapathakBackend(settings, adapter_path=adapter.expanduser())
    try:
        predictions = []
        details = []
        for row in rows:
            image = images / str(row["image_filename"])
            prediction = backend.predict(image).text
            predictions.append(prediction)
            sample = sample_core_metrics(str(row["expected_text"]), prediction)
            details.append(
                {
                    "sample_id": row["sample_id"],
                    "expected_text": row["expected_text"],
                    "prediction": prediction,
                    "cer": sample.cer,
                }
            )
    finally:
        backend.close()
        _release_torch()
    metrics = score_corpus([str(row["expected_text"]) for row in rows], predictions)
    return {
        "adapter": str(adapter),
        "base_model": str(resolved),
        "base_source": source,
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "exact_match_rate": metrics["exact_match_rate"],
        "character_accuracy": metrics["character_accuracy"],
        "word_accuracy": metrics["word_accuracy"],
        "substitutions": metrics["substitutions"],
        "deletions": metrics["deletions"],
        "insertions": metrics["insertions"],
        "lines": details,
    }


def _log_wandb(scores: dict[str, Any]) -> None:
    try:
        import wandb
    except Exception as exc:
        print(f"wandb was not logged: {exc}")
        return
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    run = wandb.init(
        project="indic-ocr-pipeline",
        mode=mode,
        job_type="smoke",
        config={"smoke_test": True},
    )
    payload: dict[str, Any] = {}
    columns = ["model", "sample_id", "expected_text", "prediction", "cer"]
    table = wandb.Table(columns=columns)
    parseq = scores["models"].get("parseq") or {}
    for name in ("base", "finetuned"):
        block = parseq.get(name) or {}
        if "cer" in block:
            payload[f"parseq_{name}/cer"] = block["cer"]
            payload[f"parseq_{name}/wer"] = block["wer"]
            payload[f"parseq_{name}/exact_match_rate"] = block["exact_match_rate"]
        for line in block.get("lines") or []:
            table.add_data(*_table_row(f"parseq_{name}", line))
    lora = scores["models"].get("chitrapathak_lora") or {}
    metric_names = (
        "cer",
        "wer",
        "exact_match_rate",
        "character_accuracy",
        "word_accuracy",
        "substitutions",
        "deletions",
        "insertions",
    )
    for key in metric_names:
        if key in lora:
            payload[f"chitrapathak_lora/{key}"] = lora[key]
    for line in lora.get("lines") or []:
        table.add_data(*_table_row("chitrapathak_lora", line))
    payload["lines"] = table
    run.log(payload)
    run.finish()
    print(f"wandb: project indic-ocr-pipeline mode {mode}")


def _table_row(model: str, line: dict[str, Any]) -> tuple[str, str, str, str, float]:
    return (
        model,
        str(line["sample_id"]),
        str(line["expected_text"]),
        str(line["prediction"]),
        float(line["cer"]),
    )


def _print_scores(scores: dict[str, Any]) -> None:
    print("SMOKE SCORES")
    print("labels: human_verified for this test folder only")
    parseq = scores["models"].get("parseq")
    if parseq:
        print(
            f"parseq base CER {parseq['base']['cer']:.4f}  "
            f"finetuned CER {parseq['finetuned']['cer']:.4f}"
        )
    lora = scores["models"].get("chitrapathak_lora")
    if lora:
        print(
            f"chitrapathak lora CER {lora['cer']:.4f}  "
            f"WER {lora['wer']:.4f}  exact {lora['exact_match_rate']:.4f}"
        )
        print(f"official base: {lora['base_model']} ({lora['base_source']})")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _release_torch() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
