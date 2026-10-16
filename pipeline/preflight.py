"""Environment checks that run before expensive work."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.records import records_dir


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def collect_checks(root: Path, config: dict[str, Any], *, fixture: bool = False) -> list[Check]:
    runtime = config.get("runtime") or {}
    checks: list[Check] = []
    version = sys.version_info
    checks.append(
        Check(
            "python",
            version >= (3, 11),
            f"{version.major}.{version.minor}.{version.micro}",
        )
    )
    for module in ("yaml", "PIL", "numpy"):
        checks.append(Check(module, importlib.util.find_spec(module) is not None, module))
    torch_ok = importlib.util.find_spec("torch") is not None
    checks.append(
        Check("torch", torch_ok, "installed" if torch_ok else "not installed", required=False)
    )
    tesseract = shutil.which("tesseract")
    require_tess = bool(runtime.get("require_tesseract", False)) and not fixture
    checks.append(
        Check(
            "tesseract",
            bool(tesseract) or not require_tess,
            tesseract or "not on PATH",
            required=require_tess,
        )
    )
    gpu = _gpu_status()
    require_gpu = bool(runtime.get("require_gpu", False)) and not fixture
    checks.append(Check("gpu", gpu[0] or not require_gpu, gpu[1], required=require_gpu))
    for relative in (
        "configs/pipeline.yaml",
        "configs/dataset.yaml",
        "configs/training.yaml",
        "configs/evaluation.yaml",
    ):
        path = root / relative
        checks.append(Check(relative, path.is_file(), str(path)))
    frozen = root / (config.get("paths") or {}).get("frozen_benchmark", "data/eval_data")
    validation = frozen / "validation.jsonl"
    present = validation.is_file()
    checks.append(
        Check(
            "frozen benchmark",
            present,
            str(validation) if present else f"not checked until the frozen stage: {validation}",
            required=False,
        )
    )
    source_names = [
        (config.get("source") or {}).get("documents_dir", "data/documents"),
        config.get("source_dir", "input/pdfs"),
        config.get("also_scan", "data/documents"),
    ]
    pdfs = []
    for name in source_names:
        source = root / str(name)
        if source.is_dir():
            pdfs.extend(source.glob("*.pdf"))
    documents = records_dir(root) / "documents.jsonl"
    have_source = bool(pdfs) or documents.is_file() or fixture
    checks.append(
        Check(
            "source documents",
            have_source,
            f"{len(pdfs)} pdf files; manifest={'yes' if documents.is_file() else 'no'}",
        )
    )
    runs = root / "runs"
    try:
        runs.mkdir(parents=True, exist_ok=True)
        probe = runs / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
        detail = str(runs)
    except OSError as exc:
        writable = False
        detail = str(exc)
    checks.append(Check("writable runs/", writable, detail))
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    checks.append(Check("disk space", free_gb >= 1.0 or fixture, f"{free_gb:.1f} GB free"))
    model_required = bool(runtime.get("require_ocr_models", False)) and not fixture
    for env_name in ("INDICOCR_MODEL_PATH", "CHITRAPATHAK_MODEL_PATH", "CHECKPOINT"):
        import os

        value = os.environ.get(env_name, "").strip()
        exists = bool(value) and Path(value).exists()
        checks.append(
            Check(
                env_name,
                exists or not model_required,
                value or "unset",
                required=model_required,
            )
        )
    return checks


def _gpu_status() -> tuple[bool, str]:
    try:
        from indic_ocr_pipeline.device import describe_device
    except Exception:
        return False, "torch not installed"
    detail = describe_device("auto")
    return (not detail.startswith("cpu"), detail)


def format_checks(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        mark = "ok" if check.ok or not check.required else "FAIL"
        if not check.ok and not check.required:
            mark = "warn"
        lines.append(f"  {mark:4} {check.name}: {check.detail}")
    return "\n".join(lines)


def checks_pass(checks: list[Check]) -> bool:
    return all(check.ok or not check.required for check in checks)
