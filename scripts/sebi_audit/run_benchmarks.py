#!/usr/bin/env python3
"""SEBI OCR audit benchmarks — does not modify production code.

Runs against local Engine (same code as ocr_service) or optional API base URL.
Default: direct Engine on this machine. Set OCR_AUDIT_API_BASE to hit HTTP API.

Outputs: docs/sebi_audit_benchmark_results.json (consumed by build_report.py)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OCR_DIR = REPO / "ocr_service"
sys.path.insert(0, str(OCR_DIR))

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

OUT = REPO / "docs" / "sebi_audit_benchmark_results.json"


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _gpu_note() -> dict:
    try:
        import torch

        out = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            out["cuda_device_name"] = torch.cuda.get_device_name(0)
            out["cuda_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            out["cuda_memory_allocated_bytes"] = torch.cuda.memory_allocated(0)
            out["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        return out
    except Exception as exc:
        return {"error": str(exc)}


def _dominant_script(text: str) -> str:
    dev = guj = 0
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o < 0x0980:
            dev += 1
        elif 0x0A80 <= o < 0x0B00:
            guj += 1
    if dev == 0 and guj == 0:
        return "none"
    return "devanagari" if dev >= guj else "gujarati"


@dataclass
class RequestResult:
    model_id: str
    image: str
    success: bool
    latency_s: float
    error: str | None
    output_chars: int
    dominant_script: str
    rss_mb_after: float | None
    gpu_snapshot: dict | None


def _run_engine_once(engine, image_path: Path, model_id: str, max_tokens: int = 512) -> RequestResult:
    from PIL import Image

    t0 = time.perf_counter()
    err = None
    text = ""
    try:
        image = Image.open(image_path).convert("RGB")
        for chunk in engine.stream(
            image,
            model_id=model_id,
            temperature=0.0,
            max_new_tokens=max_tokens,
        ):
            text = chunk
    except Exception as exc:
        err = str(exc)
    lat = time.perf_counter() - t0
    if torch := __import__("torch", fromlist=["cuda"]):
        gpu = _gpu_note() if torch.cuda.is_available() or (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ) else None
    else:
        gpu = None
    return RequestResult(
        model_id=model_id,
        image=str(image_path.name),
        success=err is None and bool(text.strip()),
        latency_s=lat,
        error=err,
        output_chars=len(text),
        dominant_script=_dominant_script(text),
        rss_mb_after=_rss_mb(),
        gpu_snapshot=gpu,
    )


def _api_ocr(base: str, image_path: Path, model_id: str, max_tokens: int = 512) -> RequestResult:
    import mimetypes

    t0 = time.perf_counter()
    err = None
    text = ""
    try:
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body_parts = []
        for key, val in [
            ("model", model_id),
            ("page", "1"),
            ("dpi", "200"),
            ("temperature", "0"),
            ("max_tokens", str(max_tokens)),
        ]:
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n")
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        data = image_path.read_bytes()
        body_parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{image_path.name}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n".encode()
            if False
            else ""
        )
        # build multipart properly
        import io

        buf = io.BytesIO()
        for key, val in [
            ("model", model_id),
            ("page", "1"),
            ("dpi", "200"),
            ("temperature", "0"),
            ("max_tokens", str(max_tokens)),
        ]:
            buf.write(f"--{boundary}\r\n".encode())
            buf.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            buf.write(f"{val}\r\n".encode())
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode()
        )
        buf.write(f"Content-Type: {mime}\r\n\r\n".encode())
        buf.write(data)
        buf.write(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"{base.rstrip('/')}/v1/ocr",
            data=buf.getvalue(),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        api_key = os.environ.get("OCR_API_KEY", "").strip()
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("ngrok-skip-browser-warning", "true")
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
        text = str(payload.get("text") or "")
    except Exception as exc:
        err = str(exc)
    lat = time.perf_counter() - t0
    return RequestResult(
        model_id=model_id,
        image=str(image_path.name),
        success=err is None and bool(text.strip()),
        latency_s=lat,
        error=err,
        output_chars=len(text),
        dominant_script=_dominant_script(text),
        rss_mb_after=_rss_mb(),
        gpu_snapshot=None,
    )


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def main() -> int:
    api_base = os.environ.get("OCR_AUDIT_API_BASE", "").strip()
    max_tokens = int(os.environ.get("OCR_AUDIT_MAX_TOKENS", "512"))
    marathi_img = OCR_DIR / "examples" / "marathi_sample.png"
    gujarati_img = OCR_DIR / "examples" / "gujarati_sample.png"
    if not marathi_img.is_file() or not gujarati_img.is_file():
        print("Missing example images", file=sys.stderr)
        return 2

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
        },
        "mode": "http_api" if api_base else "direct_engine",
        "api_base": api_base or None,
        "max_tokens_per_request": max_tokens,
        "notes": [],
    }

    if psutil is None:
        report["notes"].append("psutil not installed; RSS memory omitted")

    memory_stages: list[dict] = []
    memory_stages.append({"stage": "process_start", "rss_mb": _rss_mb(), "gpu": _gpu_note()})

    sequential: list[dict] = []
    switch_sequence = [
        ("chitrapathak-2", marathi_img, "marathi_base"),
        ("chitrapathak-marguj", gujarati_img, "gujarati_marguj"),
        ("chitrapathak-2", marathi_img, "marathi_base_repeat"),
        ("chitrapathak-marguj-mixed", gujarati_img, "gujarati_mixed"),
    ]

    engine = None
    loaded_ids: list[str] = []
    if not api_base:
        from engine import Engine

        engine = Engine()
        report["engine_device"] = engine.device
        memory_stages.append({"stage": "engine_created", "rss_mb": _rss_mb(), "gpu": _gpu_note()})

    for model_id, img, tag in switch_sequence:
        if api_base:
            res = _api_ocr(api_base, img, model_id, max_tokens=max_tokens)
        else:
            assert engine is not None
            before_loaded = getattr(engine, "_loaded_id", None)
            res = _run_engine_once(engine, img, model_id, max_tokens=max_tokens)
            after_loaded = getattr(engine, "_loaded_id", None)
            if before_loaded != after_loaded:
                loaded_ids.append(f"{before_loaded}->{after_loaded}")
        sequential.append({**asdict(res), "tag": tag})
        memory_stages.append(
            {"stage": f"after_{tag}", "rss_mb": _rss_mb(), "gpu": _gpu_note(), "model_id": model_id}
        )

    report["sequential_adapter_switch"] = sequential
    report["observed_engine_reloads"] = loaded_ids

    # Concurrency (direct engine only — demonstrates lock serialization)
    concurrency_results: dict[str, list] = {}
    if engine is not None and not api_base:
        for level in (1, 2, 4):
            if level > 1:
                report["notes"].append(
                    f"concurrency_{level}: Engine holds lock for full generate(); expect serialization"
                )
            tasks = []
            for i in range(level):
                mid = "chitrapathak-2" if i % 2 == 0 else "chitrapathak-marguj-mixed"
                img = marathi_img if i % 2 == 0 else gujarati_img
                tasks.append((mid, img))
            started = time.perf_counter()
            results: list[RequestResult] = []
            with ThreadPoolExecutor(max_workers=level) as pool:
                futs = [pool.submit(_run_engine_once, engine, img, mid, max_tokens) for mid, img in tasks]
                for fut in as_completed(futs):
                    results.append(fut.result())
            wall = time.perf_counter() - started
            lats = [r.latency_s for r in results if r.success]
            concurrency_results[str(level)] = {
                "wall_time_s": wall,
                "results": [asdict(r) for r in results],
                "success": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "median_latency_s": statistics.median(lats) if lats else None,
                "p95_latency_s": _percentile(lats, 95),
            }
        memory_stages.append({"stage": "after_concurrency", "rss_mb": _rss_mb(), "gpu": _gpu_note()})

    report["concurrency"] = concurrency_results
    report["memory_stages"] = memory_stages

    # Single-page latency repetitions (mixed model, gujarati sample)
    reps = int(os.environ.get("OCR_AUDIT_LATENCY_REPS", "3"))
    lat_runs: list[float] = []
    model = "chitrapathak-marguj-mixed"
    for _ in range(reps):
        if api_base:
            r = _api_ocr(api_base, gujarati_img, model, max_tokens=max_tokens)
        else:
            assert engine is not None
            r = _run_engine_once(engine, gujarati_img, model, max_tokens=max_tokens)
        if r.success:
            lat_runs.append(r.latency_s)
    report["latency_one_page"] = {
        "model_id": model,
        "image": gujarati_img.name,
        "repetitions": reps,
        "successful": len(lat_runs),
        "latencies_s": lat_runs,
        "median_s": statistics.median(lat_runs) if lat_runs else None,
        "p95_s": _percentile(lat_runs, 95),
        "note": "API processes one page per request; multi-page PDF requires multiple calls",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
