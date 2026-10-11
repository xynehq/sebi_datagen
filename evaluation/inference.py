"""PARSeq checkpoint loading and inference for Marathi OCR evaluation."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms as T

from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint


@dataclass
class DeviceInfo:
    device: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]


@dataclass
class InferenceResult:
    prediction: str
    latency_ms: float
    inference_error: bool
    error_message: Optional[str] = None


def resolve_device(requested: str) -> DeviceInfo:
    """Resolve --device auto|cuda|mps|cpu. Explicit cuda/mps fail if unavailable."""
    from indic_ocr_pipeline.device import DeviceError, select_device

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
    gpu_name = None
    gpu_memory_gb = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            pass
    try:
        device = select_device(requested)
    except DeviceError as exc:
        raise RuntimeError(str(exc)) from exc
    if device == "mps":
        gpu_name = gpu_name or "Apple MPS"

    return DeviceInfo(
        device=device,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


def print_device_banner(info: DeviceInfo, file=None) -> None:
    out = file or sys.stderr
    print(f"PyTorch version: {info.torch_version}", file=out)
    print(f"CUDA available: {info.cuda_available}", file=out)
    print(f"CUDA version: {info.cuda_version}", file=out)
    print(f"GPU name: {info.gpu_name}", file=out)
    print(f"GPU memory (GB): {info.gpu_memory_gb}", file=out)
    print(f"Device selected: {info.device}", file=out)
    if info.device.startswith("cuda") and info.gpu_name:
        print(f"Device: {info.device}", file=out)
        print(f"GPU: {info.gpu_name}", file=out)


def load_model(checkpoint_path: str, device: str):
    model = load_from_checkpoint(checkpoint_path)
    model = model.eval().to(device)
    return model


def _build_transform(model):
    return T.Compose([
        T.Resize(model.hparams.img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])


def _decode_probs(model, probs) -> List[str]:
    preds, _ = model.tokenizer.decode(probs)
    return [model.charset_adapter(p).strip() for p in preds]


def recognise_canonical(
    model, device: str, image_path: str, language: str = "marathi"
) -> InferenceResult:
    """Canonical frozen path: piecewise AR=3, boundary-aware join, zero overlap."""
    from evaluation.canonical import canonical_inference_settings
    from evaluation.piecewise import piecewise_result_to_inference, recognise_piecewise

    settings = canonical_inference_settings(language=language)
    img_size = getattr(getattr(model, "hparams", None), "img_size", None)
    piece = recognise_piecewise(
        model,
        device,
        image_path,
        target_aspect=settings["piece_target_aspect"],
        model_img_size=img_size,
        join_mode=settings["join_mode"],
        overlap_px=settings["piece_overlap_px"],
    )
    return piecewise_result_to_inference(piece)


def recognise_one(
    model,
    device: str,
    image_path: str,
    transform=None,
    use_tta: bool = False,
) -> InferenceResult:
    """Single-image inference; catches exceptions without skipping."""
    t0 = time.perf_counter()
    try:
        if transform is None:
            transform = _build_transform(model)
        img = Image.open(image_path).convert("RGB")
        if use_tta:
            # Reuse existing TTA from scripts/eval_tta.py
            scripts_dir = str(Path(__file__).resolve().parents[1])
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from eval_tta import tta_recognise  # type: ignore

            text = tta_recognise(model, device, img, transform).strip()
        else:
            with torch.no_grad():
                logits = model(transform(img).unsqueeze(0).to(device))
                probs = logits.softmax(-1)
                texts = _decode_probs(model, probs)
                text = texts[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            prediction=text,
            latency_ms=latency_ms,
            inference_error=False,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            prediction="",
            latency_ms=latency_ms,
            inference_error=True,
            error_message=f"{type(e).__name__}: {e}",
        )


def recognise_batch(
    model,
    device: str,
    image_paths: Sequence[str],
    batch_size: int = 1,
    use_tta: bool = False,
) -> Tuple[List[InferenceResult], bool]:
    """Run inference preserving dataset order.

    Returns (results, used_batch_fallback_to_1).
    If batch_size > 1 and TTA is on, falls back to B=1 (TTA averages variants).
    """
    transform = _build_transform(model)
    results: List[InferenceResult] = [None] * len(image_paths)  # type: ignore
    used_fallback = False

    if use_tta or batch_size <= 1:
        if batch_size > 1 and use_tta:
            used_fallback = True
        for i, path in enumerate(image_paths):
            results[i] = recognise_one(
                model, device, path, transform=transform, use_tta=use_tta
            )
        return results, used_fallback

    # Batched path: fixed img_size → stack tensors
    try:
        for start in range(0, len(image_paths), batch_size):
            chunk_paths = list(image_paths[start:start + batch_size])
            t0 = time.perf_counter()
            try:
                tensors = []
                for p in chunk_paths:
                    img = Image.open(p).convert("RGB")
                    tensors.append(transform(img))
                batch = torch.stack(tensors).to(device)
                with torch.no_grad():
                    logits = model(batch)
                    probs = logits.softmax(-1)
                    texts = _decode_probs(model, probs)
                elapsed = (time.perf_counter() - t0) * 1000.0
                per = elapsed / max(len(chunk_paths), 1)
                for j, text in enumerate(texts):
                    results[start + j] = InferenceResult(
                        prediction=text,
                        latency_ms=per,
                        inference_error=False,
                    )
            except Exception:
                # Deterministic B=1 fallback for this chunk
                used_fallback = True
                for j, p in enumerate(chunk_paths):
                    results[start + j] = recognise_one(
                        model, device, p, transform=transform, use_tta=False
                    )
    except Exception:
        used_fallback = True
        for i, path in enumerate(image_paths):
            if results[i] is None:
                results[i] = recognise_one(
                    model, device, path, transform=transform, use_tta=False
                )

    for i, r in enumerate(results):
        if r is None:
            results[i] = InferenceResult(
                prediction="",
                latency_ms=0.0,
                inference_error=True,
                error_message="missing_result",
            )
    return results, used_fallback
