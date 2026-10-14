"""Local Chitrapathak adapter.

Weights come from ``CHITRAPATHAK_MODEL_PATH`` or ``configs/ocr.yaml``. This
module does not download them and does not call a hosted API. The transcription
is a candidate, not a gold label.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

from indic_ocr_pipeline.config import BackendSettings
from indic_ocr_pipeline.schemas import OCRPrediction
from indic_ocr_pipeline.text_policy import apply_storage_normalization
from models.IndicPhotoOCR.backend import _offline_hub, _release_device, _require_device
from models.tesseract.backend import OCRError, _reject_oversized

_PROMPT = (
    "Perform OCR on this image and transcribe all visible text exactly as it appears."
)
_LINE_TOKENS = 512


class ChitrapathakBackend:
    """Chitrapathak on one full-line crop. One image at a time."""

    name = "chitrapathak"

    def __init__(self, settings: BackendSettings, adapter_path: Path | None = None) -> None:
        if not settings.model_path:
            raise OCRError(
                "chitrapathak model_path is empty. "
                "Set CHITRAPATHAK_MODEL_PATH to a local model directory."
            )
        model_path = Path(settings.model_path)
        if not (model_path / "config.json").is_file():
            raise OCRError(f"chitrapathak weights were not found at {model_path}")
        self._max_image_size = settings.max_image_size
        self._device = _require_device(settings.device)
        self._dtype = settings.dtype
        self._model, self._processor = _load(
            model_path, self._device, self._dtype, adapter_path
        )
        self._logged_image = False
        self.version = "chitrapathak-2" if adapter_path is None else "chitrapathak-lora"
        _log_inference_path(model_path)

    def predict(self, image_path: Path) -> OCRPrediction:
        import torch
        from PIL import Image

        _reject_oversized(image_path, self._max_image_size)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").copy()
        if not self._logged_image:
            self._logged_image = True
            _log_crop(image_path, rgb.size)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": _PROMPT},
                ],
            },
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[prompt], images=[rgb], padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=_LINE_TOKENS,
                do_sample=False,
            )
        raw_text = self._processor.batch_decode(
            output[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return OCRPrediction(
            backend=self.name,
            backend_version=self.version,
            text=apply_storage_normalization(raw_text),
            raw_text=raw_text,
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        _release_device(self._device)


def _log_inference_path(model_path: Path) -> None:
    """Record that this repo does not call official ``eval_model``."""
    from indic_ocr_pipeline.utils.logging import get_logger

    message = (
        "Chitrapathak inference is local transformers AutoProcessor.generate "
        f"on one RGB line crop (max_new_tokens={_LINE_TOKENS}, weights={model_path}). "
        "Official chitrapathak.inference.eval_model is not called; that entry point "
        "uses image_processor.preprocess(do_resize=False, do_rescale=False, "
        "do_center_crop=False, do_normalize=False) and max_new_tokens=3000."
    )
    print(message, flush=True)
    get_logger(__name__).info(message)


def _log_crop(image_path: Path, size: tuple[int, int]) -> None:
    from indic_ocr_pipeline.utils.logging import get_logger

    width, height = size
    message = (
        f"Chitrapathak first crop: {image_path.name} "
        f"{width}x{height} RGB, already cropped from a rendered PDF page."
    )
    print(message, flush=True)
    get_logger(__name__).info(message)


def _load(
    model_path: Path,
    device: str,
    dtype_name: str,
    adapter_path: Path | None = None,
) -> tuple[object, object]:
    _offline_hub()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise OCRError(f"chitrapathak dtype is not a torch dtype: {dtype_name}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=_device_map(device),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if adapter_path is not None:
        if not (adapter_path / "adapter_config.json").is_file():
            raise OCRError(f"Chitrapathak LoRA adapter was not found at {adapter_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, processor


def _device_map(device: str) -> str | dict[str, str]:
    if device == "mps":
        return {"": "mps"}
    return device
