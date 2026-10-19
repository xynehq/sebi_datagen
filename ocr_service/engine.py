"""Load one Chitrapathak model at a time and read a page image."""

from __future__ import annotations

import gc
import threading
from collections.abc import Iterator

from PIL import Image

from reading import CHITRA_PROMPT, clean_reading
from registry import ModelError, ModelSpec, ensure_weights, get_spec

_CHITRA_PROMPT = CHITRA_PROMPT
_CHITRA_REPETITION_PENALTY = 1.05
_INDIC_PROMPT = (
    "Transcribe the text in this image. Write any mathematical expressions in LaTeX, "
    "using $...$ for inline math and $$...$$ for display equations."
)
_INDIC_EOS = 262146
_INDIC_MAX_SIDE = 1536


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Engine:
    """LRU of one model. Switching models unloads the previous weights."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = pick_device()
        self._loaded_id: str | None = None
        self._model = None
        self._processor = None
        self._system: str | None = None
        self._prompt = _CHITRA_PROMPT
        self._eos: int | None = None
        self._max_side: int | None = None
        self._repetition_penalty = 1.0

    @property
    def device(self) -> str:
        return self._device

    def read(
        self,
        image: Image.Image,
        *,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        text = ""
        for chunk in self.stream(
            image,
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ):
            text = chunk
        return text

    def stream(
        self,
        image: Image.Image,
        *,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
    ) -> Iterator[str]:
        spec = get_spec(model_id)
        with self._lock:
            self._ensure_loaded(spec)
            yield from self._generate(
                image.convert("RGB"),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

    def _ensure_loaded(self, spec: ModelSpec) -> None:
        if self._loaded_id == spec.id and self._model is not None:
            return
        self._unload()
        base_path, adapter_path = ensure_weights(spec)
        self._model, self._processor = _load_model(
            base_path,
            adapter_path,
            device=self._device,
            subfolder=spec.subfolder,
        )
        if spec.family == "indicocr":
            self._system = None
            self._prompt = _INDIC_PROMPT
            self._eos = _INDIC_EOS
            self._max_side = _INDIC_MAX_SIDE
            self._repetition_penalty = 1.0
        else:
            self._system = "You are a helpful assistant."
            self._prompt = _CHITRA_PROMPT
            self._eos = None
            self._max_side = None
            self._repetition_penalty = _CHITRA_REPETITION_PENALTY
        self._loaded_id = spec.id

    def _unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded_id = None
        gc.collect()
        _release(self._device)

    def _generate(
        self,
        image: Image.Image,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        processor = self._processor
        model = self._model
        image = _fit_side(image, self._max_side)
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._prompt},
            ],
        }
        messages = [user]
        if self._system:
            messages.insert(0, {"role": "system", "content": self._system})
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        do_sample = temperature > 0
        kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 0.9
        if self._repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self._repetition_penalty
        if self._eos is not None:
            kwargs["eos_token_id"] = self._eos
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs["streamer"] = streamer
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**kwargs)
            except Exception as exc:
                errors.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        pieces: list[str] = []
        for piece in streamer:
            pieces.append(piece)
            yield _clean("".join(pieces))
        thread.join()
        if errors:
            raise ModelError(f"OCR failed: {errors[0]}") from errors[0]
        if not pieces:
            yield ""


def _load_model(base_path, adapter_path, *, device: str, subfolder: str | None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    attn = "eager" if device == "cpu" else "sdpa"
    extra = {"subfolder": subfolder} if subfolder else {}
    processor = AutoProcessor.from_pretrained(base_path, local_files_only=True, **extra)
    model = AutoModelForImageTextToText.from_pretrained(
        base_path,
        dtype=dtype,
        attn_implementation=attn,
        local_files_only=True,
        **extra,
    )
    model.to(device)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.to(device)
    model.eval()
    return model, processor


def _release(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _fit_side(image: Image.Image, limit: int | None) -> Image.Image:
    if limit is None:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= limit:
        return image
    scale = limit / longest
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def _clean(text: str) -> str:
    return clean_reading(text, _CHITRA_PROMPT, _INDIC_PROMPT)
