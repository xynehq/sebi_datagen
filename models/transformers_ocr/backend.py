"""Image-to-text adapter for an extra model that is not IndicOCR or Tesseract."""

from __future__ import annotations

from pathlib import Path

from indic_ocr_pipeline.schemas import OCRPrediction


class TransformersOCRBackend:
    """Load one cached vision-language model and transcribe a single crop."""

    def __init__(self, spec: dict[str, str | None]) -> None:
        from indic_ocr_pipeline.ocr.tesseract import OCRError

        self.name = str(spec.get("name") or "transformers")
        path = spec.get("path")
        if not path or not Path(path).exists():
            raise OCRError(f"{self.name} model path does not exist")
        self.version = str(spec.get("revision") or "local")
        self.prompt = str(spec.get("prompt") or "")
        self._path = str(path)
        self._model = None
        self._processor = None
        self._error = OCRError

    def predict(self, image_path: Path) -> OCRPrediction:
        model, processor = self._load()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        if self.prompt and hasattr(processor, "tokenizer"):
            tokens = processor.tokenizer(self.prompt, return_tensors="pt")
            inputs.update({key: value for key, value in tokens.items() if key not in inputs})
        generated = model.generate(**inputs)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return OCRPrediction(
            backend=self.name,
            backend_version=self.version,
            text=text.strip(),
            raw_text=text,
            metadata={"revision": self.version, "prompt": self.prompt, "model_path": self._path},
        )

    def close(self) -> None:
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise self._error("transformers is not installed") from exc
        self._processor = AutoProcessor.from_pretrained(self._path, local_files_only=True)
        self._model = AutoModelForVision2Seq.from_pretrained(self._path, local_files_only=True)
        self._model.eval()
        return self._model, self._processor
