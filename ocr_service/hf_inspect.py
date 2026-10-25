"""Hugging Face repository inspection for model registration (HTTPS only)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from registry import ModelError, normalize_hf_ref

_HF_HTTPS = re.compile(
    r"^https://huggingface\.co/(?P<org>[^/]+)/(?P<name>[^/]+)(?:/(?P<rest>.*))?$",
    re.IGNORECASE,
)
_VLM_ARCH_MARKERS = (
    "vl",
    "vision",
    "image",
    "llava",
    "idefics",
    "paligemma",
    "mllama",
    "qwen2_5_vl",
    "qwen2_vl",
    "internvl",
    "phi3_v",
    "gemma3",
    "mistral3",
    "image-text",
    "imagetext",
)
_TEXT_ONLY_MARKERS = (
    "bert",
    "gpt2",
    "t5",
    "whisper",
    "wav2vec",
    "forcausallm",
    "causal_lm",
    "seq2seq",
)
_ADAPTER_MARKERS = ("adapter_config.json",)
_FULL_MARKERS = ("config.json",)
_WEIGHT_MARKERS = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
)


@dataclass
class RepoValidation:
    repo_id: str
    revision: str | None
    model_type: str  # full | adapter | unsupported
    family: str  # chitrapathak | indicocr | vlm | trocr | unknown
    subfolder: str | None = None
    base_repo_id: str | None = None
    base_revision: str | None = None
    architecture: str | None = None
    label_suggestion: str = ""
    languages: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    file_count: int = 0
    compatible: bool = False
    message: str = ""


def parse_hf_url(url: str) -> tuple[str, str | None]:
    """Return (repo_id, revision) from an https huggingface.co URL or org/name."""
    text = (url or "").strip()
    if not text:
        raise ModelError("Paste a Hugging Face link or org/name.")
    lowered = text.lower()
    if lowered.startswith("http://"):
        raise ModelError("Use an https://huggingface.co link.")
    if lowered.startswith("https://"):
        if "huggingface.co" not in lowered:
            raise ModelError("Only huggingface.co repository links are supported.")
        match = _HF_HTTPS.match(text.split("?")[0].strip())
        if not match:
            raise ModelError("Could not read that Hugging Face link.")
        repo_id = f"{match.group('org')}/{match.group('name')}"
        rest = (match.group("rest") or "").strip("/")
        revision = None
        if rest.startswith("tree/") or rest.startswith("resolve/"):
            parts = rest.split("/")
            if len(parts) >= 2:
                revision = parts[1]
        normalize_hf_ref(repo_id, what="repository")
        return repo_id, revision
    repo_id = normalize_hf_ref(text, what="repository")
    return repo_id, None


def validate_repository(
    url: str,
    *,
    revision: str | None = None,
    family_hint: str = "",
    base_override: str = "",
) -> RepoValidation:
    """Fetch public metadata and classify the repo (no weight download)."""
    repo_id, url_rev = parse_hf_url(url)
    chosen_rev = (revision or url_rev or "").strip() or None
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError as exc:
        raise ModelError("huggingface_hub is required for model validation.") from exc

    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=chosen_rev)
    except RepositoryNotFoundError:
        raise ModelError(
            "Repository unavailable. Check the link, or that the repo is public."
        ) from None
    except Exception as exc:
        raise ModelError(f"Could not reach Hugging Face: {exc}") from exc

    resolved_rev = getattr(info, "sha", None) or chosen_rev
    siblings = list(getattr(info, "siblings", None) or [])
    names = {s.rfilename for s in siblings if getattr(s, "rfilename", None)}
    size_bytes = sum(int(getattr(s, "size", 0) or 0) for s in siblings)

    result = RepoValidation(
        repo_id=repo_id,
        revision=str(resolved_rev) if resolved_rev else None,
        model_type="unsupported",
        family="unknown",
        size_bytes=size_bytes or None,
        file_count=len(names),
        label_suggestion=repo_id.split("/")[-1].replace("_", " ").replace("-", " ").title(),
    )

    if any(name.endswith(".traineddata") for name in names) or (
        "tesseract" in repo_id.lower() and "config.json" not in names
    ):
        result.model_type = "unsupported"
        result.message = (
            "This looks like a Tesseract traineddata repository. "
            "This app runs Hugging Face image-to-text (vision) models for page OCR, "
            "not Tesseract .traineddata files."
        )
        result.warnings.append(result.message)
        return result
    if "adapter_config.json" in names:
        result.model_type = "adapter"
        _fill_adapter_metadata(api, result, base_override=base_override)
    elif "config.json" in names and any(m in names for m in _WEIGHT_MARKERS):
        result.model_type = "full"
        _fill_full_metadata(api, result, names)
    elif "config.json" in names and any("weights/ocr" in n for n in names):
        result.model_type = "full"
        result.family = "indicocr"
        result.subfolder = "weights/ocr"
        _fill_full_metadata(api, result, names, subfolder="weights/ocr")
    elif "config.json" in names:
        _fill_full_metadata(api, result, names)
        if result.architecture and _is_text_only(result.architecture):
            result.message = (
                "This looks like a text-only model. "
                "Paste a Hugging Face vision / image-to-text model for document OCR."
            )
            result.warnings.append(result.message)
            return result
        if not any(m in names for m in _WEIGHT_MARKERS):
            result.message = (
                "Found config.json but no model weights in this repo. "
                "Use a full model repository or a LoRA adapter."
            )
            result.warnings.append(result.message)
            return result
        result.model_type = "full"
    else:
        result.message = (
            "Unsupported repository layout. Expected a vision model with config.json "
            "and weights, or a PEFT adapter."
        )
        result.warnings.append(result.message)
        return result

    hint = (family_hint or "").strip().lower()
    if hint in ("chitrapathak", "indicocr", "vlm", "trocr"):
        result.family = hint
    elif result.family == "unknown":
        result.family = _guess_family(result, names)

    if base_override.strip() and result.model_type == "adapter":
        try:
            result.base_repo_id = normalize_hf_ref(base_override.strip(), what="base model")
        except ModelError:
            pass
    result.compatible, result.message = _compatibility_verdict(result, hint)
    return result


def _fill_adapter_metadata(api: Any, result: RepoValidation, *, base_override: str) -> None:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", "").strip() or None
    try:
        path = hf_hub_download(
            result.repo_id,
            "adapter_config.json",
            revision=result.revision,
            token=token,
        )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        result.warnings.append("Could not read adapter_config.json from the repository.")
        return

    base_raw = (base_override or "").strip() or str(payload.get("base_model_name_or_path") or "").strip()
    if base_raw and not base_raw.startswith(("/", ".", "~")):
        try:
            result.base_repo_id = normalize_hf_ref(base_raw, what="base model")
        except ModelError:
            result.warnings.append(
                "The adapter names a local base path. Enter the Hugging Face base model below."
            )
    peft = str(payload.get("peft_type") or "LORA")
    if peft.upper() != "LORA":
        result.warnings.append(f"Adapter type {peft} may not be supported; only LoRA is tested.")
    result.architecture = "peft_lora"


def _fill_full_metadata(api: Any, result: RepoValidation, names: set[str], subfolder: str = "") -> None:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", "").strip() or None
    config_name = f"{subfolder}/config.json".strip("/") if subfolder else "config.json"
    if config_name not in names and "config.json" in names:
        config_name = "config.json"
    try:
        path = hf_hub_download(
            result.repo_id,
            config_name,
            revision=result.revision,
            token=token,
        )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        result.warnings.append("Could not read config.json from the repository.")
        return
    archs = payload.get("architectures") or []
    result.architecture = str(archs[0]) if archs else str(payload.get("model_type") or "")
    if _is_trocr_arch(result.architecture):
        result.family = "trocr"
    elif result.family == "unknown" and _config_has_vision(payload):
        result.family = "vlm"


def _is_trocr_arch(architecture: str) -> bool:
    return "visionencoderdecoder" in (architecture or "").lower()


def _guess_family(result: RepoValidation, names: set[str]) -> str:
    if result.model_type == "adapter":
        return "chitrapathak"
    arch = (result.architecture or "").lower()
    if _is_trocr_arch(result.architecture or ""):
        return "trocr"
    if "qwen" in arch and "vl" in arch:
        return "chitrapathak"
    if any("weights/ocr" in n for n in names):
        return "indicocr"
    if "bodhan" in result.repo_id.lower() or "indic-ocr" in result.repo_id.lower():
        return "indicocr"
    if _looks_like_vlm_arch(arch):
        return "vlm"
    return "vlm" if result.model_type == "full" else "unknown"


def _config_has_vision(payload: dict[str, Any]) -> bool:
    if payload.get("vision_config") or payload.get("image_token_id"):
        return True
    for key in ("vision_tower", "mm_vision_tower", "visual"):
        if key in payload:
            return True
    return False


def _looks_like_vlm_arch(architecture: str) -> bool:
    arch = (architecture or "").lower()
    if not arch:
        return False
    if _is_text_only(arch) or _is_trocr_arch(architecture):
        return False
    return any(marker in arch for marker in _VLM_ARCH_MARKERS)


def _is_text_only(architecture: str) -> bool:
    arch = (architecture or "").lower()
    return any(marker in arch for marker in _TEXT_ONLY_MARKERS)


def _compatibility_verdict(result: RepoValidation, family_hint: str) -> tuple[bool, str]:
    if result.model_type == "adapter":
        if not result.base_repo_id:
            return (
                False,
                "This LoRA adapter does not name a Hugging Face base model. "
                "Open Advanced options and set the base vision model repo id.",
            )
        base = result.base_repo_id
        return (
            True,
            f"LoRA adapter — will download and run on base model {base}.",
        )
    if result.model_type != "full":
        return False, result.message or "Unsupported repository."
    if _is_text_only(result.architecture or ""):
        return False, result.message or "Unsupported text-only architecture."
    if result.family == "trocr" or _is_trocr_arch(result.architecture or ""):
        return (
            True,
            "Microsoft TrOCR model — downloads and runs with the TrOCR line-reader pipeline "
            "(English handwriting; not for Marathi/Gujarati).",
        )
    if result.family == "indicocr" or family_hint == "indicocr":
        return True, "Full IndicOCR-style vision model — ready to download."
    arch = result.architecture or ""
    if result.family == "chitrapathak" or _looks_like_vlm_arch(arch):
        return (
            True,
            "Full vision-language model — will download and run for page OCR.",
        )
    if result.family == "vlm":
        return (
            True,
            "Vision model — will download and run using generic image-to-text loading.",
        )
    return (
        False,
        "Could not confirm this is a vision model. "
        "Use a Hugging Face image-to-text / document OCR model repository.",
    )

