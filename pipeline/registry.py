"""Resolve two to five OCR models. A local path is used before a Hugging Face id."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.weights import WeightError, resolve_weight

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_KINDS = {"indicocr": "indicocr", "chitrapathak": "indicocr", "tesseract": "tesseract"}
_ENV_PATH = {
    "indicocr": "INDICOCR_MODEL_PATH",
    "chitrapathak": "CHITRAPATHAK_MODEL_PATH",
    "tesseract": "TESSERACT_CMD",
}


@dataclass
class ModelSpec:
    """One configured OCR model after a local path or cache lookup."""

    name: str
    kind: str
    path: str
    repo: str | None
    revision: str
    prompt: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def parse_model_flags(slots: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    """Return model slots that the command actually set. Empty means the defaults."""
    chosen = [slot for slot in slots if any(slot.get(key) for key in ("name", "path", "hf"))]
    if not chosen:
        return []
    if not 2 <= len(chosen) <= 5:
        raise ValueError("use between 2 and 5 models")
    parsed: list[dict[str, str | None]] = []
    for index, slot in enumerate(chosen, start=1):
        name = (slot.get("name") or "").strip().lower().replace("-", "_") or f"model_{index}"
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"model name must be a short lowercase identifier: {name}")
        path = (slot.get("path") or "").strip() or None
        hf_id = (slot.get("hf") or "").strip() or None
        if path is None and hf_id is None:
            raise ValueError(f"{name} needs --model-{index}-path or --model-{index}-hf")
        parsed.append({"name": name, "path": path, "hf": hf_id, "kind": _kind(name)})
    return parsed


def default_models() -> list[dict[str, str | None]]:
    """The three adapters used when the command does not name models."""
    return [
        {"name": "indicocr", "path": None, "hf": None, "kind": "indicocr"},
        {"name": "chitrapathak", "path": None, "hf": None, "kind": "indicocr"},
        {"name": "tesseract", "path": None, "hf": None, "kind": "tesseract"},
    ]


def resolve_models(
    slots: list[dict[str, str | None]],
    cache: Path,
    *,
    prompt: str = "",
    environ: dict[str, str] | None = None,
) -> list[ModelSpec]:
    """Resolve each slot. An existing local path wins over a Hugging Face id."""
    env = environ or {}
    resolved: list[ModelSpec] = []
    for slot in slots:
        name = str(slot["name"])
        kind = str(slot.get("kind") or _kind(name))
        local = _local_choice(slot, name, env)
        remote = (slot.get("hf") or "").strip()
        explicit = (slot.get("path") or "").strip()
        if explicit and local is None and not remote:
            raise WeightError(f"model path does not exist: {explicit}")
        if local is not None:
            path, repo, revision = str(local), None, _revision_from_path(local)
        elif remote:
            saved = resolve_weight(remote, cache, expect="dir" if kind != "tesseract" else "file")
            path, repo, revision = str(saved), _repo(remote), _revision_from_path(saved)
        elif name in {"indicocr", "chitrapathak", "tesseract"}:
            fallback = env.get(_ENV_PATH.get(name, ""), "")
            if name == "tesseract" and not fallback:
                fallback = "tesseract"
            if fallback and Path(fallback).expanduser().exists():
                saved = Path(fallback).expanduser().resolve()
                path, repo, revision = str(saved), None, _revision_from_path(saved)
            else:
                path, repo, revision = fallback, None, "unconfigured"
        else:
            raise WeightError(f"{name} has no local path or Hugging Face id")
        resolved.append(
            ModelSpec(
                name=name,
                kind=kind,
                path=path,
                repo=repo,
                revision=revision or "local",
                prompt=prompt,
            )
        )
    return resolved


def write_resolved(path: Path, models: list[ModelSpec]) -> None:
    """Store the resolved models so later stages reuse the same revision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.to_dict() for item in models], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_resolved(path: Path) -> dict[str, dict[str, str | None]]:
    """Return resolved models keyed by name."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["name"]): item for item in payload if isinstance(item, dict)}


def _kind(name: str) -> str:
    if name in {"indicocr", "chitrapathak", "tesseract"}:
        return "indicocr" if name != "tesseract" else "tesseract"
    lowered = name.lower()
    if "tesseract" in lowered:
        return "tesseract"
    if "indicocr" in lowered or lowered in _KINDS:
        return "indicocr"
    return "transformers"


def _local_choice(
    slot: dict[str, str | None],
    name: str,
    env: dict[str, str],
) -> Path | None:
    raw = (slot.get("path") or "").strip()
    if not raw and name == "indicocr":
        raw = env.get("INDICOCR_MODEL_PATH", "")
    if not raw and name == "chitrapathak":
        raw = env.get("CHITRAPATHAK_MODEL_PATH", "")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    return None


def _repo(spec: str) -> str | None:
    text = spec[5:] if spec.startswith("hf://") else spec
    parts = [part for part in text.replace("https://huggingface.co/", "").split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def _revision_from_path(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local"
