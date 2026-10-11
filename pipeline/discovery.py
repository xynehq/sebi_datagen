"""Find local OCR checkpoints. This module does not download weights."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

_WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
_SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "data",
    "runs",
    "tests",
    "IndicPhotoOCR",
}
_MAX_DEPTH = 3


class DiscoveryError(ValueError):
    """A model could not be selected."""


@dataclass
class DiscoveredModel:
    """One recognizable checkpoint, base model, or PEFT adapter."""

    name: str
    path: str
    family: str
    role: str
    base_model: str | None
    languages: str
    revision: str
    complete: bool
    note: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def registry_path(root: Path) -> Path:
    return root / "models" / "registry.json"


def load_registry(root: Path) -> list[dict[str, str]]:
    path = registry_path(root)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict) and item.get("path")]


def register_model(root: Path, path: Path, *, name: str | None = None) -> None:
    """Remember a path. The weight files stay where they are."""
    rows = [item for item in load_registry(root) if item.get("path") != str(path)]
    rows.append({"name": name or path.name, "path": str(path)})
    destination = registry_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def unregister_model(root: Path, path: Path) -> None:
    """Drop a registry entry. The model files are not deleted."""
    rows = [item for item in load_registry(root) if item.get("path") != str(path)]
    registry_path(root).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def default_roots(root: Path, extra: list[Path] | None = None) -> list[Path]:
    roots = [root / "models" / "adapters", root / "models" / "cache"]
    roots.extend(extra or [])
    for item in load_registry(root):
        roots.append(Path(str(item["path"])))
    return roots


def huggingface_cache() -> Path | None:
    configured = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else None
    default = Path.home() / ".cache" / "huggingface" / "hub"
    return default if default.is_dir() else None


def discover(roots: list[Path], *, include_hf_cache: bool = True) -> list[DiscoveredModel]:
    """Inspect configured roots and the Hugging Face cache. No full-disk walk."""
    found: list[DiscoveredModel] = []
    seen: set[str] = set()
    for root in roots:
        for directory in _walk(root):
            model = inspect_path(directory)
            if model is None or model.path in seen or not model.complete:
                continue
            seen.add(model.path)
            found.append(model)
    if include_hf_cache:
        cache = huggingface_cache()
        if cache is not None:
            for snapshot in _hf_snapshots(cache):
                model = inspect_path(snapshot)
                if model is None or model.path in seen or not model.complete:
                    continue
                seen.add(model.path)
                found.append(model)
    return found


def inspect_path(path: Path) -> DiscoveredModel | None:
    """Classify one file or directory. A lone weight file is not a model."""
    if not path.exists():
        return None
    if path.is_file() and path.suffix.lower() == ".ckpt":
        return _model(path, "parseq", "checkpoint", None, "PARSeq checkpoint")
    if path.is_file():
        return None
    adapter = path / "adapter_config.json"
    if adapter.is_file():
        return _adapter(path)
    config = path / "config.json"
    if (path / "weights" / "ocr" / "config.json").is_file():
        return _model(path, "indicocr", "base", None, "IndicOCR snapshot")
    if config.is_file():
        weights = _has_weights(path)
        if not weights:
            return _model(
                path, "transformers", "base", None, "config.json without weights", ok=False
            )
        return _model(path, "transformers", "base", None, "Transformers model directory")
    return None


def resolve_explicit(path: Path) -> DiscoveredModel:
    """Use the path the user named. Do not substitute another model."""
    model = inspect_path(path)
    if model is None or not model.complete:
        raise DiscoveryError(
            f"{path} is not a complete model. Expected a PEFT adapter, a Transformers "
            "directory with config.json and weights, an IndicOCR snapshot, or a .ckpt file."
        )
    return model


def format_catalog(models: list[DiscoveredModel]) -> str:
    if not models:
        return "No complete models found in the configured search roots."
    lines = ["#  name  family  role  languages  path"]
    for index, model in enumerate(models, start=1):
        base = f" base={model.base_model}" if model.base_model else ""
        lines.append(
            f"{index}  {model.name}  {model.family}  {model.role}  "
            f"languages={model.languages}  {model.path}{base}"
        )
    return "\n".join(lines)


def resolve_adapter_base(
    model: DiscoveredModel, override: Path | None = None
) -> tuple[Path, str]:
    """Return the local base directory for a PEFT adapter.

    The path recorded in the adapter is used when it already contains
    ``config.json``. Otherwise ``override`` is the configured Chitrapathak base.
    """
    if model.family != "peft" or model.role != "adapter":
        raise DiscoveryError(f"{model.path} is not a PEFT adapter")
    if not model.base_model:
        raise DiscoveryError(f"PEFT adapter {model.path} has no base_model_name_or_path")
    recorded = Path(model.base_model)
    if (recorded / "config.json").is_file():
        return recorded.resolve(), "adapter"
    if override is not None and (override / "config.json").is_file():
        return override.resolve(), "configured"
    raise DiscoveryError(
        f"PEFT adapter {model.name} records base model {model.base_model}, "
        "which is not a local directory containing config.json. "
        "Pass the official base with --chitrapathak. The adapter was not modified."
    )


def record_peft_evaluation(
    model: DiscoveredModel,
    output: Path,
    base_override: Path | None = None,
) -> dict[str, str | bool]:
    """Record an adapter and its base. Does not load weights or run PARSeq."""
    base, source = resolve_adapter_base(model, base_override)
    payload: dict[str, str | bool] = {
        "family": model.family,
        "role": model.role,
        "adapter": model.path,
        "base_model": str(base),
        "base_source": source,
        "adapter_base_record": model.base_model or "",
        "languages": model.languages,
        "backend": "chitrapathak",
        "parseq": False,
        "weights_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def select_from_catalog(
    models: list[DiscoveredModel],
    *,
    explicit: Path | None = None,
    interactive: bool = False,
    choice: str | None = None,
) -> DiscoveredModel:
    """Pick one model. An explicit path is never replaced."""
    if explicit is not None:
        return resolve_explicit(explicit)
    if choice:
        return _choice(models, choice)
    if not interactive:
        raise DiscoveryError(
            "No model was selected. Pass --model-path or --model-hf, or run with a terminal "
            "so a model can be chosen from --list-models."
        )
    print(format_catalog(models))
    print("Enter a number, or a local path.")
    answer = input("model: ").strip()
    if not answer:
        raise DiscoveryError("No model was selected.")
    picked = Path(answer).expanduser()
    if picked.exists():
        return resolve_explicit(picked)
    return _choice(models, answer)


def _choice(models: list[DiscoveredModel], answer: str) -> DiscoveredModel:
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(models):
            return models[index - 1]
    for model in models:
        if model.name == answer or model.path == answer:
            return model
    raise DiscoveryError(f"unknown model selection: {answer}")


def _adapter(path: Path) -> DiscoveredModel:
    payload = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    base = payload.get("base_model_name_or_path")
    base_text = str(base) if base else None
    weight_names = ("adapter_model.safetensors", "adapter_model.bin")
    weights = any((path / name).is_file() for name in weight_names)
    language = payload.get("language")
    languages = str(language) if isinstance(language, str) and language else "unknown"
    note = "PEFT adapter"
    if not weights:
        note = "adapter_config.json without adapter weights"
    return _model(path, "peft", "adapter", base_text, note, ok=weights, languages=languages)


def _model(
    path: Path,
    family: str,
    role: str,
    base: str | None,
    note: str,
    *,
    ok: bool = True,
    languages: str = "unknown",
) -> DiscoveredModel:
    return DiscoveredModel(
        name=path.name,
        path=str(path.resolve()),
        family=family,
        role=role,
        base_model=base,
        languages=languages,
        revision=_revision(path),
        complete=ok,
        note=note,
    )


def _has_weights(path: Path) -> bool:
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in _WEIGHTS:
            return True
        if item.name.endswith(".index.json"):
            return True
    return False


def _revision(path: Path) -> str:
    parts = path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local"


def _walk(root: Path) -> list[Path]:
    if not root.exists():
        return []
    has_config = (root / "config.json").is_file() or (root / "adapter_config.json").is_file()
    if root.is_file() or has_config:
        return [root]
    found = [root]
    for directory, dirnames, filenames in os.walk(root):
        current = Path(directory)
        depth = len(current.relative_to(root).parts)
        dirnames[:] = [name for name in dirnames if name not in _SKIP and not name.startswith(".")]
        if depth >= _MAX_DEPTH:
            dirnames[:] = []
        found.extend(current / name for name in dirnames)
        found.extend(current / name for name in filenames if name.endswith(".ckpt"))
    return found


def _hf_snapshots(cache: Path) -> list[Path]:
    hub = cache if cache.name == "hub" else cache / "hub"
    if not hub.is_dir():
        hub = cache
    snapshots: list[Path] = []
    for repo in hub.glob("models--*"):
        snap_root = repo / "snapshots"
        if not snap_root.is_dir():
            continue
        snapshots.extend(path for path in snap_root.iterdir() if path.is_dir())
    return snapshots
