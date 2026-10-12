"""Resolve a local weight path, or a Hugging Face link passed on the command."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

_FILE_SUFFIXES = {".ckpt", ".pt", ".pth", ".bin", ".safetensors"}
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_BASE_RELEASE = "https://github.com/anikde/STocr/releases/download/V2.0.0/{language}.ckpt"


def published_base(language: str) -> str:
    """Return the published PARSeq base checkpoint for a language.

    This is the language release, not a later fine-tune.
    """
    name = (language or "marathi").strip() or "marathi"
    return _BASE_RELEASE.format(language=name)


class WeightError(ValueError):
    """A model location could not be resolved."""


def resolve_weight(spec: str, cache: Path, *, expect: str = "file") -> Path:
    """Return a local file or directory for ``spec``.

    An existing local path is used as it is. A Hugging Face link or ``org/repo``
    id is downloaded into ``cache``. ``expect`` is ``file`` or ``dir``.
    """
    text = spec.strip()
    if not text:
        raise WeightError("model location is empty")
    local = _existing_local(text)
    if local is not None:
        return _match_expectation(local, expect)
    if not _is_remote(text):
        raise WeightError(f"model path does not exist: {text}")
    downloaded = _download(text, cache)
    return _match_expectation(downloaded, expect)


def _existing_local(spec: str) -> Path | None:
    if spec.startswith(("http://", "https://", "hf://")):
        return None
    path = Path(spec).expanduser()
    if path.exists():
        return path.resolve()
    return None


def _is_remote(spec: str) -> bool:
    if spec.startswith(("http://", "https://", "hf://")):
        return True
    repo, _, filename = spec.partition("/")
    if not repo or not filename or " " in spec:
        return False
    if Path(filename).suffix.lower() in _FILE_SUFFIXES:
        return True
    return _REPO.match(spec) is not None


def _match_expectation(path: Path, expect: str) -> Path:
    if expect == "dir":
        if path.is_dir():
            return path
        if (path.parent / "config.json").is_file():
            return path.parent
        raise WeightError(f"expected a model directory, got {path}")
    if path.is_file():
        return path
    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in _FILE_SUFFIXES
        ]
        if len(files) == 1:
            return files[0]
        if files:
            ckpt = [item for item in files if item.suffix.lower() == ".ckpt"]
            if len(ckpt) == 1:
                return ckpt[0]
        raise WeightError(f"expected one checkpoint file in {path}")
    raise WeightError(f"expected a checkpoint file, got {path}")


def _download(spec: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    repo, revision, filename, direct = _parse_remote(spec)
    if direct is not None:
        destination = cache / Path(urlparse(direct).path).name
        if not destination.is_file():
            urlretrieve(direct, destination)  # noqa: S310
        return destination
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise WeightError(
            "huggingface_hub is not installed. Install requirements.txt, then rerun."
        ) from exc
    if filename:
        saved = hf_hub_download(repo_id=repo, filename=filename, revision=revision, cache_dir=cache)
        return Path(saved)
    saved_dir = snapshot_download(repo_id=repo, revision=revision, cache_dir=cache)
    return Path(saved_dir)


def _parse_remote(spec: str) -> tuple[str, str | None, str | None, str | None]:
    """Return repo id, revision, filename, and a direct file URL when there is one."""
    text = spec[5:] if spec.startswith("hf://") else spec
    if text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        host = parsed.netloc.lower()
        parts = [part for part in parsed.path.split("/") if part]
        if "huggingface.co" not in host or len(parts) < 2:
            return "", None, None, text
        repo = f"{parts[0]}/{parts[1]}"
        if len(parts) >= 5 and parts[2] in {"resolve", "blob"}:
            filename = "/".join(parts[4:])
            if Path(filename).suffix.lower() in _FILE_SUFFIXES or parts[2] == "resolve":
                return repo, parts[3], filename, None
        if len(parts) == 3 and Path(parts[2]).suffix.lower() in _FILE_SUFFIXES:
            return repo, None, parts[2], None
        return repo, None, None, None
    pieces = text.split("/")
    if len(pieces) >= 3 and Path(pieces[-1]).suffix.lower() in _FILE_SUFFIXES:
        return f"{pieces[0]}/{pieces[1]}", None, "/".join(pieces[2:]), None
    return text, None, None, None
