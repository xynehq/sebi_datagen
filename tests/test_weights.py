"""Local model paths stay local. Hugging Face links are recognized without downloading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline.weights import WeightError, _parse_remote, published_base, resolve_weight


def test_existing_file_is_used(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    assert resolve_weight(str(checkpoint), tmp_path / "cache") == checkpoint.resolve()


def test_existing_directory_with_one_checkpoint(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    checkpoint = folder / "parseq.ckpt"
    checkpoint.write_bytes(b"weights")
    assert resolve_weight(str(folder), tmp_path / "cache") == checkpoint.resolve()


def test_missing_local_path_is_not_downloaded(tmp_path: Path) -> None:
    with pytest.raises(WeightError, match="does not exist"):
        resolve_weight(str(tmp_path / "missing.ckpt"), tmp_path / "cache")


def test_empty_checkpoint_uses_published_language_base() -> None:
    assert published_base("marathi").endswith("/marathi.ckpt")
    assert published_base("").endswith("/marathi.ckpt")


def test_huggingface_link_is_parsed() -> None:
    repo, revision, filename, direct = _parse_remote(
        "https://huggingface.co/org/parseq/resolve/main/model.ckpt"
    )
    assert repo == "org/parseq"
    assert revision == "main"
    assert filename == "model.ckpt"
    assert direct is None
    repo, revision, filename, direct = _parse_remote("hf://org/parseq")
    assert (repo, revision, filename, direct) == ("org/parseq", None, None, None)
