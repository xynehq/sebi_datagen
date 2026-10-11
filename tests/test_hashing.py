"""SHA-256 identity for source files."""

from __future__ import annotations

from pathlib import Path

from indic_ocr_pipeline.utils.hashing import is_sha256_hex, sha256_bytes, sha256_file

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_known_answer() -> None:
    digest = sha256_bytes(b"abc")
    assert digest == ABC_SHA256
    assert is_sha256_hex(digest)
    assert not is_sha256_hex(digest.upper())


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    payload = b"indic-ocr-pipeline"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_sha256_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == sha256_bytes(b"")
