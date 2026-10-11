"""SHA-256 helpers.

Source PDFs are identified by hash. Filenames are display metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_HEX_LENGTH = 64


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase hex SHA-256 of a file without loading it all at once."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256_hex(value: str) -> bool:
    """Return whether ``value`` is a lowercase SHA-256 hex digest."""
    if len(value) != _HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)
