"""Save a finished page reading so it can be reviewed before training.

These files are model output. They are not gold labels.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

_LOCK = threading.Lock()


def capture_root() -> Path:
    configured = os.environ.get("OCR_CAPTURE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "ocr_captures"


def save_capture(
    *,
    image: Image.Image,
    text: str,
    source_name: str,
    model: str,
    page: int,
    page_count: int,
    dpi: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, str] | None:
    """Write the page image, the raw text, and one ledger row."""
    reading = text.strip()
    if not reading:
        return None
    root = capture_root()
    pages = root / "pages"
    texts = root / "text"
    pages.mkdir(parents=True, exist_ok=True)
    texts.mkdir(parents=True, exist_ok=True)
    capture_id = uuid4().hex
    page_path = pages / f"{capture_id}.png"
    text_path = texts / f"{capture_id}.txt"
    image.save(page_path, format="PNG")
    text_path.write_text(reading + "\n", encoding="utf-8")
    record = {
        "capture_id": capture_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source_name": Path(source_name).name,
        "model": model,
        "page": page,
        "page_count": page_count,
        "dpi": dpi,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "width": image.size[0],
        "height": image.size[1],
        "page_path": str(page_path),
        "text_path": str(text_path),
        "text": reading,
        "source": "model_capture",
        "gold": False,
    }
    ledger = root / "captures.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return {"capture_id": capture_id, "page_path": str(page_path), "text_path": str(text_path)}
