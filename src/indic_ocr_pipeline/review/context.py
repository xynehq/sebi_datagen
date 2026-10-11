"""Derive page-context images from stored page PNGs and line boxes.

Does not rewrite crops, pages, or detection outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from indic_ocr_pipeline.errors import DatasetError


class ReviewContextError(DatasetError):
    """Page context could not be built from stored geometry."""


def load_png_bytes(path: Path) -> bytes:
    """Return the bytes of a local PNG. Missing files are a clear error."""
    if not path.is_file():
        raise ReviewContextError(f"missing image {path}")
    return path.read_bytes()


def page_context_png(
    page_path: Path,
    bbox: Sequence[int],
    *,
    pad: int = 80,
    max_edge: int = 1600,
) -> bytes:
    """Return a PNG of the page neighborhood around ``bbox`` with the line drawn.

    ``bbox`` is the stored line box ``[x0, y0, x1, y1]`` in page pixels.
    """
    if len(bbox) != 4:
        raise ReviewContextError("line bbox must have four integers")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ReviewContextError(f"invalid line bbox {[x0, y0, x1, y1]}")
    with Image.open(page_path) as image:
        page = image.convert("RGB")
        width, height = page.size
        left = max(0, x0 - pad)
        top = max(0, y0 - pad)
        right = min(width, x1 + pad)
        bottom = min(height, y1 + pad)
        crop = page.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        draw.rectangle(
            (x0 - left, y0 - top, x1 - left - 1, y1 - top - 1),
            outline=(220, 40, 40),
            width=2,
        )
        if max(crop.size) > max_edge:
            scale = max_edge / float(max(crop.size))
            crop = crop.resize(
                (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                Image.Resampling.BILINEAR,
            )
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()


def line_bbox_from_records(
    line: Mapping[str, object] | None,
    crop: Mapping[str, object] | None,
) -> list[int]:
    """Prefer the line bbox; fall back to the crop bbox."""
    for record in (line, crop):
        if not record:
            continue
        bbox = record.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return [int(value) for value in bbox]
    raise ReviewContextError("no line or crop bbox is available")
