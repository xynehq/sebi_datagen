"""Turn an uploaded image or one PDF page into the picture sent to the model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from image_preprocess import normalize_document_image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_MIN_DPI = 72
_MAX_DPI = 400
_DEFAULT_MAX_SIDE = 2048


class RenderError(ValueError):
    """The upload could not be turned into a page image."""


@dataclass(frozen=True)
class PreparedPage:
    image: Image.Image
    page: int
    page_count: int
    info: str


def max_side() -> int:
    raw = os.environ.get("OCR_MAX_SIDE", "").strip() or str(_DEFAULT_MAX_SIDE)
    try:
        value = int(raw)
    except ValueError as exc:
        raise RenderError("OCR_MAX_SIDE must be a whole number") from exc
    if value < 256:
        raise RenderError("OCR_MAX_SIDE must be at least 256")
    return value


def prepare(path: Path, *, page: int, dpi: int) -> PreparedPage:
    if page < 1:
        raise RenderError("page must be at least 1")
    if not _MIN_DPI <= dpi <= _MAX_DPI:
        raise RenderError(f"dpi must be between {_MIN_DPI} and {_MAX_DPI}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _prepare_pdf(path, page=page, dpi=dpi)
    if suffix in _IMAGE_SUFFIXES:
        return _prepare_image(path, page=page)
    raise RenderError("Upload a PDF, PNG, or JPEG")


def page_count(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return 1
    if suffix != ".pdf":
        raise RenderError("Upload a PDF, PNG, or JPEG")
    pdf = _open_pdf(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _prepare_image(path: Path, *, page: int) -> PreparedPage:
    if page != 1:
        raise RenderError("an image has only page 1")
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    except Exception as exc:
        raise RenderError(f"Could not open the image: {exc}") from exc
    image = _cap_long_side(normalize_document_image(image))
    width, height = image.size
    return PreparedPage(
        image=image,
        page=1,
        page_count=1,
        info=f"Processing image, {width}x{height}. DPI applies to PDFs only.",
    )


def _prepare_pdf(path: Path, *, page: int, dpi: int) -> PreparedPage:
    pdf = _open_pdf(path)
    try:
        total = len(pdf)
        if total < 1:
            raise RenderError("PDF has no pages")
        if page > total:
            raise RenderError(f"page must be between 1 and {total}")
        rendered = pdf[page - 1].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        pdf.close()
    image = _cap_long_side(normalize_document_image(rendered))
    width, height = image.size
    return PreparedPage(
        image=image,
        page=page,
        page_count=total,
        info=f"Processing page {page} of {total} at {dpi} DPI, {width}x{height}.",
    )


def _open_pdf(path: Path):
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(path)
    except Exception as exc:
        raise RenderError(f"Could not open the PDF: {exc}") from exc


def _cap_long_side(image: Image.Image) -> Image.Image:
    limit = max_side()
    width, height = image.size
    longest = max(width, height)
    if longest <= limit:
        return image
    scale = limit / longest
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized
