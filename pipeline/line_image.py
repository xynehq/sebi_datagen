"""Draw a line image for the fixture pipeline.

Production crops are cut from the rendered page. This helper is only for the
fixture, and it refuses to save a blank rectangle.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONTS = (
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Kohinoor.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf",
)


def write_text_image(path: Path, text: str) -> None:
    """Save a white crop whose black glyphs are ``text``."""
    label = text.strip()
    if not label:
        raise ValueError(f"refusing to write a blank crop at {path}")
    font = _font(32)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = probe.textbbox((0, 0), label, font=font)
    width = max(16, box[2] - box[0] + 16)
    height = max(16, box[3] - box[1] + 12)
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).text((8, 4 - box[1]), label, font=font, fill="black")
    colors = image.getcolors(maxcolors=256)
    if colors is not None and len(colors) < 2:
        raise ValueError(f"font did not draw {label!r} into {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in _FONTS:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError(
        "No Devanagari font found. Install Noto Sans Devanagari or use macOS."
    )
