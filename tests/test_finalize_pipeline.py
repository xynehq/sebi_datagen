"""UI and API must share the same finalization path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from finalize import (  # noqa: E402
    finalize_document_for_response,
    finalize_document_markdown,
    stream_preview_markdown,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "ocr_service/examples/gujarati_holidays_2015.jpg"
_GARBLED = "ક્રમાંક\nભકરસાકંતિ\nપ્રજાસત્તાક દિન."


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="holiday fixture missing")
def test_finalize_holiday_fixture_produces_markdown_table() -> None:
    image = Image.open(_FIXTURE).convert("RGB")
    out = finalize_document_markdown(image, _GARBLED, model_id="chitrapathak-marguj-mixed")
    assert ("|" in out and "---" in out) or "<table>" in out.lower()
    assert "મકરસંક્રાંતિ" in out or "<td>" in out.lower()


def test_api_and_markdown_finalize_match() -> None:
    image = Image.new("RGB", (32, 32), "white")
    body = "## Heading\n\nParagraph."
    api_text = finalize_document_for_response(image, body, model_id="chitrapathak-2")
    direct = finalize_document_markdown(image, body, model_id="chitrapathak-2")
    assert api_text == direct or api_text.replace("\n\n", "\n") == direct.replace("\n\n", "\n")


def test_stream_preview_is_non_empty_while_generating() -> None:
    preview = stream_preview_markdown("## Title\n\nLine one.")
    assert "Title" in preview
    assert "Generating" in preview or "Line one" in preview


def test_stream_preview_closes_partial_html_table() -> None:
    partial = (
        "# Title\n\n<table>\n  <thead>\n    <tr>\n      <th>A</th>\n      <th>B</th>\n"
        "    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>1</td>\n      <td>two"
    )
    preview = stream_preview_markdown(partial)
    assert "<table>" in preview.lower()
    assert "</table>" in preview.lower()
    assert "two" in preview


def test_stream_preview_turns_tsv_into_html() -> None:
    tsv = (
        "ક્રમાંક\tનામ\tતારીખ\tવાર\n"
        "૧.\tમકરસંક્રાંતિ\t૧૪મી જાન્યુઆરી\tબુધવાર\n"
        "૨.\tપ્રજાસત્તાક દિન\t૨૬મી જાન્યુઆરી\tસોમવાર\n"
    )
    preview = stream_preview_markdown(tsv)
    assert "<table>" in preview.lower()
    assert "મકરસંક્રાંતિ" in preview
    assert "Generating" not in preview
