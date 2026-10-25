"""Single finalization path for UI, API, and exports."""

from __future__ import annotations

import logging
from dataclasses import asdict

from PIL import Image

from document_geometry import GeometryReport, geometry_version, rebuild_page_from_geometry
from page_layout import apply_page_layout
from reading import CHITRA_PROMPT, clean_reading, reading_raw
from table_layout import force_tabular_layout, table_layout_version

logger = logging.getLogger(__name__)

_INDIC_PROMPT = (
    "Transcribe the text in this image. Write any mathematical expressions in LaTeX, "
    "using $...$ for inline math and $$...$$ for display equations."
)

FINALIZER_VERSION = "finalize-v2"
PAGE_LAYOUT_VERSION = "page-layout-v2"

_last_geometry_report: GeometryReport | None = None


def last_geometry_report() -> dict[str, object] | None:
    if _last_geometry_report is None:
        return None
    return asdict(_last_geometry_report)


def finalize_document_markdown(
    image: Image.Image,
    text: str,
    *,
    model_id: str,
) -> str:
    """Canonical structured markdown for a page (headings, tables, lists)."""
    global _last_geometry_report
    base = clean_reading(text, CHITRA_PROMPT, _INDIC_PROMPT)
    if not model_id:
        _last_geometry_report = GeometryReport(fallback_used=True)
        return base

    rebuilt, report = rebuild_page_from_geometry(image, base, model_id=model_id)
    _last_geometry_report = report
    if report.geometry_table_rows >= 8 and not report.fallback_used:
        logger.debug("geometry rebuild rows=%s format=%s", report.geometry_table_rows, report.table_format)
        return rebuilt

    laid_out = apply_page_layout(image, base, model_id=model_id)
    final = force_tabular_layout(image, laid_out, model_id=model_id)
    return final


def finalize_document_for_response(
    image: Image.Image,
    text: str,
    *,
    model_id: str,
) -> str:
    """Markdown stored in API `text`, exports, and the raw UI view."""
    return reading_raw(finalize_document_markdown(image, text, model_id=model_id))


def stream_preview_markdown(text: str) -> str:
    """Live rendered view while tokens arrive (cheap; no Tesseract/geometry).

    LightOn looks live because the model emits HTML tables as it generates and
    Gradio Markdown paints them immediately. We do the same when the model
    streams ``<table>`` or pipe/TSV rows, and only run the heavy row-alignment
    pass once generation finishes.
    """
    from html_tables import (
        close_incomplete_html_tables,
        convert_markdown_tables_to_html,
        convert_tsv_blocks_to_html,
        has_html_table,
    )
    from reading import reading_markdown

    body = clean_reading(text, CHITRA_PROMPT, _INDIC_PROMPT)
    if not body.strip():
        return ""

    # Prefer structured HTML as soon as enough of a table is present.
    if has_html_table(body):
        live = close_incomplete_html_tables(body)
        return live

    if "|" in body and "---" in body:
        return convert_markdown_tables_to_html(body)

    if "\t" in body and body.count("\t") >= 4:
        tsv = convert_tsv_blocks_to_html(body)
        if has_html_table(tsv):
            return tsv

    rendered = reading_markdown(body)
    return (
        f"{rendered}\n\n"
        "*Generating… full row alignment runs when this pass finishes.*"
    )
