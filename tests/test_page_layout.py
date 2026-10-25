"""General page layout: lists, tables, charts-friendly behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from page_layout import (  # noqa: E402
    _has_well_formed_table,
    _repair_list_column_dump,
    apply_page_layout,
)
from table_layout import _tesseract_table_looks_like_grid  # noqa: E402


def test_well_formed_table_is_left_alone() -> None:
    page = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert _has_well_formed_table(page)


def test_list_column_zip() -> None:
    text = "\n".join(["1.", "2.", "3.", "alpha", "beta", "gamma"])
    fixed = _repair_list_column_dump(text)
    assert "1. alpha" in fixed
    assert "2. beta" in fixed


def test_chart_like_markdown_not_forced_to_grid() -> None:
    fake_grid = "| Column 1 | Column 2 |\n| --- | --- |\n| only | two |"
    assert not _tesseract_table_looks_like_grid(fake_grid)


def test_apply_page_layout_keeps_prose() -> None:
    image = Image.new("RGB", (400, 400), "white")
    prose = "## Sales chart\n\nRevenue rose in Q1.\n\n- North\n- South"
    assert apply_page_layout(image, prose, model_id="chitrapathak-marguj-mixed") == prose


@pytest.mark.parametrize(
    "phrase",
    ["Lists:", "Charts and graphs:", "HTML", "reading order"],
)
def test_prompt_covers_layout_types(phrase: str) -> None:
    from reading import CHITRA_PROMPT  # noqa: E402

    assert phrase in CHITRA_PROMPT
