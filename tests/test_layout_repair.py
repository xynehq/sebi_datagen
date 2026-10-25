"""Column-major table dumps are repaired into row-major markdown."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from layout_repair import repair_column_major_tables  # noqa: E402
from reading import clean_reading, reading_raw  # noqa: E402


def test_gujarati_holiday_columns_become_rows() -> None:
    dumped = "\n".join(
        [
            "શાસકીય રજાઓ",
            "",
            "1",
            "2",
            "3",
            "",
            "ઉત્તરાયણ",
            "મકર સંક્રાંતિ",
            "ગુજરાત દિવસ",
            "",
            "14/01/2026",
            "15/01/2026",
            "01/05/2026",
            "",
            "બુધવાર",
            "ગુરુવાર",
            "શુક્રવાર",
        ]
    )
    fixed = repair_column_major_tables(dumped)
    assert "| 1 | ઉત્તરાયણ | 14/01/2026 | બુધવાર |" in fixed
    assert "| 3 | ગુજરાત દિવસ | 01/05/2026 | શુક્રવાર |" in fixed
    assert "1\n2\n3\n\nઉત્તરાયણ" not in fixed


def test_marathi_registration_rows_stay_together() -> None:
    dumped = "\n".join(
        [
            "## दस्त गोषवारा",
            "",
            "1",
            "2",
            "",
            "सच्चिदानंद",
            "राम पाटील",
            "",
            "प्रतिवादी",
            "वादी",
            "",
            "42",
            "38",
        ]
    )
    fixed = reading_raw(dumped)
    assert "| 1 | सच्चिदानंद | प्रतिवादी | 42 |" in fixed
    assert "| 2 | राम पाटील | वादी | 38 |" in fixed


def test_existing_markdown_table_is_left_alone() -> None:
    table = "\n".join(
        [
            "| अनु क्र. | नाव |",
            "|---|---|",
            "| 1 | सच्चिदानंद |",
        ]
    )
    assert repair_column_major_tables(table) == table


def test_boundary_fee_label_value_pairs() -> None:
    dumped = "\n".join(
        [
            "दिशा",
            "दिशा",
            "",
            "उत्तर",
            "दक्षिण",
            "",
            "शुल्क",
            "शुल्क",
            "",
            "500",
            "750",
        ]
    )
    fixed = repair_column_major_tables(dumped)
    assert "| दिशा | उत्तर | शुल्क | 500 |" in fixed
    assert "| दिशा | दक्षिण | शुल्क | 750 |" in fixed


def test_clause_list_is_not_treated_as_columns() -> None:
    clauses = "\n".join(
        [
            "१. पहिली अट",
            "२. दुसरी अट",
            "३. तिसरी अट",
            "४. चौथी अट",
        ]
    )
    assert repair_column_major_tables(clauses) == clauses
    assert clean_reading(clauses) == clauses
