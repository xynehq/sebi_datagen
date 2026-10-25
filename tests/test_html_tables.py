"""HTML table output (LightOn-style) becomes markdown rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from html_tables import convert_html_tables_in_text  # noqa: E402
from reading import finalize_reading  # noqa: E402

_SAMPLE = """
# Title

<table>
  <thead>
    <tr><th>क्रमांक</th><th>नाम</th><th>तारीख</th></tr>
  </thead>
  <tbody>
    <tr><td>१.</td><td>मकरसंकांति</td><td>१४ जान्युआरी</td></tr>
    <tr><td>२.</td><td>प्रजासत्ताक दिन</td><td>२६ जान्युआरी</td></tr>
  </tbody>
</table>
"""


def test_html_table_converts_to_markdown_rows() -> None:
    out = convert_html_tables_in_text(_SAMPLE)
    assert "| १. | मकरसंकांति | १४ जान्युआरी |" in out
    assert "<table" not in out


def test_finalize_reading_keeps_html() -> None:
    out = finalize_reading(_SAMPLE)
    assert "<table>" in out.lower()
    assert "<td>२.</td>" in out or "२." in out
    assert "<td>प्रजासत्ताक दिन</td>" in out or "प्रजासत्ताक दिन" in out
