"""Synthetic structured pages for deterministic structure tests."""

from __future__ import annotations


def holiday_table_reference(rows: int = 3, language: str = "guj") -> str:
    """Known row-major HTML table (structure ground truth, not real OCR wording)."""
    if language == "guj":
        header = ["ક્રમાંક", "જાહેર રજાનું નામ", "તારીખ / શક સંવત", "અઠવાડિયાનો વાર"]
        body = [
            ("૧.", "મકરસંક્રાંતિ", "૧૪મી જાન્યુઆરી, ૨૦૧૫ / ૨૪, પૌષ, ૧૯૩૬", "બુધવાર"),
            ("૨.", "પ્રજાસત્તાક દિન", "૨૬મી જાન્યુઆરી, ૨૦૧૫ / ૬, માઘ, ૧૯૩૬", "સોમવાર"),
            ("૩.", "મહા શિવરાત્રી", "૧૭મી ફેબ્રુઆરી, ૨૦૧૫ / ૨૮, માઘ, ૧૯૩૬", "મંગળવાર"),
        ]
    else:
        header = ["क्रमांक", "नाम", "तारीख", "वार"]
        body = [
            ("१.", "मकरसंक्रांति", "१४ जान्युआरी", "बुधवार"),
            ("२.", "प्रजासत्ताक दिन", "२६ जान्युआरी", "सोमवार"),
            ("३.", "महा शिवरात्री", "१७ फरवरी", "मंगळवार"),
        ]
    lines = [
        "# સામાન્ય વહીવટ વિભાગ" if language == "guj" else "# विभाग",
        "",
        "## જાહેરનામું" if language == "guj" else "## सूचना",
        "",
        "<table>",
        "  <thead>",
        "    <tr>",
    ]
    lines.extend(f"      <th>{cell}</th>" for cell in header)
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for row in body[:rows]:
        lines.append("    <tr>")
        lines.extend(f"      <td>{cell}</td>" for cell in row)
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def column_major_bad_prediction(rows: int = 3) -> str:
    guj_serial = "૦૧૨૩૪૫૬૭૮૯"
    serials = [f"{guj_serial[i]}." for i in range(1, rows + 1)]
    names = [f"રજા_{i}" for i in range(1, rows + 1)]
    dates = [f"{10 + i}મી જાન્યુઆરી" for i in range(rows)]
    weekdays = ["સોમવાર", "મંગળવાર", "બુધવાર", "ગુરુવાર", "શુક્રવાર"][:rows]
    return "\n".join(serials + names + dates + weekdays)
