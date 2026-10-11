"""HTML report for one pipeline run."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(run_dir: Path, summary: dict[str, Any]) -> Path:
    path = run_dir / "reports" / "final_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    scoreboard = summary.get("scoreboard") or {}
    rows = scoreboard.get("rows") or []
    body_rows = []
    for row in rows:
        metrics = row.get("metrics") or {}
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('checkpoint')))}</td>"
            f"<td>{html.escape(str(metrics.get('cer')))}</td>"
            f"<td>{html.escape(str(metrics.get('content_cer')))}</td>"
            f"<td>{html.escape(str(metrics.get('wer')))}</td>"
            f"<td>{html.escape(str(metrics.get('exact_match_rate')))}</td>"
            "</tr>"
        )
    selected = html.escape(str(scoreboard.get("selected", "")))
    frozen = summary.get("frozen") or {}
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR pipeline report {html.escape(str(summary.get("run_id", "")))}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1917; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #d6d3d1; padding: 0.4rem 0.6rem; }}
  </style>
</head>
<body>
  <h1>OCR pipeline report</h1>
  <p>Run {html.escape(str(summary.get("run_id", "")))}</p>
  <h2>Dataset</h2>
  <pre>{html.escape(json.dumps(summary.get("counts") or {}, ensure_ascii=False, indent=2))}</pre>
  <h2>Checkpoint scoreboard</h2>
  <p>Primary metric: strict CER, minimized. Frozen-100 was not used for selection.</p>
  <table>
    <tr><th>Checkpoint</th><th>CER</th><th>Content CER</th><th>WER</th><th>Exact match</th></tr>
    {"".join(body_rows)}
  </table>
  <p>Selected checkpoint: {selected}</p>
  <h2>Frozen-100</h2>
  <pre>{html.escape(json.dumps(frozen, ensure_ascii=False, indent=2)[:4000])}</pre>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path
