"""A review page with Accept, Reject, Accept all, and Reject all.

The pipeline does not wait. It writes this page and stops. Button clicks save
reports/review/decisions.json. The next command records those decisions.
"""

from __future__ import annotations

import html
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pipeline.dataset_flow import read_jsonl
from pipeline.gold import BLOCKED_REVIEWERS

PACK_DIR = Path("reports/review")
DECISIONS_NAME = "decisions.json"
PAGE_NAME = "index.html"
_DECISION_LABELS = frozenset({"none", "accept", "correct", "reject"})
_LABEL_ALIASES = {
    "human_verified": "accept",
    "human_corrected": "correct",
    "rejected": "reject",
}


def acceptance_text(row: dict[str, Any]) -> str | None:
    """Text to store when the operator accepts a line as proposed."""
    proposed = row.get("proposed_text")
    if isinstance(proposed, str) and proposed.strip():
        return proposed.strip()
    predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
    counts: dict[str, int] = {}
    for value in predictions.values():
        if isinstance(value, str) and value.strip():
            text = value.strip()
            counts[text] = counts.get(text, 0) + 1
    if counts:
        best = max(counts, key=lambda text: (counts[text], text))
        if counts[best] >= 2:
            return best
    for name in ("indicocr", "chitrapathak", "tesseract"):
        value = predictions.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def default_decisions(rows: list[dict[str, Any]], reviewer_id: str) -> dict[str, Any]:
    """Leave every line unreviewed. The text box still shows the proposed line."""
    return {
        "reviewer_id": reviewer_id,
        "rows": [
            {
                "crop_id": row["crop_id"],
                "decision": "none",
                "text": acceptance_text(row) or "",
            }
            for row in rows
        ],
    }


class ReviewSaveError(ValueError):
    """A posted review decision cannot be stored."""


def load_decisions(root: Path, rows: list[dict[str, Any]], reviewer_id: str) -> dict[str, Any]:
    """Return the decisions file, or the default accept/reject choices."""
    path = root / PACK_DIR / DECISIONS_NAME
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload
    return default_decisions(rows, reviewer_id)


def save_decisions(root: Path, payload: dict[str, Any]) -> Path:
    """Write the decisions file the accept command reads."""
    root = root.resolve()
    rows = read_jsonl(root / "manifests" / "consensus.jsonl")
    if not rows:
        raise ReviewSaveError("No consensus rows to review.")
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ReviewSaveError("Decisions must include a rows list.")
    reviewer = str(payload.get("reviewer_id") or "operator").strip() or "operator"
    if reviewer.lower() in BLOCKED_REVIEWERS:
        raise ReviewSaveError(f"reviewer {reviewer} cannot record decisions")
    known = {str(row["crop_id"]) for row in rows}
    incoming: dict[str, dict[str, str]] = {}
    for item in payload["rows"]:
        if not isinstance(item, dict) or not item.get("crop_id"):
            raise ReviewSaveError(f"decision row is missing crop_id: {item!r}")
        crop_id = str(item["crop_id"])
        if crop_id not in known:
            raise ReviewSaveError(f"unknown crop {crop_id}")
        label = str(item.get("decision") or "accept").strip().lower()
        label = _LABEL_ALIASES.get(label, label)
        if label not in _DECISION_LABELS:
            raise ReviewSaveError(f"unknown decision {label!r}")
        text = item.get("text") if isinstance(item.get("text"), str) else ""
        incoming[crop_id] = {"crop_id": crop_id, "decision": label, "text": text}
    current = load_decisions(root, rows, reviewer)
    kept = {
        str(item["crop_id"]): item
        for item in current.get("rows", [])
        if isinstance(item, dict) and item.get("crop_id")
    }
    ordered = []
    for row in rows:
        crop_id = str(row["crop_id"])
        ordered.append(
            incoming.get(crop_id)
            or kept.get(crop_id)
            or {"crop_id": crop_id, "decision": "reject", "text": ""}
        )
    path = root / PACK_DIR / DECISIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"reviewer_id": reviewer, "rows": ordered}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_review_pack(root: Path, *, reviewer_id: str = "operator") -> Path:
    """Write the review page and a decisions file the accept command reads."""
    root = root.resolve()
    rows = read_jsonl(root / "manifests" / "consensus.jsonl")
    if not rows:
        raise FileNotFoundError(
            "No consensus rows yet. Run through calibrate, then open the review page."
        )
    destination = root / PACK_DIR
    destination.mkdir(parents=True, exist_ok=True)
    decisions_path = destination / DECISIONS_NAME
    payload = load_decisions(root, rows, reviewer_id)
    if not decisions_path.is_file():
        save_decisions(root, payload)
    page = destination / PAGE_NAME
    page.write_text(_page(root, rows, payload, decisions_path), encoding="utf-8")
    return page


def announce_review(root: Path) -> Path:
    """Write the pack and print where the review page will open."""
    page = write_review_pack(root)
    print("REVIEW")
    print(f"  page: {page}")
    print("The browser opens on its own. When every line is reviewed, confirm")
    print("to export the dataset and continue with training and evaluation.")
    return page


class ReviewServer(ThreadingHTTPServer):
    """Local review server that stops when the operator confirms."""

    review_root: Path
    reviewer_id: str
    review_confirmed: threading.Event

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        root: Path,
        reviewer_id: str,
    ) -> None:
        self.review_root = root
        self.reviewer_id = reviewer_id
        self.review_confirmed = threading.Event()
        super().__init__(address, handler)


def start_review_server(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    reviewer_id: str = "operator",
) -> ReviewServer:
    """Serve the review page and save button clicks to the decisions file."""
    root = root.resolve()
    write_review_pack(root)
    handler = _handler(root)
    try:
        return ReviewServer((host, port), handler, root, reviewer_id)
    except OSError:
        return ReviewServer((host, 0), handler, root, reviewer_id)


def confirm_review(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    reviewer_id: str = "operator",
    open_browser: bool = True,
) -> bool:
    """Open the review page and wait until the user confirms continuation."""
    server = start_review_server(root, host=host, port=port, reviewer_id=reviewer_id)
    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/reports/review/index.html"
    print(f"Review page: {url}")
    print("The run is waiting. Review every line, then confirm to continue.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Review was not confirmed.")
        return False
    finally:
        server.server_close()
    return server.review_confirmed.is_set()


def serve_review(root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> int:
    """Open the review page and wait for confirmation."""
    if confirm_review(root, host=host, port=port):
        return 0
    print("Review was not confirmed. Training did not start.")
    return 0


def _page(
    root: Path,
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    decisions_path: Path,
) -> str:
    by_id = {
        str(item["crop_id"]): item
        for item in payload.get("rows", [])
        if isinstance(item, dict) and item.get("crop_id")
    }
    cards = []
    proposed: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        crop_id = str(row["crop_id"])
        proposed[crop_id] = acceptance_text(row) or ""
        cards.append(_card(root, index, row, by_id.get(crop_id)))
    state = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    proposed_json = json.dumps(proposed, ensure_ascii=False).replace("<", "\\u003c")
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR review</title>
<style>
{_PAGE_CSS}
</style>
</head>
<body>
<header>
  <h1>Line review</h1>
  <p class="muted" id="counts">{len(rows)} lines</p>
  <p>Every line starts as not reviewed. Accept or Reject marks that line.
  When every line is reviewed, a confirmation asks whether to continue.
  Yes closes this page and starts dataset export, training, and evaluation.
  No leaves this page open.</p>
  <p class="toolbar">
    <button type="button" id="accept-all">Accept all</button>
    <button type="button" id="reject-all">Reject all</button>
    <span id="save-status" class="muted"></span>
  </p>
</header>
<main>
{body}
</main>
<script id="review-state" type="application/json">{state}</script>
<script id="proposed-text" type="application/json">{proposed_json}</script>
<script>
{_PAGE_SCRIPT}
</script>
</body>
</html>
"""


_PAGE_CSS = """
body { margin: 0; font-family: Georgia, serif; background: #f6f3ec; color: #1c1917; }
header {
  position: sticky; top: 0; z-index: 1;
  padding: 1rem 1.25rem; background: #fffdf8; border-bottom: 1px solid #d6d3d1;
}
main { padding: 1rem 1.25rem 2rem; display: grid; gap: 1rem; }
article {
  display: grid; grid-template-columns: 1.1fr 1fr; gap: 1rem;
  background: #fffdf8; border: 1px solid #d6d3d1; border-radius: 10px; padding: 1rem;
}
article.none { border-color: #d6d3d1; }
article.accept { border-color: #166534; background: #f0fdf4; }
article.correct { border-color: #a16207; background: #fffbeb; }
article.reject { border-color: #991b1b; background: #fef2f2; }
[data-role="state"] { font-weight: 700; }
article.accept [data-role="state"] { color: #166534; }
article.correct [data-role="state"] { color: #a16207; }
article.reject [data-role="state"] { color: #991b1b; }
img { max-width: 100%; background: #fff; }
pre, textarea {
  white-space: pre-wrap;
  font-family: "Kohinoor Devanagari", "Noto Sans Devanagari", sans-serif;
  font-size: 1.05rem;
}
textarea { width: 100%; min-height: 4.5rem; box-sizing: border-box; }
.muted { color: #57534e; }
.toolbar, .actions { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
button {
  font: inherit; padding: 0.45rem 0.9rem; border-radius: 8px; cursor: pointer;
  border: 1px solid #d6d3d1; background: #fff;
}
button[aria-pressed="true"] { color: #fff; border-color: transparent; }
article.accept button[data-action="accept"][aria-pressed="true"],
#accept-all { background: #166534; color: #fff; border-color: #166534; }
article.correct button[data-action="accept"][aria-pressed="true"] { background: #a16207; }
article.reject button[data-action="reject"][aria-pressed="true"],
#reject-all { background: #991b1b; color: #fff; border-color: #991b1b; }
@media (max-width: 900px) { article { grid-template-columns: 1fr; } }
"""


_PAGE_SCRIPT = """
const state = JSON.parse(document.getElementById("review-state").textContent);
const proposed = JSON.parse(document.getElementById("proposed-text").textContent);
const statusNode = document.getElementById("save-status");
let touched = false;

function rowById(cropId) {
  return state.rows.find((row) => row.crop_id === cropId);
}

function paint() {
  let accepted = 0;
  let corrected = 0;
  let rejected = 0;
  let pending = 0;
  for (const article of document.querySelectorAll("article[data-crop-id]")) {
    const row = rowById(article.dataset.cropId);
    const decision = row.decision || "none";
    article.classList.remove("none", "accept", "correct", "reject");
    article.classList.add(decision);
    const field = article.querySelector("[data-role=text]");
    if (document.activeElement !== field) field.value = row.text || "";
    article.querySelector('[data-action="accept"]').setAttribute(
      "aria-pressed", String(decision === "accept" || decision === "correct")
    );
    article.querySelector('[data-action="reject"]').setAttribute(
      "aria-pressed", String(decision === "reject")
    );
    const label = article.querySelector("[data-role=state]");
    if (decision === "accept") {
      accepted += 1;
      label.textContent = "Accepted";
    } else if (decision === "correct") {
      corrected += 1;
      label.textContent = "Accepted with your edited text";
    } else if (decision === "reject") {
      rejected += 1;
      label.textContent = "Rejected";
    } else {
      pending += 1;
      label.textContent = "Not reviewed";
    }
  }
  const parts = [pending + " not reviewed"];
  if (accepted) parts.push(accepted + " accepted");
  if (corrected) parts.push(corrected + " corrected");
  if (rejected) parts.push(rejected + " rejected");
  document.getElementById("counts").textContent = parts.join(", ");
}

function choice(cropId, decision, text) {
  const row = rowById(cropId);
  const cleaned = (text || "").trim();
  if (decision === "reject") {
    row.decision = "reject";
    row.text = cleaned;
    return "Rejected this line.";
  }
  if (!cleaned) {
    row.decision = "none";
    row.text = "";
    return "This line has no text, so it stayed not reviewed.";
  }
  row.text = cleaned;
  row.decision = cleaned === (proposed[cropId] || "") ? "accept" : "correct";
  return row.decision === "correct"
    ? "Accepted this line with your edited text."
    : "Accepted this line.";
}

async function save(message) {
  touched = true;
  paint();
  statusNode.textContent = message + " Saving…";
  try {
    const response = await fetch("/api/decisions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state),
    });
    if (!response.ok) throw new Error(await response.text());
    statusNode.textContent = message + " Saved.";
  } catch (error) {
    statusNode.textContent = message +
      " Not saved. Start ./run_pipeline.sh --serve-review and open that page.";
    return;
  }
  await maybeContinue(message);
}

function unreviewedCount() {
  return state.rows.filter((row) => !row.decision || row.decision === "none").length;
}

async function maybeContinue(message) {
  if (unreviewedCount() !== 0) return;
  const yes = window.confirm(
    "Every line is reviewed. Continue with dataset export, training, and evaluation?"
  );
  if (!yes) {
    statusNode.textContent = message + " Saved. This page stays open until you confirm.";
    return;
  }
  statusNode.textContent = "Starting dataset export, training, and evaluation…";
  try {
    const response = await fetch("/api/continue", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state),
    });
    if (!response.ok) throw new Error(await response.text());
    document.body.innerHTML = "<p>Confirmed. Dataset export, training, and evaluation " +
      "are continuing in the terminal. You can close this page.</p>";
    window.close();
  } catch (error) {
    statusNode.textContent = error.message || "Could not continue. This page stays open.";
  }
}

document.getElementById("accept-all").addEventListener("click", () => {
  let accepted = 0;
  let pending = 0;
  for (const article of document.querySelectorAll("article[data-crop-id]")) {
    const note = choice(
      article.dataset.cropId,
      "accept",
      article.querySelector("[data-role=text]").value
    );
    if (note.startsWith("Accepted")) accepted += 1;
    else pending += 1;
  }
  const message = "Accepted " + accepted + " lines." +
    (pending ? " " + pending + " stayed not reviewed." : "");
  save(message);
});

document.getElementById("reject-all").addEventListener("click", () => {
  for (const row of state.rows) row.decision = "reject";
  save("Rejected all " + state.rows.length + " lines.");
});

for (const article of document.querySelectorAll("article[data-crop-id]")) {
  article.querySelector('[data-action="accept"]').addEventListener("click", () => {
    const note = choice(
      article.dataset.cropId,
      "accept",
      article.querySelector("[data-role=text]").value
    );
    save(note);
  });
  article.querySelector('[data-action="reject"]').addEventListener("click", () => {
    save(choice(article.dataset.cropId, "reject", article.querySelector("[data-role=text]").value));
  });
}

fetch("decisions.json", {cache: "no-store"}).then(async (response) => {
  if (!response.ok || touched) return;
  const fresh = await response.json();
  if (!fresh.rows || touched) return;
  state.rows = fresh.rows;
  state.reviewer_id = fresh.reviewer_id || state.reviewer_id;
  paint();
}).catch(() => paint());
paint();
"""


def _card(
    root: Path,
    index: int,
    row: dict[str, Any],
    decision: dict[str, Any] | None,
) -> str:
    image = _image_src(root, row.get("crop_path"))
    predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
    lines = []
    for name in ("indicocr", "chitrapathak", "tesseract"):
        value = predictions.get(name)
        shown = html.escape(value) if isinstance(value, str) and value else "(empty)"
        lines.append(f"<div><strong>{name}</strong><pre>{shown}</pre></div>")
    chosen = ""
    if decision and isinstance(decision.get("text"), str):
        chosen = decision["text"]
    if not chosen:
        chosen = acceptance_text(row) or ""
    crop_id = html.escape(str(row.get("crop_id")))
    return f"""<article data-crop-id="{crop_id}">
  <div>
    <h2>{index}. {crop_id}</h2>
    <p class="muted">{html.escape(str(row.get("agreement_class") or ""))}</p>
    <img src="{html.escape(image)}" alt="line crop">
  </div>
  <div>
    {"".join(lines)}
    <label>Text
    <textarea data-role="text">{html.escape(chosen)}</textarea></label>
    <div class="actions">
      <button type="button" data-action="accept">Accept</button>
      <button type="button" data-action="reject">Reject</button>
      <span data-role="state" class="muted"></span>
    </div>
  </div>
</article>"""


def _handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            target = _safe_file(root, urlparse(self.path).path)
            if target is None:
                self.send_error(404)
                return
            content_type = "application/octet-stream"
            if target.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif target.suffix == ".json":
                content_type = "application/json; charset=utf-8"
            elif target.suffix == ".png":
                content_type = "image/png"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/decisions":
                self._save_only()
                return
            if path == "/api/continue":
                self._confirm_and_continue()
                return
            self.send_error(404)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length < 0 or length > 5_000_000:
                raise ReviewSaveError("Review upload is too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ReviewSaveError("Decisions must be an object.")
            return payload

        def _save_only(self) -> None:
            try:
                save_decisions(root, self._body())
            except (json.JSONDecodeError, ReviewSaveError, UnicodeError) as exc:
                self._plain(400, str(exc))
                return
            self._json(b'{"ok":true}')

        def _confirm_and_continue(self) -> None:
            from pipeline.accept_review import AcceptError, accept_reviews

            try:
                payload = self._body()
                path = save_decisions(root, payload)
                stored = json.loads(path.read_text(encoding="utf-8"))
                unfinished = [
                    item
                    for item in stored.get("rows", [])
                    if str(item.get("decision") or "none").strip().lower() in {"none", "pending"}
                ]
                if unfinished:
                    raise ReviewSaveError("Review every line before continuing.")
                reviewer = str(getattr(self.server, "reviewer_id", "") or "operator")
                code = accept_reviews(root, reviewer_id=reviewer)
            except (json.JSONDecodeError, ReviewSaveError, UnicodeError, AcceptError) as exc:
                self._plain(400, str(exc))
                return
            if code != 0:
                self._plain(400, "Could not record the review.")
                return
            confirmed = getattr(self.server, "review_confirmed", None)
            if confirmed is not None:
                confirmed.set()
            self._json(b'{"ok":true}')
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _plain(self, status: int, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[review] {self.address_string()} {fmt % args}")

    return Handler


def _safe_file(root: Path, url_path: str) -> Path | None:
    relative = unquote(url_path).lstrip("/")
    if relative in {"", "index.html"}:
        relative = "reports/review/index.html"
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _image_src(root: Path, crop_path: object) -> str:
    if not isinstance(crop_path, str) or not crop_path:
        return ""
    path = Path(crop_path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.resolve().relative_to((root / PACK_DIR).resolve())
    except ValueError:
        relative = Path("..") / ".." / crop_path
    return relative.as_posix()
