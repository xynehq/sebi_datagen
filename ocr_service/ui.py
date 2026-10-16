"""Gradio page that matches the LightOn OCR demo, plus a DPI control."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

import gradio as gr

from capture import save_capture
from engine import Engine
from registry import ModelError, load_specs, model_status
from render import RenderError, page_count, prepare

_DEFAULT_MODEL = "chitrapathak-2"
_LIST_BREAK = re.compile(r"(?<=\S)[ \t]+(?=[०-९0-9]{1,3}\.\s)")
_ROOT = Path(__file__).resolve().parent
_EXAMPLES = (
    (_ROOT / "examples" / "english_notice.png", "English notice"),
    (_ROOT / "examples" / "marathi_sample.png", "Marathi sample"),
    (_ROOT / "examples" / "gujarati_sample.png", "Gujarati sample"),
    (_ROOT / "examples" / "registration_slip.png", "Registration slip"),
)


def _as_lines(text: str) -> str:
    """Keep each list item on its own line, the way a page reader shows it."""
    text = _LIST_BREAK.sub("\n", text.replace("\r\n", "\n"))
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def _as_markdown(text: str) -> str:
    """Markdown collapses a single newline, so each line becomes its own block."""
    return "\n\n".join(_as_lines(text).split("\n"))


def _word_count(text: str) -> int:
    return len(text.split())


def _speed_label(words: int, elapsed: float) -> str:
    """Words per second since the first extracted word."""
    if words <= 0:
        return ""
    rate = words / max(elapsed, 0.05)
    if rate >= 10:
        return f"{rate:.0f} words/s"
    return f"{rate:.1f} words/s"


def _links_html(dataset: str, finetune: str, speed: str = "", device: str = "") -> str:
    speed_html = f' · <span class="sebi-speed">{speed}</span>' if speed else ""
    return (
        '<div class="sebi-sub">'
        '<p class="sebi-links">'
        f'<a href="{dataset}">📊 Dataset</a>'
        " · "
        f'<a href="{finetune}">📓 Finetuning</a>'
        f"{speed_html}</p>"
        '<p class="sebi-how"><strong>How to use:</strong> '
        "Select a model → Upload image/PDF → Click Extract text"
        f" · <strong>Device:</strong> {device}</p>"
        "</div>"
    )


def _write_export(text: str) -> str:
    """Save the reading as a text file the Export button can download."""
    folder = Path(tempfile.gettempdir()) / "indic-vachak"
    folder.mkdir(exist_ok=True)
    path = folder / f"extracted-{time.time_ns()}.txt"
    body = text if text.endswith("\n") else f"{text}\n"
    path.write_text(body, encoding="utf-8")
    return str(path)


def build_demo(engine: Engine) -> gr.Blocks:
    specs = load_specs()
    choices = [(spec.label, spec.id) for spec in specs]
    dataset = (
        "https://drive.google.com/drive/folders/"
        "1BZ8GoMuW1glTLcNn5qaBx6RVgkp6QHSE?usp=drive_link"
    )
    finetune = "https://huggingface.co/snxtyle"
    links = _links_html(dataset, finetune)

    def info_text(model_id: str) -> str:
        """Return formatted model info string."""
        try:
            spec = next(item for item in load_specs() if item.id == model_id)
        except StopIteration:
            return "Unknown model."
        status = model_status(spec)
        state = {
            "ready": "Weights are on this machine.",
            "not_downloaded": "Weights are not downloaded yet. The first Extract downloads them.",
            "repo_not_set": status.get("detail") or "Repo id is not set.",
        }.get(str(status["status"]), str(status["status"]))
        adapter = spec.adapter or "none"
        return (
            f"**Description:** {spec.description}\n\n"
            f"**Weights:** {state}\n\n"
            f"**Base:** `{spec.base}`  \n**Adapter:** `{adapter}`"
        )

    def preview(file_path: str | None):
        """Update page slider and preview image based on uploaded file."""
        if not file_path:
            return gr.update(maximum=20, value=1), None
        path = Path(file_path)
        try:
            total = page_count(path)
            prepared = prepare(path, page=1, dpi=72)
        except (RenderError, OSError):
            return gr.update(), None
        return gr.update(maximum=max(total, 1), value=1), prepared.image

    def extract(
        file_path: str | None,
        model_id: str,
        temperature: float,
        page_num: float,
        dpi: float,
        streaming: bool,
        max_tokens: float,
    ):
        """Process uploaded file (image or PDF) and extract text with optional streaming."""
        hidden = gr.update(visible=False)
        if not file_path:
            yield "Upload an image or PDF first.", "", "", None, links, hidden
            return
        try:
            prepared = prepare(Path(file_path), page=int(page_num), dpi=int(dpi))
        except RenderError as exc:
            yield str(exc), str(exc), "", None, links, hidden
            return
        yield gr.update(), gr.update(), prepared.info, prepared.image, links, hidden
        shown = ""
        started: float | None = None
        try:
            chunks = engine.stream(
                prepared.image,
                model_id=model_id,
                temperature=float(temperature),
                max_new_tokens=int(max_tokens),
            )
            for shown in chunks:
                now = time.perf_counter()
                if started is None:
                    started = now
                lines = _as_lines(shown)
                speed_html = _links_html(
                    dataset,
                    finetune,
                    _speed_label(_word_count(lines), now - started),
                    device_name,
                )
                if streaming:
                    yield (
                        _as_markdown(shown),
                        lines,
                        prepared.info,
                        prepared.image,
                        speed_html,
                        hidden,
                    )
                else:
                    yield (
                        gr.update(),
                        gr.update(),
                        prepared.info,
                        prepared.image,
                        speed_html,
                        hidden,
                    )
            lines = _as_lines(shown)
            save_capture(
                image=prepared.image,
                text=lines,
                source_name=Path(file_path).name,
                model=model_id,
                page=prepared.page,
                page_count=prepared.page_count,
                dpi=int(dpi),
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            elapsed = 0.0 if started is None else time.perf_counter() - started
            speed_html = _links_html(
                dataset,
                finetune,
                _speed_label(_word_count(lines), elapsed),
                device_name,
            )
            export = (
                gr.update(value=_write_export(lines), visible=True)
                if lines.strip()
                else hidden
            )
            yield _as_markdown(shown), lines, prepared.info, prepared.image, speed_html, export
        except ModelError as exc:
            message = str(exc)
            yield message, message, prepared.info, prepared.image, links, hidden

    def clear():
        return (
            None,
            _DEFAULT_MODEL,
            info_text(_DEFAULT_MODEL),
            "*Extracted text will appear here...*",
            "",
            "",
            None,
            1,
            200,
            0.0,
            True,
            2048,
            links,
            gr.update(value=None, visible=False),
        )

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("IBM Plex Sans"),
            gr.themes.GoogleFont("Montserrat"),
            "ui-sans-serif",
            "sans-serif",
        ],
    )
    css = """
    html { color-scheme: light; }
    html.dark { color-scheme: dark; }
    body, .gradio-container {
      background: #f3f6fb !important;
      color: #0f172a !important;
    }
    .gradio-container {
      --layout-gap: 10px;
      padding: 12px 16px 20px !important;
    }
    html.dark body, html.dark .gradio-container {
      background:
        radial-gradient(900px 420px at 8% -10%, #1e3a5f 0%, transparent 55%),
        radial-gradient(700px 380px at 100% 0%, #16324a 0%, transparent 50%),
        #070d18 !important;
      color: #e8eef7 !important;
    }
    footer, footer * { color: #475569 !important; }
    html.dark footer, html.dark footer * { color: #cbd5e1 !important; }
    #sebi-hero {
      border: 1px solid #dbe3ef;
      border-radius: 16px;
      padding: 16px 20px 14px 20px !important;
      background: #ffffff;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      margin-bottom: 8px;
      gap: 0 !important;
    }
    html.dark #sebi-hero {
      border-color: #24344d;
      background: linear-gradient(135deg, #101a2e 0%, #0d1728 70%);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    }
    #sebi-hero .eyebrow {
      color: #2563eb;
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin: 0 0 6px 0;
    }
    html.dark #sebi-hero .eyebrow { color: #8eb6ff; }
    #sebi-hero h1 {
      color: #0f172a;
      letter-spacing: -0.03em;
      margin: 0;
      font-size: 1.45rem;
      line-height: 1.15;
    }
    html.dark #sebi-hero h1 { color: #ffffff !important; }
    #sebi-hero p { color: #334155; margin: 0; line-height: 1.45; }
    html.dark #sebi-hero p,
    html.dark #sebi-hero .sebi-how,
    html.dark #sebi-hero strong { color: #e8eef7 !important; }
    .sebi-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      column-gap: 18px;
      row-gap: 8px;
      align-items: center;
    }
    .sebi-head h1 { grid-column: 1; grid-row: 1; margin: 0; }
    .sebi-head .juspay-brand { grid-column: 2; grid-row: 1; justify-self: end; }
    .sebi-head .sebi-lead {
      grid-column: 1;
      grid-row: 2;
      margin: 0 !important;
    }
    .sebi-head .theme-switch { grid-column: 2; grid-row: 2; justify-self: end; }
    .sebi-sub p {
      margin: 0 0 4px 0 !important;
      line-height: 1.45;
    }
    .sebi-sub p:last-child { margin-bottom: 0 !important; }
    .juspay-brand svg { width: 32px; height: 32px; display: block; }
    .theme-switch {
      position: relative !important;
      width: 64px;
      height: 30px;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      background: #e8eef6;
      padding: 0 9px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex: none;
      box-sizing: border-box;
    }
    .theme-switch .icon {
      width: 15px;
      height: 15px;
      z-index: 1;
      display: block;
    }
    .theme-switch .knob {
      position: absolute !important;
      top: 2px !important;
      left: 2px !important;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: #ffffff;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.28);
      transition: transform 0.18s ease;
      z-index: 0;
    }
    html.dark .theme-switch,
    .dark .theme-switch {
      border-color: #24344d !important;
      background: #0b1220 !important;
    }
    html.dark .theme-switch .knob,
    .dark .theme-switch .knob {
      transform: translateX(34px) !important;
      background: #334155 !important;
      box-shadow: none;
    }
    .gradio-container .block {
      border: 1px solid #24344d !important;
      border-radius: 12px !important;
      box-shadow: none !important;
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      color: #e8eef7 !important;
    }
    .gradio-container textarea,
    .gradio-container input:not([type="checkbox"]),
    .gradio-container select,
    .gradio-container .empty,
    .gradio-container .secondary-wrap,
    .gradio-container .file-preview {
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      color: #e8eef7 !important;
      border-color: #24344d !important;
    }
    .gradio-container .sebi-section,
    .gradio-container .sebi-reading,
    .gradio-container .sebi-meta {
      border: none !important;
      box-shadow: none !important;
      background: transparent !important;
      background-color: transparent !important;
    }
    .gradio-container .sebi-noborder,
    .gradio-container .sebi-noborder .secondary-wrap {
      border: none !important;
      box-shadow: none !important;
    }
    .gradio-container.sebi-light .block:not(.sebi-section):not(.sebi-meta):not(.sebi-noborder),
    .gradio-container.sebi-light textarea,
    .gradio-container.sebi-light input:not([type="checkbox"]),
    .gradio-container.sebi-light select,
    .gradio-container.sebi-light .empty,
    .gradio-container.sebi-light .secondary-wrap,
    .gradio-container.sebi-light .sebi-panel,
    .gradio-container.sebi-light #sebi-hero {
      background: #ffffff !important;
      background-color: #ffffff !important;
      color: #0f172a !important;
      border-color: #d5deea !important;
    }
    .gradio-container.sebi-light .sebi-section,
    .gradio-container.sebi-light .sebi-meta,
    .gradio-container.sebi-light .sebi-noborder,
    .gradio-container.sebi-light #sebi-hero {
      background: transparent !important;
      background-color: transparent !important;
    }
    .gradio-container.sebi-light #sebi-hero {
      background: #ffffff !important;
      background-color: #ffffff !important;
      border-color: #dbe3ef !important;
    }
    .gradio-container.sebi-light #sebi-hero h1,
    .gradio-container.sebi-light #sebi-hero p,
    .gradio-container.sebi-light .sebi-section h3,
    .gradio-container.sebi-light .sebi-meta {
      color: #0f172a !important;
    }
    .sebi-how {
      font-family: Montserrat, ui-sans-serif, system-ui, sans-serif;
      font-size: 14px;
    }
    .sebi-links, .sebi-how { margin: 2px 0 0 0; }
    .sebi-links {
      display: flex;
      align-items: baseline;
      flex-wrap: wrap;
      gap: 6px;
    }
    .sebi-links a { color: #1d4ed8; text-decoration: none; }
    html.dark .sebi-links a,
    .dark .sebi-links a { color: #8eb6ff; }
    .sebi-links a:hover { text-decoration: underline; }
    .sebi-speed {
      font-variant-numeric: tabular-nums;
      color: #1d4ed8;
    }
    html.dark .sebi-speed,
    .dark .sebi-speed { color: #8eb6ff; }
    #sebi-hero .block,
    #sebi-hero .html-container,
    #sebi-hero .html-container.padding {
      border: none !important;
      background: transparent !important;
      background-color: transparent !important;
      box-shadow: none !important;
      padding: 0 !important;
      margin: 0 !important;
      min-height: 0 !important;
      overflow: visible !important;
    }
    .sebi-panel .html-container.padding {
      padding: 2px 0 !important;
    }
    .sebi-panel {
      background: #0e1626 !important;
      border: 1px solid #24344d !important;
      border-radius: 18px !important;
      padding: 4px 12px 8px 12px !important;
      color: #e8eef7 !important;
    }
    html.sebi-light .sebi-panel,
    .gradio-container.sebi-light .sebi-panel {
      background: #ffffff !important;
      border: 1px solid #dbe3ef !important;
      color: #0f172a !important;
    }
    .gradio-container .sebi-output {
      background: #0e1626 !important;
      border: 1px solid #24344d !important;
      border-radius: 12px !important;
      min-height: 160px;
      color: #e8eef7 !important;
    }
    .gradio-container.sebi-light .sebi-output {
      background: #ffffff !important;
      border-color: #d5deea !important;
      color: #0f172a !important;
    }
    html.dark .sebi-panel,
    .gradio-container.dark .sebi-panel {
      background: #0e1626 !important;
      border: 1px solid #24344d !important;
      color: #e8eef7 !important;
    }
    .gradio-container.dark #sebi-hero {
      border-color: #24344d !important;
      background: linear-gradient(135deg, #101a2e 0%, #0d1728 70%) !important;
    }
    .gradio-container.dark #sebi-hero h1,
    .gradio-container.dark #sebi-hero p,
    .gradio-container.dark #sebi-hero strong,
    .gradio-container.dark .sebi-section h3,
    .gradio-container.dark .sebi-meta,
    .gradio-container.dark .sebi-reading {
      color: #e8eef7 !important;
    }
    .gradio-container.dark .theme-switch {
      border-color: #24344d !important;
      background: #0b1220 !important;
    }
    .gradio-container.dark .theme-switch .knob {
      transform: translateX(34px) !important;
      background: #334155 !important;
    }
    .sebi-section h3,
    .sebi-meta,
    .sebi-reading {
      color: #e8eef7;
      font-size: 15px;
      margin: 4px 0 2px 0;
    }
    html.sebi-light .sebi-section h3,
    html.sebi-light .sebi-meta,
    html.sebi-light .sebi-reading { color: #0f172a !important; }
    button.primary {
      background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%) !important;
      border: none !important;
      color: #f8fafc !important;
      border-radius: 10px !important;
    }
    button.secondary {
      background: #152033 !important;
      color: #dbe7f7 !important;
      border: 1px solid #334155 !important;
      border-radius: 10px !important;
    }
    .sebi-panel.sebi-light button.secondary {
      background: #ffffff !important;
      color: #0f172a !important;
      border-color: #cbd5e1 !important;
    }
    .sebi-panel.sebi-light,
    .block.sebi-light:not(.sebi-section):not(.sebi-meta):not(.sebi-noborder):not(.sebi-plain):not(.sebi-output),
    .block.sebi-output.sebi-light:not(:has(textarea)),
    #sebi-hero.sebi-light {
      background: #ffffff !important;
      background-color: #ffffff !important;
      color: #0f172a !important;
      border: 1px solid #d5deea !important;
    }
    .sebi-section.sebi-light,
    .sebi-meta.sebi-light,
    .sebi-noborder,
    .sebi-noborder.sebi-light,
    .sebi-noborder .wrap,
    .sebi-noborder .wrap-inner,
    .sebi-noborder .secondary-wrap,
    .sebi-noborder select {
      background: transparent !important;
      background-color: transparent !important;
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
    }
    #sebi-hero.sebi-light h1,
    #sebi-hero.sebi-light p,
    .sebi-section.sebi-light h3,
    .sebi-meta.sebi-light {
      color: #0f172a !important;
    }
    .form,
    .form:has(.sebi-noborder),
    .form:has(.sebi-plain),
    .form:has(.sebi-output) {
      background: transparent !important;
      background-color: transparent !important;
      border: none !important;
      box-shadow: none !important;
    }
    .sebi-output,
    .sebi-output .prose {
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      color: #e8eef7 !important;
      border: 1px solid #24344d !important;
      border-radius: 12px !important;
      box-shadow: none !important;
      min-height: 240px;
    }
    .sebi-output textarea {
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      color: #e8eef7 !important;
      border: none !important;
      box-shadow: none !important;
      min-height: 240px !important;
    }
    .sebi-panel.sebi-light,
    .sebi-panel.sebi-light textarea,
    .sebi-panel.sebi-light input:not([type="checkbox"]),
    .sebi-panel.sebi-light select,
    .sebi-panel.sebi-light .secondary-wrap,
    .sebi-panel.sebi-light .wrap,
    .sebi-panel.sebi-light .empty,
    .sebi-panel.sebi-light .file-preview,
    .block.sebi-light textarea,
    .block.sebi-light input:not([type="checkbox"]),
    .block.sebi-light select,
    .block.sebi-light .secondary-wrap,
    .block.sebi-light .wrap,
    .block.sebi-light .empty,
    .block.sebi-light .file-preview,
    .sebi-output.sebi-light,
    .sebi-output.sebi-light textarea,
    .sebi-output.sebi-light .prose {
      background: #ffffff !important;
      background-color: #ffffff !important;
      color: #0f172a !important;
      border-color: #d5deea !important;
      box-shadow: none !important;
    }
    .sebi-noborder.sebi-light,
    .sebi-noborder.sebi-light .secondary-wrap,
    .sebi-noborder.sebi-light .wrap,
    .sebi-panel.sebi-light .sebi-noborder .wrap,
    .sebi-panel.sebi-light .sebi-noborder .secondary-wrap,
    .sebi-output.sebi-light textarea {
      background: transparent !important;
      background-color: transparent !important;
      border: none !important;
      box-shadow: none !important;
      color: #0f172a !important;
    }
    .sebi-output.sebi-light textarea {
      background: #ffffff !important;
      background-color: #ffffff !important;
    }
    html.dark .gradio-container .column:not(.sebi-panel),
    html.dark .gradio-container .row,
    .gradio-container.dark .column:not(.sebi-panel),
    .gradio-container.dark .row {
      background: transparent !important;
      border-color: transparent !important;
    }
    html.dark .gradio-container .sebi-panel,
    .gradio-container.dark .sebi-panel {
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      border: 1px solid #24344d !important;
      color: #e8eef7 !important;
    }
    .block.sebi-plain,
    .block.sebi-plain.sebi-light {
      border: none !important;
      background: transparent !important;
      background-color: transparent !important;
      box-shadow: none !important;
    }
    .sebi-noborder .secondary-wrap {
      border: 1px solid #24344d !important;
      border-radius: 10px !important;
      padding: 10px 14px !important;
      background: transparent !important;
      background-color: transparent !important;
      box-shadow: none !important;
    }
    .sebi-noborder.sebi-light .secondary-wrap,
    .sebi-panel.sebi-light .sebi-noborder .secondary-wrap {
      border: 1px solid #d5deea !important;
      background: #ffffff !important;
      background-color: #ffffff !important;
      color: #0f172a !important;
    }
    .block.sebi-output:has(textarea),
    .block.sebi-output.sebi-light:has(textarea) {
      border: none !important;
      background: transparent !important;
      background-color: transparent !important;
      box-shadow: none !important;
      padding: 0 !important;
      min-height: 0 !important;
    }
    .block.sebi-output:has(textarea) textarea {
      border: 1px solid #24344d !important;
      border-radius: 12px !important;
      padding: 16px 18px !important;
      min-height: 240px !important;
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      color: #e8eef7 !important;
      box-shadow: none !important;
    }
    .block.sebi-output.sebi-light:has(textarea) textarea {
      border: 1px solid #d5deea !important;
      background: #ffffff !important;
      background-color: #ffffff !important;
      color: #0f172a !important;
    }
    .block.sebi-output:not(:has(textarea)) {
      padding: 16px 18px !important;
    }
    .block.sebi-output:not(:has(textarea)) .prose,
    .prose.sebi-output {
      border: none !important;
      background: transparent !important;
      background-color: transparent !important;
      box-shadow: none !important;
      min-height: 0 !important;
      padding: 0 !important;
    }
    .sebi-panel .block:not(.sebi-output) {
      overflow: visible !important;
    }
    input[type="checkbox"] {
      appearance: auto !important;
      -webkit-appearance: auto !important;
      accent-color: #2563eb !important;
      background: transparent !important;
      background-image: none !important;
      width: 16px !important;
      height: 16px !important;
      margin: 0 8px 0 0 !important;
      vertical-align: middle;
    }
    """
    head = """
    <script>
    (() => {
      const key = "indic-vachak-theme";
      let dark = true;
      const paint = () => {
        document.documentElement.classList.toggle("dark", dark);
        document.documentElement.classList.toggle("sebi-light", !dark);
        document.body?.classList.toggle("dark", dark);
        document.querySelectorAll(".gradio-container, gradio-app").forEach((node) => {
          node.classList.toggle("dark", dark);
          node.classList.toggle("sebi-light", !dark);
        });
        const marked = ".sebi-panel, .block, #sebi-hero, .sebi-output";
        document.querySelectorAll(marked).forEach((node) => {
          node.classList.toggle("sebi-light", !dark);
        });
        const toggle = document.getElementById("theme-toggle");
        if (toggle) toggle.setAttribute("aria-checked", dark ? "true" : "false");
      };
      const apply = (next) => {
        dark = next;
        localStorage.setItem(key, dark ? "dark" : "light");
        paint();
      };
      const saved = localStorage.getItem(key);
      apply(saved ? saved === "dark" : true);
      document.addEventListener("DOMContentLoaded", paint);
      setTimeout(paint, 400);
      setTimeout(paint, 1200);
      const watch = new MutationObserver(() => paint());
      const startWatch = () => {
        if (!document.body) return;
        watch.observe(document.body, { childList: true, subtree: true });
      };
      if (document.body) startWatch();
      else document.addEventListener("DOMContentLoaded", startWatch);
      document.addEventListener("click", (event) => {
        if (!event.target.closest("#theme-toggle")) return;
        apply(!dark);
      });
    })();
    </script>
    """
    logo = (_ROOT / "static" / "juspay-logo.svg").read_text(encoding="utf-8")
    device_name = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(
        engine.device, engine.device
    )
    header = f"""
<div class="sebi-head">
  <h1>Indic Vachak - SEBI</h1>
  <div class="juspay-brand">{logo}</div>
  <p class="sebi-lead">⚡ Marathi and Gujarati OCR for a PNG, JPEG, or PDF page.</p>
  <button type="button" id="theme-toggle" class="theme-switch"
    role="switch" aria-checked="true" aria-label="Toggle color theme">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="4" fill="#f59e0b"/>
          <g stroke="#f59e0b" stroke-width="2" stroke-linecap="round">
            <path d="M12 2v2M12 20v2M2 12h2M20 12h2"/>
            <path d="M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4"/>
            <path d="M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/>
          </g>
        </svg>
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
          <path fill="#e2e8f0" d="M15 3a8 8 0 1 0 6 13 7 7 0 1 1-6-13z"/>
        </svg>
        <span class="knob"></span>
  </button>
</div>
"""
    links = _links_html(dataset, finetune, device=device_name)

    def load_example(evt: gr.SelectData):
        """Load selected example image into file input."""
        return str(_EXAMPLES[evt.index][0])

    with gr.Blocks(title="Indic Vachak - SEBI", theme=theme, css=css, head=head) as demo:
        with gr.Column(elem_id="sebi-hero"):
            gr.HTML(header, elem_classes=["sebi-section"])
            speed_line = gr.HTML(links, elem_classes=["sebi-section"])
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["sebi-panel"]):
                gr.Markdown("### Document", elem_classes=["sebi-section"])
                model_selector = gr.Dropdown(
                    choices=choices,
                    value=_DEFAULT_MODEL,
                    label="Model",
                    info="Shown even when the weights are not on this machine yet.",
                    elem_classes=["sebi-noborder"],
                )
                model_info = gr.Markdown(
                    value=info_text(_DEFAULT_MODEL),
                    elem_classes=["sebi-meta"],
                )
                file_input = gr.File(
                    label="Upload Image or PDF",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                rendered_image = gr.Image(
                    label="Preview",
                    type="pil",
                    height=320,
                    interactive=False,
                )
                gr.Markdown("### Page setup", elem_classes=["sebi-section"])
                num_pages = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    label="PDF: Page Number",
                    info="Select which page to extract",
                    elem_classes=["sebi-plain"],
                )
                dpi = gr.Slider(
                    minimum=72,
                    maximum=400,
                    value=200,
                    step=1,
                    label="DPI",
                    info="PDF resolution. The long side is capped at 2048 px. Images ignore DPI.",
                    elem_classes=["sebi-plain"],
                )
                page_info = gr.Textbox(
                    label="Processing Info",
                    value="",
                    interactive=False,
                    elem_classes=["sebi-plain"],
                )
                gr.Markdown("### Generation", elem_classes=["sebi-section"])
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Temperature",
                    info="0.0 keeps the reading fixed. Higher values vary the text.",
                    elem_classes=["sebi-plain"],
                )
                enable_streaming = gr.Checkbox(
                    label="Enable Streaming",
                    value=True,
                    info="Show text as it is generated",
                    elem_classes=["sebi-plain"],
                )
                max_output_tokens = gr.Slider(
                    minimum=256,
                    maximum=8192,
                    value=2048,
                    step=256,
                    label="Max Output Tokens",
                    info="Longest text the model may write for this page",
                    elem_classes=["sebi-plain"],
                )
                with gr.Row():
                    submit_btn = gr.Button("Extract text", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=2, elem_classes=["sebi-panel"]):
                gr.Markdown("### Extracted text", elem_classes=["sebi-section"])
                output_text = gr.Markdown(
                    value="*Extracted text will appear here...*",
                    elem_classes=["sebi-output"],
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )
                export_btn = gr.DownloadButton(
                    "Export",
                    visible=False,
                    size="sm",
                    variant="secondary",
                    elem_classes=["sebi-plain"],
                )
        with gr.Row():
            with gr.Column():
                raw_output = gr.Textbox(
                    label="Raw Markdown Output",
                    placeholder="Raw text will appear here...",
                    lines=18,
                    max_lines=40,
                    elem_classes=["sebi-output"],
                )
        with gr.Accordion("Example documents (click an image to load)", open=False):
            example_gallery = gr.Gallery(
                value=[(str(path), caption) for path, caption in _EXAMPLES],
                show_label=False,
                columns=4,
                rows=1,
                height=220,
                allow_preview=False,
                object_fit="contain",
                elem_id="sebi-examples",
            )

        submit_btn.click(
            fn=extract,
            api_name="process_input",
            inputs=[
                file_input,
                model_selector,
                temperature,
                num_pages,
                dpi,
                enable_streaming,
                max_output_tokens,
            ],
            outputs=[output_text, raw_output, page_info, rendered_image, speed_line, export_btn],
        )
        file_input.change(
            fn=preview,
            api_name="update_slider_and_preview",
            inputs=[file_input],
            outputs=[num_pages, rendered_image],
        )
        example_gallery.select(
            fn=load_example,
            api_name="load_example_image",
            outputs=[file_input],
        )
        model_selector.change(
            fn=info_text,
            api_name="get_model_info_text",
            inputs=[model_selector],
            outputs=[model_info],
        )
        clear_btn.click(
            api_name="lambda",
            fn=clear,
            outputs=[
                file_input,
                model_selector,
                model_info,
                output_text,
                raw_output,
                page_info,
                rendered_image,
                num_pages,
                dpi,
                temperature,
                enable_streaming,
                max_output_tokens,
                speed_line,
                export_btn,
            ],
        )
    return demo
