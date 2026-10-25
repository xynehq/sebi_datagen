"""Gradio page that matches the LightOn OCR demo, plus a DPI control."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gradio as gr

from capture import save_capture
from engine import Engine
from finalize import finalize_document_markdown, stream_preview_markdown
from reading import reading_markdown, reading_raw
from auth import check_ui_model_admin, model_admin_required
from model_jobs import get_job_status
from model_management import (
    register_from_validation,
    remove_custom,
    resolve_add_model_urls,
    resolve_default_model_id,
    set_default,
    validate_add_model_urls,
    validate_hf_url,
)
from registry import (
    ModelError,
    list_user_models,
    load_specs,
    model_status,
)
from render import RenderError, page_count, prepare

_FAMILY_CHOICES = [
    ("Chitrapathak page reader", "chitrapathak"),
    ("IndicOCR reader", "indicocr"),
]

_DEFAULT_MODEL = "chitrapathak-marguj-mixed"
_ROOT = Path(__file__).resolve().parent
_EXAMPLES = (
    (_ROOT / "examples" / "english_notice.png", "English notice"),
    (_ROOT / "examples" / "marathi_sample.png", "Marathi sample"),
    (_ROOT / "examples" / "gujarati_sample.png", "Gujarati sample"),
    (_ROOT / "examples" / "gujarati_holidays_2015.jpg", "Gujarati holidays table"),
    (_ROOT / "examples" / "registration_slip.png", "Registration slip"),
)


def _model_choices() -> list[tuple[str, str]]:
    """Read models.yaml. The dropdown is built from this, not from hard-coded ids."""
    return [(spec.label, spec.id) for spec in load_specs()]


def _hf_link_presets() -> list[tuple[str, str]]:
    """Built-in Hugging Face links users can pick instead of pasting."""
    seen: set[str] = set()
    choices: list[tuple[str, str]] = [("— Paste your own link below —", "")]
    for spec in load_specs():
        if spec.user_added:
            continue
        if spec.adapter and spec.adapter not in seen:
            seen.add(spec.adapter)
            choices.append(
                (f"{spec.label} (adapter)", f"https://huggingface.co/{spec.adapter}")
            )
        if spec.base not in seen:
            seen.add(spec.base)
            choices.append((f"{spec.label} (base)", f"https://huggingface.co/{spec.base}"))
    return choices


def _default_model_id() -> str:
    try:
        return resolve_default_model_id()
    except Exception:
        ids = {spec.id for spec in load_specs()}
        if _DEFAULT_MODEL in ids:
            return _DEFAULT_MODEL
        return next(iter(ids))


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
    shown = speed if speed else "0 words/s"
    speed_html = f' · <span class="sebi-speed">⚡ {shown}</span>'
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
    dataset = (
        "https://drive.google.com/drive/folders/"
        "1BZ8GoMuW1glTLcNn5qaBx6RVgkp6QHSE?usp=drive_link"
    )
    finetune = "https://huggingface.co/snxtyle"
    links = _links_html(dataset, finetune)
    start_model = _default_model_id()

    def info_text(model_id: str | None):
        """Refresh description, base, and adapter when the Model changes."""
        chosen = (model_id or "").strip() or _default_model_id()
        try:
            spec = next(item for item in load_specs() if item.id == chosen)
        except StopIteration:
            return gr.update(value=f"Unknown model: `{chosen}`.")
        status = model_status(spec)
        state = {
            "ready": "Weights are on this machine.",
            "not_downloaded": "Weights are not downloaded yet. The first Extract downloads them.",
            "repo_not_set": status.get("detail") or "Repo id is not set.",
        }.get(str(status["status"]), str(status["status"]))
        if spec.adapter:
            adapter_line = f"`{spec.adapter}`"
        else:
            adapter_line = "`none`"
        if spec.adapter_path_env and os.environ.get(spec.adapter_path_env, "").strip():
            state = f"{state} Adapter weights are loaded from env `{spec.adapter_path_env}`."
        return gr.update(
            value=(
                f"**Model:** `{spec.id}`\n\n"
                f"**Description:** {spec.description}\n\n"
                f"**Weights:** {state}\n\n"
                f"**Base:** `{spec.base}`\n\n"
                f"**Adapter:** {adapter_line}"
            )
        )

    def refresh_models(selected: str | None = None):
        """Re-read the model list so a new row appears without editing code."""
        choices = _model_choices()
        if not choices:
            return gr.update(choices=[], value=None), gr.update(value="No models configured.")
        ids = {item[1] for item in choices}
        current = (selected or "").strip()
        fallback = _default_model_id()
        if fallback not in ids:
            fallback = choices[0][1]
        value = current if current in ids else fallback
        return (
            gr.update(choices=choices, value=value, interactive=True, filterable=False),
            info_text(value),
        )

    def _custom_models_markdown() -> str:
        from model_store import list_custom_models

        rows = list_custom_models(include_disabled=True)
        if not rows:
            return "**Your added models:** none yet."
        lines = ["**Your added models** (OCR tab → Model list when status is *ready*):"]
        for row in rows:
            if row.status == "ready" and row.enabled:
                state = "ready — select on OCR tab"
            elif row.status == "failed":
                state = f"failed — {row.failure_reason or 'see status below'}"
            else:
                state = row.status
            lines.append(f"- **{row.label}** (`{row.id}`): {state}")
        return "\n".join(lines)

    def _user_remove_choices() -> list[tuple[str, str]]:
        from model_store import list_custom_models

        rows = list_custom_models(include_disabled=True)
        if not rows:
            return [("No added models yet", "")]
        return [
            (f"{row.label} ({row.id}) — {row.status}", row.id)
            for row in rows
        ]

    def _failed_model_choices() -> list[tuple[str, str]]:
        from model_store import list_custom_models

        failed = [
            row for row in list_custom_models(include_disabled=True) if row.status == "failed"
        ]
        if not failed:
            return [("No failed models", "")]
        return [(f"{row.label} ({row.id})", row.id) for row in failed]

    def _admin_gate(admin_key: str) -> None:
        check_ui_model_admin(admin_key=admin_key)

    def _suggest_display_name(base_hf_url: str, adapter_hf_url: str) -> str:
        from hf_inspect import parse_hf_url

        primary = (adapter_hf_url or base_hf_url or "").strip()
        if not primary:
            return ""
        try:
            repo_id, _ = parse_hf_url(primary)
        except ModelError:
            return ""
        slug = repo_id.split("/")[-1]
        return slug.replace("_", " ").replace("-", " ").strip().title()

    def sync_display_name(base_hf_url: str, adapter_hf_url: str):
        return gr.update(value=_suggest_display_name(base_hf_url, adapter_hf_url))

    def _format_validation(data: dict) -> str:
        lines = [
            f"**Type:** {data.get('model_type', 'unknown')}",
            f"**Repository:** `{data.get('repo_id')}`",
        ]
        if data.get("revision"):
            lines.append(f"**Revision:** `{data['revision']}`")
        if data.get("base_repo_id"):
            lines.append(f"**Base model:** `{data['base_repo_id']}`")
        if data.get("architecture"):
            lines.append(f"**Architecture:** {data['architecture']}")
        if data.get("size_bytes"):
            mb = int(data["size_bytes"]) / (1024 * 1024)
            lines.append(f"**Repository size (approx.):** {mb:.1f} MB")
        lines.append(
            "_Disk size is not the same as GPU memory needed at inference time._"
        )
        if data.get("warnings"):
            lines.append("**Warnings:**")
            for item in data["warnings"]:
                lines.append(f"- {item}")
        lines.append(f"**Result:** {data.get('message', '')}")
        return "\n\n".join(lines)

    def _format_validation_plain(data: dict) -> str:
        """Plain text for the Add model status box (readable without Markdown)."""
        lines = [
            f"Type: {data.get('model_type', 'unknown')}",
            f"Repository: {data.get('repo_id')}",
        ]
        if data.get("revision"):
            lines.append(f"Revision: {data['revision']}")
        if data.get("base_repo_id"):
            lines.append(f"Base model: {data['base_repo_id']}")
        if data.get("architecture"):
            lines.append(f"Architecture: {data['architecture']}")
        if data.get("size_bytes"):
            mb = int(data["size_bytes"]) / (1024 * 1024)
            lines.append(f"Repository size (approx.): {mb:.1f} MB")
        lines.append(
            "Note: disk size is not the same as GPU memory needed at inference."
        )
        if data.get("warnings"):
            lines.append("Warnings:")
            for item in data["warnings"]:
                lines.append(f"  - {item}")
        lines.append(f"Result: {data.get('message', '')}")
        return "\n".join(lines)

    def _incompatible_headline(data: dict) -> str:
        msg = str(data.get("message") or "").lower()
        if "tesseract" in msg or "traineddata" in msg:
            return (
                "CANNOT ADD — Use a Hugging Face vision / image-to-text model for page OCR, "
                "not Tesseract .traineddata.\n\n"
            )
        if "text-only" in msg:
            return "CANNOT ADD — This repo looks text-only. Use an image-to-text vision model.\n\n"
        return "CANNOT ADD — This repo is not a downloadable vision OCR model.\n\n"

    def apply_hf_preset(preset_url: str):
        url = (preset_url or "").strip()
        if not url:
            return gr.update(), gr.update()
        lowered = url.lower()
        is_lora = "adapter_config" not in lowered and any(
            m in lowered
            for m in (
                "snxtyle/chitrapathak",
                "chitrapathak_marguj",
                "chitrapathak-marguj",
                "snxtyle/indicocr",
                "indicocr-marguj",
            )
        )
        if is_lora and "chitrapathak-2" not in lowered:
            if "indic" in lowered:
                return (
                    gr.update(value="https://huggingface.co/bodhan-ai/indic-ocr"),
                    gr.update(value=url),
                )
            return (
                gr.update(value="https://huggingface.co/krutrim-ai-labs/Chitrapathak-2"),
                gr.update(value=url),
            )
        return gr.update(value=url), gr.update(value="")

    def validate_model_url(
        base_hf_url: str,
        adapter_hf_url: str,
        family_label: str,
        admin_key: str,
    ):
        if not (base_hf_url or "").strip():
            return "Paste the Hugging Face base model link.", None, gr.update()
        if model_admin_required() and not (admin_key or "").strip():
            return (
                "Enter the admin key (same as OCR_API_KEY on this server).",
                None,
                gr.update(),
            )
        try:
            _admin_gate(admin_key)
            data = validate_add_model_urls(
                base_hf_url,
                adapter_hf_url,
                family_hint=family_label,
                admin_key=admin_key,
            )
        except ModelError as exc:
            return f"Could not validate: {exc}", None, gr.update()
        suggested = _suggest_display_name(base_hf_url, adapter_hf_url) or str(
            data.get("label_suggestion") or ""
        )
        label_update = gr.update(value=suggested)
        plain = _format_validation_plain(data)
        if data.get("base_url") and data.get("adapter_url"):
            plain = (
                f"Base link: {data['base_url']}\n"
                f"Adapter link: {data['adapter_url']}\n\n{plain}"
            )
        elif data.get("base_url"):
            plain = f"Base link: {data['base_url']}\n\n{plain}"
        if not data.get("compatible"):
            return _incompatible_headline(data) + plain, None, label_update
        return (
            "OK — Click Download & add to install this model.\n\n" + plain,
            data,
            label_update,
        )

    def register_model_download(
        base_hf_url: str,
        adapter_hf_url: str,
        display_name: str,
        label_override: str,
        note: str,
        family_label: str,
        admin_key: str,
        validation_data: dict | None,
        selected: str | None,
    ):
        _md = gr.update(value=_custom_models_markdown())
        if not (base_hf_url or "").strip():
            return (
                "Paste the Hugging Face base model link.",
                gr.update(),
                gr.update(),
                gr.update(choices=_user_remove_choices()),
                "",
                gr.update(),
                None,
                _md,
            )
        if model_admin_required() and not (admin_key or "").strip():
            return (
                "Enter the admin key to add a model.",
                gr.update(),
                gr.update(),
                gr.update(choices=_user_remove_choices()),
                "",
                gr.update(),
                None,
                _md,
            )
        data = validation_data
        summary = ""
        if not data:
            try:
                _admin_gate(admin_key)
                data = validate_add_model_urls(
                    base_hf_url,
                    adapter_hf_url,
                    family_hint=family_label,
                    admin_key=admin_key,
                )
                summary = _format_validation_plain(data)
            except ModelError as exc:
                return (
                    f"Could not add model: {exc}",
                    gr.update(),
                    gr.update(),
                    gr.update(choices=_user_remove_choices()),
                    "",
                    gr.update(),
                    None,
                    _md,
                )
        if not data.get("compatible"):
            body = summary or _format_validation_plain(data)
            return (
                _incompatible_headline(data) + body,
                gr.update(),
                gr.update(),
                gr.update(choices=_user_remove_choices()),
                "",
                gr.update(
                    value=_suggest_display_name(base_hf_url, adapter_hf_url)
                    or data.get("label_suggestion")
                    or ""
                ),
                None,
                _md,
            )
        try:
            _admin_gate(admin_key)
            register_url = str(data.get("register_url") or "")
            base_override = str(data.get("register_base_override") or "")
            if not register_url:
                register_url, base_override = resolve_add_model_urls(
                    base_hf_url, adapter_hf_url
                )
            auto_label = (
                (label_override or "").strip()
                or (display_name or "").strip()
                or _suggest_display_name(base_hf_url, adapter_hf_url)
                or str(data.get("label_suggestion") or "")
            )
            payload = register_from_validation(
                url=register_url,
                label=auto_label,
                description=note or "",
                languages="unknown",
                family=family_label,
                base_override=base_override,
                admin_key=admin_key,
            )
        except ModelError as exc:
            return (
                f"Could not start download: {exc}",
                gr.update(),
                gr.update(),
                gr.update(choices=_user_remove_choices()),
                "",
                gr.update(),
                None,
                _md,
            )
        job_id = str(payload.get("job_id") or "")
        model_id = str((payload.get("model") or {}).get("id") or "")
        model_update, info_update = refresh_models(selected)
        return (
            (summary + "\n\n" if summary else "")
            + f"Download started for {model_id or 'new model'}.\n"
            + "Watch Status below. When it says ready, open the OCR tab and choose it in Model.",
            model_update,
            info_update,
            gr.update(choices=_user_remove_choices()),
            job_id,
            gr.update(),
            data,
            gr.update(value=_custom_models_markdown()),
        )

    def poll_download_job(job_id: str, selected: str | None):
        empty_job = gr.update()
        models_md = gr.update(value=_custom_models_markdown())
        if not (job_id or "").strip():
            return gr.update(), gr.update(), "", empty_job, models_md
        try:
            status = get_job_status(job_id.strip())
        except ModelError as exc:
            return gr.update(), gr.update(), f"Status: {exc}", empty_job, models_md
        message = f"Status: {status.get('message', '')}"
        progress = status.get("progress")
        if progress is not None:
            message += f" ({int(float(progress) * 100)}%)"
        models_md = gr.update(value=_custom_models_markdown())
        if status.get("status") == "ready":
            model_id = str(status.get("model_id") or "")
            model_update, info_update = refresh_models(model_id or selected)
            return (
                model_update,
                info_update,
                message + "\n\nReady. Open the OCR tab and pick it in Model.",
                gr.update(value=""),
                models_md,
            )
        if status.get("status") == "failed":
            return gr.update(), gr.update(), message, gr.update(value=""), models_md
        return gr.update(), gr.update(), message, gr.update(), models_md

    def retry_failed_model(model_id: str, admin_key: str):
        from model_jobs import retry_failed_job

        if not (model_id or "").strip():
            return (
                "Pick a failed model below, then click Retry.",
                gr.update(),
                gr.update(),
                "",
                gr.update(value=_custom_models_markdown()),
            )
        try:
            _admin_gate(admin_key)
            job_id = retry_failed_job(model_id.strip())
        except ModelError as exc:
            return (
                str(exc),
                gr.update(),
                gr.update(),
                "",
                gr.update(value=_custom_models_markdown()),
            )
        return (
            "Retry started. Watch Status below; when ready, use Refresh model list on the OCR tab.",
            gr.update(),
            gr.update(),
            job_id,
            gr.update(value=_custom_models_markdown()),
        )

    def remove_model_from_form(model_id: str, selected: str | None, admin_key: str):
        _md = gr.update(value=_custom_models_markdown())
        _retry = gr.update(choices=_failed_model_choices(), value="")
        if not (model_id or "").strip():
            return (
                "Choose a model under Remove an added model, then click Remove.",
                gr.update(),
                gr.update(),
                gr.update(choices=_user_remove_choices()),
                _md,
                _retry,
            )
        try:
            _admin_gate(admin_key)
            remove_custom(model_id.strip(), active_model_id=selected, admin_key=admin_key)
        except ModelError as exc:
            return (
                f"Could not remove the model: {exc}",
                gr.update(),
                gr.update(),
                gr.update(choices=_user_remove_choices()),
                _md,
                _retry,
            )
        if selected and selected.strip() == model_id.strip():
            selected = _default_model_id()
        model_update, info_update = refresh_models(selected)
        return (
            "Removed that model from your list. Weight files in the cache were kept.",
            model_update,
            info_update,
            gr.update(choices=_user_remove_choices(), value=""),
            gr.update(value=_custom_models_markdown()),
            gr.update(choices=_failed_model_choices(), value=""),
        )

    def make_default_model(model_id: str | None, admin_key: str):
        if not (model_id or "").strip():
            return "Pick a model in the list above first."
        try:
            _admin_gate(admin_key)
            set_default(model_id.strip(), admin_key=admin_key)
        except ModelError as exc:
            return f"Could not set default: {exc}"
        return (
            f"**{model_id}** will be selected when you open this page. "
            "It does not change the built-in factory default for other users."
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
                raw = reading_raw(shown)
                speed_html = _links_html(
                    dataset,
                    finetune,
                    _speed_label(_word_count(raw), now - started),
                    device_name,
                )
                if streaming:
                    yield (
                        stream_preview_markdown(shown),
                        raw,
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
            finalized = finalize_document_markdown(
                prepared.image,
                shown,
                model_id=model_id,
            )
            shown = finalized
            layout_note = ""
            if "<table" in shown.lower():
                layout_note = " · HTML table (row-aligned)."
            elif "|" in shown and "---" in shown:
                layout_note = " · Markdown table (row-aligned)."
            else:
                import shutil

                if not shutil.which("tesseract"):
                    layout_note = " · Install tesseract (+ guj lang) for table row alignment."
            raw = reading_raw(shown)
            save_capture(
                image=prepared.image,
                text=raw,
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
                _speed_label(_word_count(raw), elapsed),
                device_name,
            )
            export = (
                gr.update(value=_write_export(raw), visible=True)
                if raw.strip()
                else hidden
            )
            page_info = f"{prepared.info}{layout_note}"
            yield reading_markdown(raw), raw, page_info, prepared.image, speed_html, export
        except ModelError as exc:
            message = str(exc)
            yield message, message, prepared.info, prepared.image, links, hidden

    def clear():
        model_update, info_update = refresh_models(_default_model_id())
        return (
            None,
            model_update,
            info_update,
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
      column-gap: 16px;
      row-gap: 8px;
      align-items: center;
    }
    .sebi-head h1 { grid-column: 1; grid-row: 1; margin: 0; }
    .sebi-head .juspay-brand { grid-column: 2; grid-row: 1; justify-self: end; align-self: center; }
    .sebi-head .sebi-lead {
      grid-column: 1;
      grid-row: 2;
      margin: 0 !important;
    }
    .sebi-head .theme-switch {
      grid-column: 2;
      grid-row: 2;
      justify-self: end;
      align-self: center;
    }
    #sebi-hero .sebi-head {
      padding-bottom: 0 !important;
    }
    #sebi-hero .sebi-sub {
      padding-top: 0 !important;
    }
    #sebi-hero .block + .block {
      margin-top: 8px !important;
    }
    #sebi-hero .sebi-sub p {
      margin: 0 0 8px 0 !important;
      line-height: 1.45;
    }
    #sebi-hero .sebi-sub p:last-child { margin-bottom: 0 !important; }
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
    .gradio-container .sebi-panel > .tabs .tabitem .form .block,
    .gradio-container .sebi-panel > .block:not(.sebi-section):not(.sebi-meta):not(.sebi-plain):not(.sebi-noborder):not(.sebi-output) {
      border: 1px solid #24344d !important;
      border-radius: 12px !important;
      box-shadow: none !important;
      background: #0e1626 !important;
      background-color: #0e1626 !important;
      color: #e8eef7 !important;
    }
    #sebi-add-tab .block,
    #sebi-add-tab .form .block {
      border: none !important;
      background: transparent !important;
      box-shadow: none !important;
      padding: 2px 0 !important;
      margin: 0 !important;
    }
    #sebi-add-tab .label-wrap span {
      background: transparent !important;
      border: none !important;
      color: #cbd5e1 !important;
      font-weight: 600 !important;
      padding: 0 !important;
    }
    .gradio-container.sebi-light #sebi-add-tab .label-wrap span {
      color: #334155 !important;
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
    .sebi-copy-btn {
      position: absolute;
      top: 8px;
      right: 10px;
      z-index: 4;
      border: 1px solid #334155;
      background: #152033;
      color: #e8eef7;
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 12px;
      line-height: 1.2;
      cursor: pointer;
    }
    html.sebi-light .sebi-copy-btn {
      background: #ffffff;
      color: #0f172a;
      border-color: #cbd5e1;
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
    .sebi-input-box .wrap,
    .sebi-input-box .wrap-inner,
    .sebi-input-box .secondary-wrap {
      border: 1px solid #64748b !important;
      border-radius: 10px !important;
      padding: 6px 10px !important;
      background: #0e1626 !important;
    }
    .gradio-container.sebi-light .sebi-input-box .wrap,
    .gradio-container.sebi-light .sebi-input-box .secondary-wrap {
      border: 1px solid #94a3b8 !important;
      background: #ffffff !important;
    }
    .sebi-input-box textarea,
    .sebi-input-box input {
      min-height: 2.75rem !important;
    }
    .sebi-add-panel {
      border: 1px solid #475569 !important;
      border-radius: 12px !important;
      padding: 14px !important;
      margin-bottom: 10px !important;
    }
    .gradio-container.sebi-light .sebi-add-panel {
      border-color: #cbd5e1 !important;
      background: #f8fafc !important;
    }
    /* Add model tab: stop every Gradio .block from looking like a blue label pill */
    .sebi-add-panel .block {
      border: none !important;
      background: transparent !important;
      box-shadow: none !important;
      padding: 4px 0 !important;
      margin: 0 !important;
      overflow: visible !important;
    }
    .sebi-add-panel .label-wrap,
    .sebi-add-panel .label-wrap span,
    .sebi-add-panel label span {
      background: transparent !important;
      border: none !important;
      border-radius: 0 !important;
      box-shadow: none !important;
      padding: 0 !important;
      margin: 0 0 6px 0 !important;
      color: #cbd5e1 !important;
      font-size: 0.875rem !important;
      font-weight: 600 !important;
      line-height: 1.3 !important;
      display: block !important;
      width: auto !important;
      max-width: 100% !important;
    }
    .gradio-container.sebi-light .sebi-add-panel .label-wrap span {
      color: #334155 !important;
    }
    #sebi-add-tab .sebi-hf-repo-input .wrap,
    #sebi-add-tab .sebi-hf-repo-input .secondary-wrap {
      border: 1px solid #64748b !important;
      border-radius: 8px !important;
      background: #0b1220 !important;
      padding: 4px 8px !important;
    }
    #sebi-add-tab .sebi-hf-repo-input textarea,
    #sebi-add-tab .sebi-hf-repo-input input {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace !important;
      font-size: 13px !important;
      line-height: 1.45 !important;
      color: #e2e8f0 !important;
      background: transparent !important;
      border: none !important;
    }
    .gradio-container.sebi-light #sebi-add-tab .sebi-hf-repo-input .wrap,
    .gradio-container.sebi-light #sebi-add-tab .sebi-hf-repo-input .secondary-wrap {
      border-color: #94a3b8 !important;
      background: #f8fafc !important;
    }
    .gradio-container.sebi-light #sebi-add-tab .sebi-hf-repo-input textarea,
    .gradio-container.sebi-light #sebi-add-tab .sebi-hf-repo-input input {
      color: #0f172a !important;
    }
    .sebi-meta .prose :where(code):not(pre code) {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace !important;
      font-size: 0.88em !important;
      background: transparent !important;
      border: none !important;
      border-radius: 0 !important;
      padding: 0 !important;
      color: inherit !important;
    }
    .sebi-add-status textarea {
      min-height: 140px !important;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
      font-size: 13px !important;
      line-height: 1.45 !important;
      border: 2px solid #475569 !important;
      border-radius: 10px !important;
      background: #0b1220 !important;
      color: #f1f5f9 !important;
    }
    .gradio-container.sebi-light .sebi-add-status textarea {
      border-color: #94a3b8 !important;
      background: #ffffff !important;
      color: #0f172a !important;
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
        const area = document.querySelector(".sebi-output textarea");
        const host = area && area.parentElement;
        if (host && !host.querySelector(".sebi-copy-btn")) {
          host.style.position = "relative";
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "sebi-copy-btn";
          btn.textContent = "Copy";
          btn.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const value = area.value || "";
            try {
              await navigator.clipboard.writeText(value);
            } catch (err) {
              area.focus();
              area.select();
              document.execCommand("copy");
            }
            btn.textContent = "Copied";
            setTimeout(() => { btn.textContent = "Copy"; }, 1200);
          });
          host.appendChild(btn);
        }
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
                validation_state = gr.State(None)
                job_state = gr.State("")
                with gr.Tabs():
                    with gr.Tab("OCR"):
                        gr.Markdown("### Document", elem_classes=["sebi-section"])
                        model_selector = gr.Dropdown(
                            choices=_model_choices(),
                            value=start_model,
                            label="Model",
                            filterable=False,
                            info=(
                                "Built-in models appear when download status is ready. "
                                "Use **Add model** to install from Hugging Face."
                            ),
                            elem_classes=["sebi-noborder"],
                        )
                        model_info = gr.Markdown(
                            value=info_text(start_model)["value"],
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
                            info="Rendered view updates live (tables as HTML/TSV arrive). Geometry/Tesseract row alignment still runs at the end",
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
                    with gr.Tab("Add model"):
                        with gr.Column(elem_id="sebi-add-tab"):
                            gr.Markdown(
                                "### Add a Hugging Face model\n"
                                "**Base** — full vision model repo, or the foundation model when you use an adapter.\n"
                                "**Adapter** — optional LoRA repo on top of that base (leave empty for a standalone model)."
                            )
                            add_base_hf_url = gr.Textbox(
                                label="Base model (Hugging Face link)",
                                placeholder="https://huggingface.co/krutrim-ai-labs/Chitrapathak-2",
                                lines=1,
                                elem_classes=["sebi-hf-repo-input"],
                            )
                            add_adapter_hf_url = gr.Textbox(
                                label="Adapter (optional Hugging Face link)",
                                placeholder="https://huggingface.co/snxtyle/chitrapathak-MarGuj",
                                lines=1,
                                elem_classes=["sebi-hf-repo-input"],
                            )
                            add_display_name = gr.Textbox(
                                label="Name in model list",
                                placeholder="Updates when you paste a Hugging Face link",
                                lines=1,
                                interactive=False,
                                elem_classes=["sebi-plain"],
                            )
                            with gr.Accordion("Use a different display name", open=False):
                                add_label_override = gr.Textbox(
                                    label="Custom name (optional)",
                                    placeholder="Leave empty to use the automatic name above",
                                    lines=1,
                                    elem_classes=["sebi-plain"],
                                )
                            add_admin_key = gr.Textbox(
                                label="Admin key",
                                placeholder="Required only when OCR_API_KEY is set",
                                type="password",
                                lines=1,
                                visible=model_admin_required(),
                            )
                            with gr.Row():
                                register_btn = gr.Button("Download & add", variant="primary")
                                validate_btn = gr.Button("Check link", variant="secondary")
                            add_status = gr.Textbox(
                                label="Status",
                                value="",
                                lines=10,
                                max_lines=24,
                                interactive=False,
                                placeholder="Validation, download progress, and errors appear here.",
                                elem_classes=["sebi-add-status"],
                            )
                            with gr.Accordion("Example links", open=False):
                                add_preset = gr.Dropdown(
                                    choices=_hf_link_presets(),
                                    value="",
                                    label="Fill base / adapter from an example",
                                    elem_classes=["sebi-noborder"],
                                )
                            add_family = gr.State("")
                            add_note = gr.Textbox(visible=False)
                        custom_models_list = gr.Markdown(
                            value=_custom_models_markdown(),
                            elem_classes=["sebi-meta"],
                        )
                        with gr.Row():
                            refresh_models_btn = gr.Button("Refresh model list", variant="secondary")
                        with gr.Accordion("Retry a failed model", open=False):
                            retry_pick = gr.Dropdown(
                                choices=_failed_model_choices(),
                                value="",
                                label="Failed model",
                                filterable=False,
                                elem_classes=["sebi-noborder"],
                            )
                            retry_btn = gr.Button("Retry download & test", variant="secondary")
                        with gr.Accordion("Remove an added model", open=False):
                            remove_pick = gr.Dropdown(
                                choices=_user_remove_choices(),
                                value="",
                                label="Model to remove",
                                elem_classes=["sebi-noborder"],
                            )
                            remove_btn = gr.Button("Remove", variant="secondary")
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
        model_selector.input(
            fn=info_text,
            inputs=[model_selector],
            outputs=[model_info],
        )
        demo.load(
            fn=refresh_models,
            inputs=[model_selector],
            outputs=[model_selector, model_info],
        )
        add_preset.change(
            fn=apply_hf_preset,
            inputs=[add_preset],
            outputs=[add_base_hf_url, add_adapter_hf_url],
        ).then(
            fn=sync_display_name,
            inputs=[add_base_hf_url, add_adapter_hf_url],
            outputs=[add_display_name],
        )
        add_base_hf_url.change(
            fn=sync_display_name,
            inputs=[add_base_hf_url, add_adapter_hf_url],
            outputs=[add_display_name],
        )
        add_adapter_hf_url.change(
            fn=sync_display_name,
            inputs=[add_base_hf_url, add_adapter_hf_url],
            outputs=[add_display_name],
        )
        validate_btn.click(
            fn=validate_model_url,
            inputs=[add_base_hf_url, add_adapter_hf_url, add_family, add_admin_key],
            outputs=[add_status, validation_state, add_display_name],
        )
        register_btn.click(
            fn=register_model_download,
            inputs=[
                add_base_hf_url,
                add_adapter_hf_url,
                add_display_name,
                add_label_override,
                add_note,
                add_family,
                add_admin_key,
                validation_state,
                model_selector,
            ],
            outputs=[
                add_status,
                model_selector,
                model_info,
                remove_pick,
                job_state,
                add_display_name,
                validation_state,
                custom_models_list,
            ],
        )
        refresh_models_btn.click(
            fn=refresh_models,
            inputs=[model_selector],
            outputs=[model_selector, model_info],
        )
        job_timer = gr.Timer(value=5)
        job_timer.tick(
            fn=poll_download_job,
            inputs=[job_state, model_selector],
            outputs=[model_selector, model_info, add_status, job_state, custom_models_list],
        )
        retry_btn.click(
            fn=retry_failed_model,
            inputs=[retry_pick, add_admin_key],
            outputs=[add_status, model_selector, model_info, job_state, custom_models_list],
        )
        remove_btn.click(
            fn=remove_model_from_form,
            inputs=[remove_pick, model_selector, add_admin_key],
            outputs=[
                add_status,
                model_selector,
                model_info,
                remove_pick,
                custom_models_list,
                retry_pick,
            ],
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
