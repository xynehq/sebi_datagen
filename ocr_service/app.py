"""Page OCR service: Gradio UI plus /v1 API.

Start from the repository root with ./ocr_service/run.sh
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from capture import save_capture
from engine import Engine
from auth import check_model_admin
from model_jobs import get_job_status
from model_management import (
    clear_default,
    list_custom_with_jobs,
    register_from_validation,
    remove_custom,
    set_default,
    validate_hf_url,
)
from model_store import init_store, set_enabled
from registry import ModelError, get_spec, list_models
from document_geometry import geometry_version
from finalize import FINALIZER_VERSION, PAGE_LAYOUT_VERSION, finalize_document_for_response, last_geometry_report
from table_layout import table_layout_version
from render import RenderError, prepare
from ui import build_demo

_SERVICE_DIR = Path(__file__).resolve().parent
_ALLOWED = {".pdf", ".png", ".jpg", ".jpeg"}


def _load_env() -> None:
    path = _SERVICE_DIR.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()
init_store()

engine = Engine()
app = FastAPI(title="Indic Vachak - SEBI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _max_bytes() -> int:
    raw = os.environ.get("OCR_MAX_UPLOAD_BYTES", "").strip() or str(30 * 1024 * 1024)
    try:
        value = int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="OCR_MAX_UPLOAD_BYTES is not a whole number",
        ) from exc
    if value < 1:
        raise HTTPException(status_code=500, detail="OCR_MAX_UPLOAD_BYTES must be at least 1")
    return value


def require_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("OCR_API_KEY", "").strip()
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def _check_params(
    model: str,
    page: int,
    dpi: int,
    temperature: float,
    max_tokens: int,
) -> None:
    try:
        get_spec(model)
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be at least 1")
    if not 72 <= dpi <= 400:
        raise HTTPException(status_code=400, detail="dpi must be between 72 and 400")
    if not 0 <= temperature <= 1:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 1")
    if not 1 <= max_tokens <= 8192:
        raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 8192")


async def _store_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _ALLOWED:
        raise HTTPException(status_code=415, detail="Upload a PDF, PNG, or JPEG")
    limit = _max_bytes()
    descriptor, raw_name = tempfile.mkstemp(suffix=suffix)
    path = Path(raw_name)
    written = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="upload is larger than the allowed size",
                    )
                handle.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if written == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="upload is empty")
    return path


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ModelError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, RenderError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/health")
def health(_auth: None = Depends(require_key)) -> dict[str, object]:
    import shutil

    report = last_geometry_report() or {}
    return {
        "status": "ok",
        "device": engine.device,
        "finalizer": FINALIZER_VERSION,
        "page_layout": PAGE_LAYOUT_VERSION,
        "table_layout": table_layout_version(),
        "document_geometry": geometry_version(),
        "tesseract_on_path": bool(shutil.which("tesseract")),
        "ocr_table_layout": os.environ.get("OCR_TABLE_LAYOUT", "1"),
        "ocr_table_format": os.environ.get("OCR_TABLE_FORMAT", "html"),
        "last_geometry": report,
    }


@app.get("/v1/models")
def models(_auth: None = Depends(require_key)) -> dict[str, object]:
    try:
        rows = list_models()
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {"models": rows}


def _admin_dep(authorization: str | None = Header(default=None)) -> None:
    try:
        check_model_admin(authorization=authorization)
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/v1/models/custom")
def models_custom(_auth: None = Depends(_admin_dep)) -> dict[str, object]:
    return {"models": list_custom_with_jobs()}


@app.post("/v1/models/validate")
def models_validate(
    body: dict[str, object],
    _auth: None = Depends(_admin_dep),
) -> dict[str, object]:
    url = str(body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        result = validate_hf_url(
            url,
            revision=str(body.get("revision") or "") or None,
            family_hint=str(body.get("family") or ""),
            base_override=str(body.get("base_repo_id") or ""),
            _authorized=True,
        )
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {"validation": result}


@app.post("/v1/models/register")
def models_register(
    body: dict[str, object],
    _auth: None = Depends(_admin_dep),
) -> dict[str, object]:
    url = str(body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        payload = register_from_validation(
            url=url,
            label=str(body.get("label") or ""),
            description=str(body.get("description") or ""),
            model_id=str(body.get("model_id") or ""),
            languages=str(body.get("languages") or "unknown"),
            family=str(body.get("family") or ""),
            revision=str(body.get("revision") or "") or None,
            base_override=str(body.get("base_repo_id") or ""),
            _authorized=True,
        )
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return payload


@app.get("/v1/models/jobs/{job_id}")
def models_job(job_id: str, _auth: None = Depends(_admin_dep)) -> dict[str, object]:
    try:
        return get_job_status(job_id)
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.patch("/v1/models/custom/{model_id}")
def models_custom_patch(
    model_id: str,
    body: dict[str, object],
    _auth: None = Depends(_admin_dep),
) -> dict[str, object]:
    if "enabled" in body:
        try:
            set_enabled(model_id, bool(body["enabled"]))
        except ModelError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {"ok": True}


@app.delete("/v1/models/custom/{model_id}")
def models_custom_delete(model_id: str, _auth: None = Depends(_admin_dep)) -> dict[str, object]:
    try:
        remove_custom(model_id, active_model_id=engine._loaded_id)
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/v1/models/custom/{model_id}/default")
def models_custom_default(model_id: str, _auth: None = Depends(_admin_dep)) -> dict[str, object]:
    try:
        set_default(model_id)
    except ModelError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {"ok": True, "default_model_id": model_id}


@app.delete("/v1/models/default")
def models_default_clear(_auth: None = Depends(_admin_dep)) -> dict[str, object]:
    clear_default()
    return {"ok": True}


@app.post("/v1/ocr")
async def ocr(
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form(...),  # noqa: B008
    page: int = Form(1),  # noqa: B008
    dpi: int = Form(200),  # noqa: B008
    temperature: float = Form(0.0),  # noqa: B008
    max_tokens: int = Form(2048),  # noqa: B008
    _auth: None = Depends(require_key),  # noqa: B008
) -> dict[str, object]:
    _check_params(model, page, dpi, temperature, max_tokens)
    path = await _store_upload(file)
    try:
        prepared = prepare(path, page=page, dpi=dpi)
        streamed = ""
        for streamed in engine.stream(
            prepared.image,
            model_id=model,
            temperature=temperature,
            max_new_tokens=max_tokens,
        ):
            pass
        text = finalize_document_for_response(
            prepared.image,
            streamed,
            model_id=model,
        )
        save_capture(
            image=prepared.image,
            text=text,
            source_name=file.filename or path.name,
            model=model,
            page=prepared.page,
            page_count=prepared.page_count,
            dpi=dpi,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (ModelError, RenderError) as exc:
        raise _http_error(exc) from exc
    finally:
        path.unlink(missing_ok=True)
    width, height = prepared.image.size
    return {
        "model": model,
        "page": prepared.page,
        "page_count": prepared.page_count,
        "dpi": dpi,
        "width": width,
        "height": height,
        "text": text,
    }


@app.post("/v1/ocr/stream")
async def ocr_stream(
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form(...),  # noqa: B008
    page: int = Form(1),  # noqa: B008
    dpi: int = Form(200),  # noqa: B008
    temperature: float = Form(0.0),  # noqa: B008
    max_tokens: int = Form(2048),  # noqa: B008
    _auth: None = Depends(require_key),  # noqa: B008
) -> StreamingResponse:
    _check_params(model, page, dpi, temperature, max_tokens)
    path = await _store_upload(file)
    try:
        prepared = prepare(path, page=page, dpi=dpi)
    except (ModelError, RenderError) as exc:
        path.unlink(missing_ok=True)
        raise _http_error(exc) from exc

    def events():
        last = ""
        try:
            for text in engine.stream(
                prepared.image,
                model_id=model,
                temperature=temperature,
                max_new_tokens=max_tokens,
            ):
                last = text
                payload = json.dumps({"text": text}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            final = finalize_document_for_response(
                prepared.image,
                last,
                model_id=model,
            )
            payload = json.dumps({"text": final}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            save_capture(
                image=prepared.image,
                text=final,
                source_name=file.filename or path.name,
                model=model,
                page=prepared.page,
                page_count=prepared.page_count,
                dpi=dpi,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            yield 'data: {"done": true}\n\n'
        except (ModelError, RenderError) as exc:
            payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        finally:
            path.unlink(missing_ok=True)

    return StreamingResponse(events(), media_type="text/event-stream")


demo = build_demo(engine)
demo.queue()
app = gr.mount_gradio_app(app, demo, path="/", show_api=True)
