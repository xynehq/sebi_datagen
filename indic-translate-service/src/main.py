"""HTTP API for IndicTrans2 Indic → English translation, as lighton-ocr-server's translate_client calls it.

POST /translate  {"texts": [...], "src_lang": "mr", "fix_numbers": true}  ->  {"translations": [...]}
GET  /health

Settings (env): TRANSLATE_MODEL (default ai4bharat/indictrans2-indic-en-1B), TRANSLATE_DEVICE
(default cuda if available, else cpu). For air-gapped hosts set HF_HUB_OFFLINE=1 with the model in
the HuggingFace cache (the Docker image bakes it in).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from translator import FLORES_CODE, Translator

MODEL_NAME = os.environ.get("TRANSLATE_MODEL", "ai4bharat/indictrans2-indic-en-1B")
MAX_TEXTS_PER_REQUEST = int(os.environ.get("TRANSLATE_MAX_TEXTS", "64"))

STATE: dict[str, Any] = {}


def _device() -> str:
    wanted = os.environ.get("TRANSLATE_DEVICE", "")
    if wanted:
        return wanted
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE["translator"] = Translator(MODEL_NAME, _device())
    yield
    STATE.clear()


app = FastAPI(title="indic-translate-service", lifespan=lifespan)


class TranslateRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    src_lang: str
    fix_numbers: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    translator: Translator | None = STATE.get("translator")
    if translator is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok", "languages": sorted(FLORES_CODE), **translator.health()}


@app.post("/translate")
async def translate(req: TranslateRequest) -> dict[str, list[str]]:
    if req.src_lang not in FLORES_CODE:
        raise HTTPException(400, f"unsupported src_lang {req.src_lang!r}; use one of {sorted(FLORES_CODE)}")
    if len(req.texts) > MAX_TEXTS_PER_REQUEST:
        raise HTTPException(413, f"at most {MAX_TEXTS_PER_REQUEST} texts per request")
    translator: Translator = STATE["translator"]
    translations = await asyncio.to_thread(
        translator.translate, req.texts, req.src_lang, fix_numbers_on=req.fix_numbers
    )
    return {"translations": translations}
