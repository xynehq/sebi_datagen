---
title: Indic Vachak - SEBI
emoji: 📄
colorFrom: blue
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
---

# Indic Vachak - SEBI

Page OCR for a PNG, JPEG, or PDF. This is the page app in `ocr_service/`. It is not the line-crop training pipeline in the repository root.

One process serves **both**:

| How | Who | URL |
|---|---|---|
| **UI** (Gradio) | Us — try models, pages, DPI, streaming | `http://<host>:7860/` |
| **API** (`/v1/...`) | Everyone — apps, scripts, xyne / integrations | `http://<host>:7860/v1/ocr` |

Same models, same finalize path. UI **Raw Markdown** matches API `text`.

Dark theme by default. Theme switch under the Juspay mark. Local URL stays `http://127.0.0.1:7860/` with no theme query.

## Deploy (simple)

On a CUDA/GPU machine (or any machine with Python 3):

```bash
git clone https://github.com/xynehq/sebi_datagen.git
cd sebi_datagen
./ocr_service/run.sh
```

That one command:

1. Creates `ocr_service/.venv` if needed  
2. Installs the root `requirements.txt` when packages are missing  
3. Starts the server on `0.0.0.0:7860`  
4. Opens the browser when it can  

| | |
|---|---|
| **UI** (test in browser) | `http://<server-ip>:7860/` |
| **API** health | `http://<server-ip>:7860/v1/health` |
| **API** models | `http://<server-ip>:7860/v1/models` |
| **API** OCR | `POST http://<server-ip>:7860/v1/ocr` |
| OpenAPI docs | `http://<server-ip>:7860/docs` |

Other port (for example while testing):

```bash
OCR_PORT=7861 ./ocr_service/run.sh
```

Optional `.env` at the **repository root** (copy from `.env.example`):

- `HF_TOKEN` — private Hugging Face repos only  
- `OCR_API_KEY` — require Bearer token on `/v1`  
- `MODEL_CACHE_DIR` — where weights download (default `ocr_service/model_cache/`)  

Built-in adapters are already in `models.yaml`. You do **not** set adapter paths by hand for normal use. First Extract downloads missing weights from Hugging Face.

On CUDA hosts, install a CUDA build of PyTorch if pip gave you a CPU-only wheel.

Do **not** use a separate manual `venv` + `pip` + `uvicorn` flow unless you are debugging. Prefer `./ocr_service/run.sh`.

## UI (for our testing)

Open `http://<host>:7860/` after `./ocr_service/run.sh`. Use this to try models and pages yourself before wiring the API.

1. Pick a **Model** (built-in list, or one from **Add model**).  
2. Upload a PNG, JPEG, or PDF, or open **Example documents**.  
3. For a PDF: page number and DPI (images ignore DPI).  
4. Temperature, streaming, max tokens.  
5. **Extract text**.  
6. **Rendered** = formatted view while you watch. **Raw Markdown** = canonical text (same string as API `text`).  
7. **Export** downloads that raw text when a reading finishes.  
8. **Clear** resets the form.  
9. **Use via API** at the bottom opens Gradio’s API sidebar for the same server.

### Live view vs final layout

| When | What you see |
|---|---|
| While streaming | Text and tables update live. Partial HTML tables are closed for display. Tab-separated or pipe tables are shown as HTML as they arrive. |
| When generation finishes | Geometry / Tesseract row alignment runs once. That final HTML table is what Export and `/v1/ocr` keep. |

Streaming does not wait for Tesseract. Full row alignment is the end pass.

Chitrapathak models are asked for Markdown with HTML `<table>` (one `<tr>` per record). IndicOCR stays a plain transcription. Column-by-column dumps can be repaired at the end (`OCR_TABLE_LAYOUT=1`).

Words/s show next to Dataset / Finetuning while text arrives.

## Models (built-in)

| Id | Weights | Use |
|---|---|---|
| `chitrapathak-2` | `krutrim-ai-labs/Chitrapathak-2` | Default. Marathi / Devanagari pages. |
| `chitrapathak-marguj` | base + [`snxtyle/chitrapathak-MarGuj`](https://huggingface.co/snxtyle/chitrapathak-MarGuj) | Stacked Marathi→Gujarati LoRA. Prefer for Gujarati. |
| `chitrapathak-marguj-mixed` | base + [`snxtyle/chitrapathak_marguj_mixed`](https://huggingface.co/snxtyle/chitrapathak_marguj_mixed) | One shared LoRA for Marathi and Gujarati. |
| `indicocr` | [`bodhan-ai/indic-ocr`](https://huggingface.co/bodhan-ai/indic-ocr) `weights/ocr` | Published IndicOCR page reader. |
| `indicocr-marguj` | that base + [`snxtyle/IndicOCR-MarGuj`](https://huggingface.co/snxtyle/IndicOCR-MarGuj) | IndicOCR LoRA for Gujarati. |

List source: `ocr_service/models.yaml`. First use downloads into `ocr_service/model_cache/`. Local Desktop paths are refused.

## Add a model (no code)

Open the **Add model** tab on the page.

1. Admin key only if the server set `OCR_API_KEY` / `OCR_MODEL_ADMIN_KEY`.  
2. Paste a Hugging Face **https://huggingface.co/…** link (full vision model, or LoRA in Advanced).  
3. **Check link** → **Download & add**.  
4. On the **OCR** tab, select it in **Model** and Extract.

Supported families: **chitrapathak**, **indicocr**, **vlm**, **trocr**. Text-only / speech / Tesseract `.traineddata` are rejected.

Page-added models: `data/ocr_model_registry/` (gitignored). Built-ins stay in `models.yaml`.

API: `POST /v1/models/validate`, `POST /v1/models/register`, `GET /v1/models/jobs/{id}`, `GET /v1/models/custom`, `DELETE /v1/models/custom/{id}`.

| Message | What to do |
|---|---|
| Adding models is turned off | Set `OCR_API_KEY` or `OCR_MODEL_ADMIN_KEY` and use it as the admin key. |
| Repository unavailable | Check link, visibility, optional `HF_TOKEN`. |
| Unsupported architecture | Use a vision image-to-text repo. |
| need sentencepiece | Install `sentencepiece` (in root `requirements.txt`) and retry. |
| Enter the base model | Paste the base HF id for the LoRA. |
| Failed download | Fix repo or disk, then retry or remove and add again. |

## Saved readings

Under `data/ocr_captures/` (gitignored):

| What | Where |
|---|---|
| Page image | `pages/<id>.png` |
| Raw text | `text/<id>.txt` |
| Ledger row | `captures.jsonl` |

`source` is `model_capture`, `gold` is false. Not training labels until a person reviews them. Override folder with `OCR_CAPTURE_DIR`.

## API (for everyone / integrations)

Same server as the UI. Call `/v1` from scripts, backends, or xyne. You do not need the Gradio page for production callers.

If `OCR_API_KEY` is set, every `/v1` request needs `Authorization: Bearer <key>`. The UI Extract button does not send that key.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/health` | Liveness |
| `GET` | `/v1/models` | Built-in + page-added models |
| `POST` | `/v1/ocr` | One page → JSON with `text` |
| `POST` | `/v1/ocr/stream` | SSE; cumulative `{"text":"..."}`, then `{"done":true}` |

Also: model register/validate under `/v1/models/...` (see **Add a model**). Full schema: `http://<host>:7860/docs`.

### Form fields (`/v1/ocr` and `/v1/ocr/stream`)

| Field | Notes |
|---|---|
| `file` | PNG, JPEG, or PDF |
| `model` | e.g. `chitrapathak-marguj-mixed` |
| `page` | ≥ 1 (PDFs) |
| `dpi` | 72–400, PDFs only |
| `temperature` | 0–1 |
| `max_tokens` | 1–8192 |

### Examples

```bash
curl -s http://127.0.0.1:7860/v1/health
curl -s http://127.0.0.1:7860/v1/models

curl -s -F "file=@ocr_service/examples/english_notice.png" \
  -F "model=chitrapathak-marguj-mixed" \
  -F "page=1" \
  -F "dpi=200" \
  -F "temperature=0" \
  -F "max_tokens=2048" \
  http://127.0.0.1:7860/v1/ocr
```

Stream (add `-N`):

```bash
curl -N -s -F "file=@ocr_service/examples/english_notice.png" \
  -F "model=chitrapathak-marguj-mixed" \
  -F "page=1" \
  -F "dpi=200" \
  -F "temperature=0" \
  -F "max_tokens=2048" \
  http://127.0.0.1:7860/v1/ocr/stream
```

With API key:

```bash
curl -s -H "Authorization: Bearer $OCR_API_KEY" \
  -F "file=@ocr_service/examples/english_notice.png" \
  -F "model=chitrapathak-2" \
  http://127.0.0.1:7860/v1/ocr
```

Response `text` for Chitrapathak is Markdown with HTML tables. IndicOCR is plain transcription. Long side capped at 2048 px. Upload limit 30 MB. The upload file is deleted after the request; captures under `data/ocr_captures/` stay.

## Share with ngrok

1. Put `NGROK_AUTHTOKEN` in the root `.env` (gitignored).  
2. Start the app with `./ocr_service/run.sh` (port 7860).  
3. Then:

```bash
ngrok start indic-vachak --config ocr_service/ngrok.yml
```

`ocr_service/ngrok.yml` is gitignored. Do not commit tokens.

## Hugging Face Space

Needs Gradio/Docker host and GPU. `snxtyle` needs [HF Pro](https://huggingface.co/pro) to create a Space. Dockerfile in this folder is ready for that.
