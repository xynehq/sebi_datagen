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

Indic Vachak - SEBI reads one page from a PNG, JPEG, or PDF. It is the page app, not the line-crop training pipeline in the repository root.

The screen is dark by default. A switch under the Juspay mark turns on the light theme. The address stays `http://127.0.0.1:7860/` with no theme query.

## Run

From the repository root:

```bash
./ocr_service/run.sh
```

The first run creates `ocr_service/.venv` and installs the packages. Later runs skip that and open the browser.

Packages for the pipeline and this app are both in the repository root `requirements.txt`. There is no second requirements file. The process uses an NVIDIA GPU, then an Apple GPU, then CPU. On a CUDA machine, install the CUDA build of PyTorch before that file.

- Page: `http://127.0.0.1:7860/`
- OpenAPI: `http://127.0.0.1:7860/docs`

Copy `.env.example` to `.env` at the repository root when you need a private Hugging Face token, an API key, a different model cache, or a different capture folder. That is the only environment example.

## What the page does

1. Choose a model. Built-in names come from `ocr_service/models.yaml`. Extra Hugging Face vision models are added from the **Add model** tab (see below) and then appear in the same **Model** list.
2. Upload a PNG, JPEG, or PDF, or open **Example documents** and click an image.
3. For a PDF, pick the page and the DPI. Images ignore DPI.
4. Set temperature, streaming, and the longest text the model may write.
5. Click **Extract text**. Use the **Rendered** tab for the formatted document and **Raw Markdown** for the canonical text (same as `/v1/ocr` `text`). With **Enable Streaming** on, the rendered view updates live; table/layout alignment runs when generation finishes. Chitrapathak is asked for clean Markdown: `#` titles, ordinary paragraphs, and HTML `<table>` markup with one `<tr>` per record. That is the text `/v1/ocr` returns. IndicOCR stays a plain transcription. Repeated lines and code fences are removed before the text is shown. If the model dumps a table column-by-column, the service rebuilds rows from Tesseract **coordinates** while keeping recognized text in the original reading (`OCR_TABLE_LAYOUT=1`, `TESSERACT_CMD`, `OCR_TESSERACT_LANG`). HTML tables stay HTML in the API payload; they are not converted to pipe tables. Well-formed tables, lists, and prose from the model are left unchanged.
6. While text is arriving, words per second show on the same line as **Dataset** and **Finetuning**.
7. When the reading finishes, **Export** appears under the extracted text and downloads that text as a `.txt` file.
8. **Clear** resets the page, including temperature and streaming.
9. **Use via API** at the bottom opens the sidebar. The named calls are `/process_input`, `/update_slider_and_preview`, `/load_example_image`, `/get_model_info_text`, and `/lambda`.

Dataset opens the shared Drive folder. Finetuning opens `https://huggingface.co/snxtyle`.

## Models

## Adding an OCR model

Open the **Add model** tab.

1. If the server sets **OCR_API_KEY** (or **OCR_MODEL_ADMIN_KEY**), enter it under **Admin key**. Otherwise anyone with access to the UI can add models.
2. Paste a public **https://huggingface.co/…** link. Use **Base model** for a full vision repo. Optionally paste a **LoRA adapter** in Advanced options. Display name is filled from the Hugging Face id; you can edit it. Text-only, speech, and Tesseract `.traineddata` repos are rejected.
3. Click **Check link**, then **Download & add**. **Status** shows validation, download, load errors, and when the model is ready. Failed jobs can be retried or removed.
4. On the **OCR** tab, pick the model in **Model** (the list is not a search box) and run **Extract text**.

**Full vision model** — `config.json` plus weights. Chitrapathak-class and generic image-to-text VLMs load with `AutoModelForImageTextToText`. TrOCR / `VisionEncoderDecoderModel` repos load as the **trocr** family (ViT image processor + tokenizer; `sentencepiece` is required).

**LoRA adapter** — `adapter_config.json`. The base vision model is taken from the adapter config when possible; set **Base model** only when that is missing.

Added models are stored in `data/ocr_model_registry/registry.sqlite` (override with **OCR_MODEL_REGISTRY_DB**). Built-in models stay in `ocr_service/models.yaml`. Weight files still go to `ocr_service/model_cache/` (or **MODEL_CACHE_DIR**). Removing a model from the list does not delete shared cache files.

Supported families: **chitrapathak**, **indicocr**, **vlm**, and **trocr**. Not every public Hugging Face repo will load.

API (same admin Bearer token as above): `POST /v1/models/validate`, `POST /v1/models/register`, `GET /v1/models/jobs/{id}`, `GET /v1/models/custom`, `DELETE /v1/models/custom/{id}`.

**Troubleshooting**

| Message | What to do |
|---|---|
| Adding models is turned off | Set `OCR_API_KEY` or `OCR_MODEL_ADMIN_KEY` on the server and use it as the admin key. |
| Repository unavailable | Check the link, repo visibility, and optional `HF_TOKEN` for private repos. |
| Unsupported architecture | Use a vision image-to-text repo (VLM, Chitrapathak, IndicOCR, or TrOCR). Tesseract traineddata is not an engine. |
| need sentencepiece | Install `sentencepiece` (already in root `requirements.txt`) and retry the job. |
| Enter the base model | Paste the base Hugging Face id for the LoRA. |
| Failed download | Read the status reason; fix the repo or disk space, then retry or remove and add again. |

Current built-in ids include `chitrapathak-2`, `chitrapathak-marguj`, `chitrapathak-marguj-mixed`, `indicocr`, and `indicocr-marguj`.

The first Extract or `/v1/ocr` call for a missing model downloads it into `ocr_service/model_cache/`. Local Desktop paths are refused. Set `HF_TOKEN` in the root `.env` only for a private repo.

## Saved readings

A finished reading is stored under `data/ocr_captures/`. That folder is gitignored.

| What | Where |
|---|---|
| Page image that was read | `data/ocr_captures/pages/<id>.png` |
| Raw text | `data/ocr_captures/text/<id>.txt` |
| One ledger row | `data/ocr_captures/captures.jsonl` |

Each row records the source file name, model, page, DPI, temperature, max tokens, image size, and both file paths. `source` is `model_capture` and `gold` is false. These rows are not training labels. A person still has to accept or correct the text before it can enter `data/datasets/`.

Set `OCR_CAPTURE_DIR` in the root `.env` to store them somewhere else.

## API

If `OCR_API_KEY` is set, send `Authorization: Bearer <key>` on every `/v1` call. The page itself does not use that key.

```bash
curl -s http://127.0.0.1:7860/v1/health
curl -s http://127.0.0.1:7860/v1/models

curl -s -F "file=@ocr_service/examples/english_notice.png" \
  -F "model=chitrapathak-2" \
  -F "page=1" \
  -F "dpi=200" \
  -F "temperature=0" \
  -F "max_tokens=2048" \
  http://127.0.0.1:7860/v1/ocr
```

`POST /v1/ocr/stream` uses the same form fields. Add `-N`. Each event is the full text so far: `{"text": "..."}`. The last event is `{"done": true}`.

For `chitrapathak-2`, `chitrapathak-marguj`, and `chitrapathak-marguj-mixed`, `text` is Markdown with HTML tables (`<table>`, `<thead>`, `<tbody>`). A title is a `#` or `##` line. Each table record stays on one `<tr>`. A clause keeps its number, as in `१३. तुम्हांस आवश्यकता…`. The image is sent before that instruction, with the system line `You are a helpful assistant.` IndicOCR ids still return plain transcription, including LaTeX for math. Custom VLM and TrOCR models use the same `/v1/ocr` fields.

Anyone can open the site or call the API at that same address. No API key is required unless `OCR_API_KEY` is set.

```bash
curl -s -F "file=@ocr_service/examples/english_notice.png" \
  -F "model=chitrapathak-2" \
  -F "page=1" \
  -F "dpi=200" \
  -F "temperature=0" \
  -F "max_tokens=2048" \
  https://vivacious-cure-pouncing.ngrok-free.dev/v1/ocr
```

`model` is `chitrapathak-2`, `chitrapathak-marguj`, `chitrapathak-marguj-mixed`, `indicocr`, or `indicocr-marguj`. `page` starts at 1. `dpi` is 72–400 and applies to PDFs. `temperature` is 0–1. `max_tokens` is 1–8192.

PDF pages render at `dpi / 72`, then the long side is capped at 2048 px (`OCR_MAX_SIDE`). Uploads are limited to 30 MB (`OCR_MAX_UPLOAD_BYTES`). The upload is deleted when the request finishes. The capture files stay.

## Share this machine with ngrok

Sign in at [dashboard.ngrok.com](https://dashboard.ngrok.com/signup) as `suraj.nagre@juspay.in`. Copy the authtoken from [your authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).

The public site is one address: https://vivacious-cure-pouncing.ngrok-free.dev

The authtoken stays in the root `.env` as `NGROK_AUTHTOKEN` and in `ocr_service/ngrok.yml`. Git ignores both. With the app already running:

```bash
ngrok start indic-vachak --config ocr_service/ngrok.yml
```

## Hugging Face Space

A Space like the LightOn demo needs a running Gradio or Docker host. The `snxtyle` account cannot create one until [Hugging Face Pro](https://huggingface.co/pro) is enabled. The Docker file in this folder is ready for that. These models also need a GPU in the Space hardware settings.
