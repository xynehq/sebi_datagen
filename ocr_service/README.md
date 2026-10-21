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

1. Choose a model. All four stay in the list even when the weights are not on the machine yet.
2. Upload a PNG, JPEG, or PDF, or open **Example documents** and click an image.
3. For a PDF, pick the page and the DPI. Images ignore DPI.
4. Set temperature, streaming, and the longest text the model may write.
5. Click **Extract text**. The rendered reading stays above. **Raw Markdown Output** stays below. Chitrapathak is asked for clean markdown: `#` titles, a markdown table with one row per item, and the clause number on the same line as the clause. That is the text `/v1/ocr` returns, so a fine-tune trained on the same shape can replace the model without changing the caller. IndicOCR stays a plain transcription. Repeated lines and code fences are removed before the text is shown.
6. While text is arriving, words per second show on the same line as **Dataset** and **Finetuning**.
7. When the reading finishes, **Export** appears under the extracted text and downloads that text as a `.txt` file.
8. **Clear** resets the page, including temperature and streaming.
9. **Use via API** at the bottom opens the sidebar. The named calls are `/process_input`, `/update_slider_and_preview`, `/load_example_image`, `/get_model_info_text`, and `/lambda`.

Dataset opens the shared Drive folder. Finetuning opens `https://huggingface.co/snxtyle`.

## Models

| Id | Weights | Use |
|---|---|---|
| `chitrapathak-2` | `krutrim-ai-labs/Chitrapathak-2` | Default. Marathi pages stay in Devanagari. |
| `chitrapathak-marguj` | base plus [`snxtyle/chitrapathak-MarGuj`](https://huggingface.co/snxtyle/chitrapathak-MarGuj) | Gujarati reader. It writes Gujarati. |
| `chitrapathak-marguj-mixed` | base plus [`snxtyle/chitrapathak_marguj_mixed`](https://huggingface.co/snxtyle/chitrapathak_marguj_mixed) | One shared LoRA for Marathi and Gujarati. |
| `indicocr` | [`bodhan-ai/indic-ocr`](https://huggingface.co/bodhan-ai/indic-ocr) `weights/ocr` | Published IndicOCR reader. |
| `indicocr-marguj` | that base plus [`snxtyle/IndicOCR-MarGuj`](https://huggingface.co/snxtyle/IndicOCR-MarGuj) | IndicOCR LoRA for Gujarati. It writes Gujarati. |

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

For `chitrapathak-2`, `chitrapathak-marguj`, and `chitrapathak-marguj-mixed`, `text` is markdown. A title is a `#` or `##` line. A table keeps each item on its own row. A clause keeps its number, as in `१३. तुम्हांस आवश्यकता…`. The image is sent before that instruction, with the system line `You are a helpful assistant.` IndicOCR ids still return plain transcription, including LaTeX for math.

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
