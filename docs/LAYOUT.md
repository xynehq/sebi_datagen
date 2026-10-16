# Layout and compatibility

## Source

| Path | Role |
| --- | --- |
| `src/indic_ocr_pipeline/` | Dataset, review, OCR adapter, and device code |
| `pipeline/` | Orchestrator, model discovery, and CLI |
| `evaluation/` | Metrics and checkpoint inference |
| `training/` | Split guards and checkpoint selection |
| `models/IndicPhotoOCR/` | Vendored PARSeq engine. Import it as `IndicPhotoOCR` |
| `models/chitrapathak/`, `models/tesseract/` | Backend adapters. Weights are not stored here |
| `configs/languages/` | Marathi, Gujarati, and Bengali profiles |
| `configs/*.yaml` | Shared defaults |
| `scripts/` | Thin stage wrappers. They import the package |
| `ocr_service/` | Indic Vachak page reader. Separate from the line pipeline |

## Configuration precedence

1. CLI flags such as `--language`, `--model-path`, and `--checkpoint`.
2. Environment variables. `INDIC_OCR_DEVICE` overrides `MARATHI_OCR_DEVICE` when both are set.
3. `configs/local.yaml` when present. That file is gitignored.
4. `configs/run.yaml`, then `configs/pipeline.yaml`.
5. `configs/languages/<language>.yaml` for script, Tesseract language, and language output directories.

## Data that was not moved

The Marathi corpus already has documents, crops, predictions, and review files. Those paths stay where the running pipeline reads them:

- `data/documents/`
- `data/pages/`
- `data/crops/`
- `data/output/`
- `data/ocr_captures/` for page-reader images and raw text. Not gold
- `data/review/`
- `runs/` for Marathi resume state

Moving them would break resume. Gujarati and Bengali runs write under `runs/<language>/` and `reports/<language>/`. Bengali dataset output is `data/bengali/real10k/` when a real source exists.

New run directories record logs under `runs/<run-id>/logs/` and reports under `runs/<run-id>/reports/`. Model caches stay in `models/cache/` and the Hugging Face cache, separate from language datasets.
