# Indic OCR pipeline

This repository builds OCR training sets for Indian languages from PDFs or from image/text pairs, reviews them with a person, fine-tunes a recognizer, and scores that recognizer. It is a dataset, training, and evaluation pipeline. It is not an OCR web service.

## Languages and backends

| Language | Dataset workflow | Notes |
| --- | --- | --- |
| Marathi | Verified on the existing PDF corpus and fixture pipeline | Existing crops and manifests stay in their current directories so a run can resume |
| Gujarati | Configuration and external adapter resolution | A PEFT adapter is checked against its base model. This repository does not train Chitrapathak LoRA |
| Bengali | Dataset preparation command | Writes `data/bengali/real10k` only when a real labeled source is present |

OCR backends are IndicOCR, Chitrapathak, and Tesseract. Training and checkpoint evaluation use PARSeq when the checkpoint is a `.ckpt` file. A model is not described as supporting a language unless its own metadata says so. Unknown support stays unknown.

## Installation

```bash
./setup.sh
```

`./setup.sh` creates or reuses `.venv`, installs this project with `pip install -e .`, and prints the accelerator it detected: CUDA, then Apple MPS, then CPU. It does not download model weights. A later `./run_pipeline.sh` reuses that environment and does not reinstall dependencies when the required packages already import.

## Commands

```bash
./run_pipeline.sh --list-models
./run_pipeline.sh --language marathi --stage dataset
./run_pipeline.sh --language gujarati --stage eval --model-path /path/to/guj_v1
./run_pipeline.sh --language bengali --stage dataset
```

The installed command is `indic-ocr`, which is what the shell wrapper runs. Leave the device unset. Set `INDIC_OCR_DEVICE` to `cuda`, `cuda:0`, `mps`, or `cpu` only to force one. `MARATHI_OCR_DEVICE` is still accepted.

## Models

`./run_pipeline.sh --list-models` searches `models/adapters`, `models/cache`, `models/registry.json`, `--model-search-root`, and the Hugging Face cache. It does not scan the whole disk.

```bash
./run_pipeline.sh --model-path /path/to/model --stage eval
./run_pipeline.sh --model-hf organization/model-name --stage eval
./run_pipeline.sh --model-url https://example.com/model.ckpt --stage eval
```

An explicit path is never replaced with a different model. A missing explicit model is an error. Weights stay outside git.

## Data workflows

**PDFs.** Put PDFs in `data/raw/documents` or `input/pdfs`, then run the Marathi dataset stage. The pipeline renders pages, finds lines, runs OCR, compares the backends, and exports a training split only from human-approved rows.

**Already labeled images.** Bengali preparation reads `data/bengali/source/labels.jsonl` and `data/bengali/source/images/`. Each row needs `image_filename` and `expected_text`. The command does not invent rows.

**Gold.** `gold=true` only after a person approves or corrects a line. Model agreement is not gold. Auto-accepted rows stay out of the human-only training set unless `--include-auto-accepted` is passed, and they remain `gold=false`. The frozen exam is not used for training, calibration, or checkpoint selection.

## Evaluation

Checkpoint selection uses the development validation split. The frozen exam is scored once after that choice. Metrics include strict character error rate, word error rate, exact match, and the existing whitespace diagnostics. PARSeq geometry is not applied to a Chitrapathak adapter.

## Where files live

| Kind | Location |
| --- | --- |
| Language settings | `configs/languages/` |
| Shared defaults | `configs/*.yaml` |
| Local overrides, not committed | `configs/local.yaml`, `.env` |
| Marathi production data already on disk | `data/`, `manifests/`, `runs/` |
| Other languages | `data/<language>/`, `runs/<language>/`, `reports/<language>/` |
| Shared model cache | `models/cache/`, the Hugging Face cache |
| Vendored PARSeq engine | `models/IndicPhotoOCR/` (import name stays `IndicPhotoOCR`) |

Precedence is: CLI flags, then environment variables, then `configs/local.yaml` when the loader merges it, then `configs/run.yaml` and `configs/pipeline.yaml`, then the language file. See [docs/LAYOUT.md](docs/LAYOUT.md).

## Documentation

- [Getting started and operator detail](docs/GUIDE.md)
- [Layout and compatibility](docs/LAYOUT.md)
- [Model management](docs/MODEL_MANAGEMENT.md)
- [Label policy](docs/LABEL_POLICY.md)
- [Review policy](docs/REVIEW_POLICY.md)
- [Evaluation pipeline](docs/PIPELINE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Vendored engine](models/THIRD_PARTY.md)

## Known limitations

- Chitrapathak LoRA training is not implemented. A PEFT adapter is not sent to the PARSeq trainer.
- Bengali `real10k` is not created until a real labeled source is supplied.
- Gujarati adapter evaluation records the adapter and its base model. It does not by itself compute character error rate.
- The default `./run_pipeline.sh` interpreter needs `pytorch_lightning` before a real PARSeq training job can start. `./setup.sh` installs the declared requirements into `.venv`.
- Marathi artifacts stay on the existing paths so the current corpus can resume. They were not migrated into `runs/marathi/`.
