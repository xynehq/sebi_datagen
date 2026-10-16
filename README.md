# Indic OCR pipeline

This repository turns scanned PDFs into an OCR training set, lets a person check each line, fine-tunes a recognizer, and scores that recognizer.

You run the whole pipeline with `./setup.sh`. It writes files on disk. It does not serve an OCR API.

```text
PDFs
  → page images
  → line crops
  → three OCR readings per line
  → a review page you accept or reject
  → train / validation / test files
  → one epoch of PARSeq fine-tuning
  → scores on the validation set
```

Three OCR models read each line: IndicOCR, Chitrapathak, and Tesseract. Their agreement is a suggestion. A line becomes training gold only after a person accepts or corrects it. Training starts from the published PARSeq base for that language and fine-tunes it for one epoch.

## Page reader

Indic Vachak - SEBI is a separate app in `ocr_service/`. It reads a whole PNG, JPEG, or PDF page. It does not run this pipeline, and a model reading is not training gold.

```bash
./ocr_service/run.sh
```

One public address, while this machine is running the app and ngrok:

https://vivacious-cure-pouncing.ngrok-free.dev

On this machine the same app is `http://127.0.0.1:7860/`. Anyone can open that public page or call `POST /v1/ocr` on it. No API key is required unless `OCR_API_KEY` is set in `.env`.

Finished readings are written to `data/ocr_captures/` (`pages/`, `text/`, and `captures.jsonl`). Each row is `source: model_capture` and `gold: false`. A person still has to accept or correct the text before it can enter `data/datasets/`.

There is one `requirements.txt`, one `.env.example`, and one `.gitignore`, all at the repository root. Copy `.env.example` to `.env`. The ngrok token stays in `.env` and in `ocr_service/ngrok.yml`. Both are gitignored.

The screen, models, API, and tunnel steps are in [ocr_service/README.md](ocr_service/README.md).

## What you need

- Python 3.11 or newer
- IndicOCR and Chitrapathak weights you already have, for the OCR stage
- Tesseract on your PATH, with the language you are reading. Marathi uses `mar+eng`

The train stage downloads the published language base if it is not already cached. IndicOCR and Chitrapathak are not downloaded. Put those two on disk and point `.env` at them.

## Set up

From the repository root:

```bash
chmod +x setup.sh run_pipeline.sh
./setup.sh
```

`./setup.sh` creates `.venv` and installs this project only when those packages are missing. When they are already installed, it goes straight to the pipeline. It deletes generated outputs from the last run, then runs the full Marathi pipeline from the PDFs. Source PDFs in `data/documents/` stay. Running it again starts that pipeline over. It prints the device it found: CUDA, then Apple MPS, then CPU. The review page opens in the browser, and the command waits there until you confirm.

Copy the environment file and fill in the paths you have:

```bash
cp .env.example .env
```

```bash
INDICOCR_MODEL_PATH=/absolute/path/to/indicocr
CHITRAPATHAK_MODEL_PATH=/absolute/path/to/chitrapathak
TESSERACT_CMD=tesseract
```

Leave `CHECKPOINT` unset. `./setup.sh` always trains from the published language base. Pass `--checkpoint /path/to/file.ckpt` only when one run should start from a different file.

`./setup.sh` and `./run_pipeline.sh` load `.env` when that file exists. Leave `INDIC_OCR_DEVICE` empty to auto-detect. Set it to `cuda`, `mps`, or `cpu` only when you want to force one.

Chitrapathak also needs the Transformers stack. If `./setup.sh` did not already provide it:

```bash
.venv/bin/python -m pip install -e ".[chitrapathak]"
```

Check the machine before a long run:

```bash
./run_pipeline.sh --check
./run_pipeline.sh --list-stages
./run_pipeline.sh --list-models
```

`--list-models` looks in `models/adapters`, `models/cache`, `models/registry.json`, and the Hugging Face cache. It does not scan the whole disk and it does not download anything.

A small synthetic run needs no weights:

```bash
./run_pipeline.sh --fixture --offline
```

## Put the PDFs in place

Marathi PDFs go in `data/documents`, or pass another folder:

```bash
./run_pipeline.sh --language marathi --input-dir /path/to/pdfs
```

Use more than one PDF. Splits are by document. With a single document, that document is kept for validation and the training split is empty.

## Run the pipeline

```bash
./setup.sh
```

That one command installs the environment when needed, clears generated outputs, and runs every stage on every PDF in `data/documents`. Training starts from the published Marathi base. The review page opens when the lines are ready, and the command continues after you confirm.

The default dataset size is all lines. Pass a number only when you want a smaller run:

```bash
./setup.sh --dataset-size 10
```

Chitrapathak may return up to 512 tokens for each line crop by default. Raise the limit with `--chitrapathak-max-tokens` (the official full-page reader uses 3000), or set `CHITRAPATHAK_MAX_NEW_TOKENS` in `.env`:

```bash
./setup.sh --chitrapathak-max-tokens 3000
```

`./run_pipeline.sh` runs the same stages without clearing previous outputs. Completed stages are skipped when their inputs have not changed. A failed stage leaves the earlier files in place.

### 1. Build lines up to review

```bash
./run_pipeline.sh --language marathi
```

This ingests the PDFs, renders pages, finds lines, runs the three OCR backends, and writes a suggested reading for each line. It stops before dataset export when no human-approved lines exist yet, and prints the review page path.

`--offline` keeps the run from contacting Hugging Face or Weights & Biases.

### 2. Review the lines

The pipeline opens the review page in your browser and waits.

Every line starts as **not reviewed**. Use the buttons on the page:

| Button | What it does |
| --- | --- |
| Accept | Marks that line as gold, using the text in the box. If you edited the text, the edit is stored as the correction. |
| Reject | Marks that line out of training. |
| Accept all | Accepts every line that has text. |
| Reject all | Rejects every line. |

When every line has a choice, a confirmation asks whether to continue.

- **Yes** closes the page. The same run exports the dataset, then trains and evaluates.
- **No** leaves the page open so you can change lines and confirm later.

Opening `reports/review/index.html` in the editor shows the source. Use the page the pipeline opened.

You can also open it yourself:

```bash
./run_pipeline.sh --serve-review --reviewer operator --offline
```

### 3. Record the review without the page

```bash
./run_pipeline.sh --accept-proposed --reviewer operator --offline
```

`--reviewer` must be a real person id. `agent`, `auto`, `assistant`, and `cursor` are refused.

This command writes the decisions into `data/output/verified.jsonl`, then continues:

1. **Export** builds `data/datasets/train.jsonl`, `val.jsonl`, and `test.jsonl` from accepted and corrected lines. Rejected lines and lines still marked not reviewed stay out.
2. **Train** fine-tunes PARSeq for 1 epoch (`configs/training.yaml`). While training is running, validation runs at 30% and 70% of the steps. The checkpoint with the better score is kept.
3. **Evaluate** scores that checkpoint on the validation set: character error rate, word error rate, and exact match.
4. **Select** writes the chosen checkpoint to the run directory.
5. **Frozen exam** scores `data/eval_data` once, after selection, when that directory is present. It is never used to train or to pick the checkpoint. When the exam is not on this machine, the stage is skipped and the validation scores stand.
6. **Report** writes `reports/final_report.html`.

To record the review and stop before export:

```bash
./run_pipeline.sh --accept-proposed --no-continue --reviewer operator
```

A line that is already recorded cannot be changed by running the command again. The same choice is skipped. A different choice stops the command and keeps the old row.

## What each stage does

`./run_pipeline.sh --list-stages` prints the same names.

| Stage | Result |
| --- | --- |
| preflight | Checks Python, configs, and configured model paths |
| ingest | Registers the PDFs |
| render | Writes a PNG for each page |
| detect | Finds text regions |
| crop | Cuts one image per line |
| ocr | Writes IndicOCR, Chitrapathak, and Tesseract text |
| compare | Picks a suggested line from those three readings |
| calibrate | Builds the review sample |
| export | Writes train, validation, and test files from human gold |
| train | Fine-tunes PARSeq |
| eval_checkpoints | Scores checkpoints on the validation set |
| select | Picks the best validation checkpoint |
| frozen | Scores the held-out exam in `data/eval_data` |
| report | Writes the final HTML report |

## Run one part

```bash
./run_pipeline.sh --language marathi --stage dataset
./run_pipeline.sh --stage train --offline
./run_pipeline.sh --stage eval --offline
./run_pipeline.sh --from ocr --offline
./run_pipeline.sh --stages ingest,render,ocr --offline
./run_pipeline.sh --dry-run
```

`--stage dataset` runs through export and still waits for review when no gold exists. `--from ocr` starts at OCR and continues to the end. `--dry-run` prints the plan and does not run it.

Start one training run from a file other than the published base:

```bash
./run_pipeline.sh --from train --checkpoint /absolute/path/to/recognizer.ckpt --offline
```

An explicit `--model-path` is used as given. A missing explicit path is an error.

## Languages

| Language | How to run it |
| --- | --- |
| Marathi | `./run_pipeline.sh --language marathi`. PDFs in `data/documents`. Tesseract language `mar+eng`. Existing data stays in `data/` and `runs/`. |
| Gujarati | `./run_pipeline.sh --language gujarati --stage eval --model-path /path/to/adapter`. This checks a PEFT adapter against its base model. This repository does not train a Chitrapathak LoRA. |
| Bengali | `./run_pipeline.sh --language bengali --stage dataset`. Reads `data/bengali/source/labels.jsonl` and `data/bengali/source/images/`. Each row needs `image_filename` and `expected_text`. `data/bengali/real10k` is written only when that source has enough labeled images. |

Language files are in `configs/languages/`. A model is described as supporting a language only when its own metadata says so.

## Where files go

| What | Where |
| --- | --- |
| Source PDFs | `data/documents` |
| Page images and line crops | `data/pages`, `data/crops` |
| Stage records | `data/output/*.jsonl` |
| Human decisions | `data/output/verified.jsonl` |
| Review page | `reports/review/index.html` and `reports/review/decisions.json` |
| Train / validation / test | `data/datasets/` |
| Page-reader captures | `data/ocr_captures/`. Model output, not gold. |
| One run | `runs/<date>_<n>/` |
| Shared settings | `configs/*.yaml` |
| Your machine only | `configs/local.yaml` (copy `configs/local.yaml.example`), `.env` |
| PARSeq code | `models/IndicPhotoOCR/` |

CLI flags override environment variables. Those override `configs/local.yaml`, then `configs/run.yaml`, `configs/pipeline.yaml`, and the language file.

## Training defaults

From `configs/training.yaml` and `configs/pipeline.yaml`:

- Dataset labels are at most 10 characters, counting digits. At least 95% are shorter than 5. Any longer label is 8 to 10 characters (`configs/dataset.yaml`).
- Starts from the published language base when `base_checkpoint` and `CHECKPOINT` are empty
- 1 epoch, batch size 8, learning rate `0.0001`
- Validation inside training at 30% and 70% of the optimizer steps
- One more score after training, on `data/datasets/val.jsonl`
- Leave `training.steps` empty to use epochs times the number of batches

## If something fails

| What you see | What to do |
| --- | --- |
| Review buttons do not save | Use the browser page the pipeline opened, not the HTML file in the editor. |
| The confirmation does not continue | Choose Yes only after every line is accepted or rejected. No keeps the page open. |
| Export or training does not start | Confirm on the review page, or run `./run_pipeline.sh --accept-proposed --reviewer operator`. |
| `Training split is empty` | Add a second PDF. One document is reserved for validation. |
| A model path is missing | Fix the path in `.env`. Setup does not download it. |
| Frozen exam was skipped | Place the exam under `data/eval_data`, then run `./run_pipeline.sh --from frozen`. |
| A recorded line cannot be changed | The first recorded accept or reject for that crop is kept. |

Tests:

```bash
.venv/bin/python -m pytest
```

## More detail

- [Operator guide](docs/GUIDE.md)
- [Layout](docs/LAYOUT.md)
- [Label policy](docs/LABEL_POLICY.md)
- [Review policy](docs/REVIEW_POLICY.md)
- [Model management](docs/MODEL_MANAGEMENT.md)
- [Evaluation](docs/PIPELINE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Vendored PARSeq engine](models/THIRD_PARTY.md)
