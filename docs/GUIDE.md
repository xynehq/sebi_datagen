# Detailed operator guide

This is the long-form guide. Start at the repository README for install and commands.

# Language-agnostic OCR dataset pipeline

This repository builds an OCR training set from PDFs, reviews it with a person, fine-tunes a recognizer, and scores that recognizer. It is a dataset, training, and evaluation pipeline. It is not an OCR web service: there is no HTTP API, no Docker image, and no vLLM server.

Language profiles live in `configs/languages/`. Marathi keeps the existing data and `runs/` layout. Gujarati and Bengali write under their own `data/`, `runs/`, and `reports/` directories. A model is not described as supporting a language unless its metadata says so.

```bash
./run_pipeline.sh --language marathi --stage dataset
./run_pipeline.sh --list-models
./run_pipeline.sh --language gujarati --stage eval --model-path /path/to/guj_v1
./run_pipeline.sh --language bengali --stage dataset
```

`--list-models` searches `models/adapters`, `models/cache`, `models/registry.json`, `--model-search-root`, and the Hugging Face cache. It does not scan the whole disk and it does not download weights. Chitrapathak LoRA training is not implemented here. `gujarati --stage eval --model-path` checks that the path is a PEFT adapter and that its base model exists locally; it does not run the PARSeq trainer and it does not by itself compute CER. Bengali dataset preparation reads `data/bengali/source/labels.jsonl` and `data/bengali/source/images/`. It does not invent rows when that source is missing.

One launcher runs the whole workflow and resumes it after a failure:

```bash
./setup.sh
./run_pipeline.sh
```

`./setup.sh` prepares the environment and does not download weights. `./run_pipeline.sh` is the only way to run stages. Completed stages are skipped when their outputs are still valid and their inputs have not changed.

## What you get

```text
PDFs
  → page images
  → regions
  → line crops
  → one transcription per model
  → one consensus suggestion per crop
  → a human audit sample
  → human gold, and auto-accepted rows only when a measured bound allows it
  → train / validation / held-out splits
  → PARSeq fine-tune with validation during training
  → checkpoint chosen on the development validation split
  → one score of the frozen exam
  → a report
```

For N line crops and M OCR models the pipeline stores N crops, N×M predictions, and N candidates. The M strings are votes on one line. They are not M training samples.

Agreement is a suggestion. It is not a label. Gold in `data/golden/` is human review only.

## What this repository refuses to do

- It does not turn 3-agree or 2-agree into ground truth.
- It does not train on empty gold, on rejected lines, on bad boundaries, or on rows whose reviewer is missing or blocked.
- It does not train on the frozen exam, tune a threshold on it, or pick a checkpoint from it.
- It does not split a human audit budget into equal thirds. Disagreement and majority lines take more of the budget than unanimous lines.
- It does not download model weights from `./setup.sh`.
- It does not replace IndicOCR or Tesseract with a generic transformers loader. Those stay adapters. An extra model can be a transformers image-to-text checkpoint.
- It does not delete source PDFs, rendered pages, crops, predictions, or old run directories when a later stage fails.

## Setup

Python 3.11 or newer. Tesseract is a system program, used only when that backend runs. PyTorch is required for IndicOCR, Chitrapathak, training, and real checkpoint scoring. The synthetic fixture does not need a GPU or a checkpoint file.

```bash
chmod +x setup.sh run_pipeline.sh
./setup.sh
```

`./setup.sh` is safe to run again. It will:

1. Refuse a Python older than 3.11.
2. Create `.venv` when it is missing, or reuse it.
3. Install the pinned packages in `requirements.txt`.
4. Print the Python version, the PyTorch version, and the CUDA device when CUDA is available. CPU and Apple MPS still run the pipeline when CUDA is absent.
5. Create `data/documents`, `data/output`, `runs`, `models/cache`, and `configs/languages`.
6. Fail with the variable name when `INDICOCR_MODEL_PATH`, `CHITRAPATHAK_MODEL_PATH`, or `CHECKPOINT` is set and that path does not exist. It does not fetch the file.

`./run_pipeline.sh` loads `.env` from the repository root when that file exists, then starts `python -m pipeline.cli`. The first time, if no interpreter can import PyYAML, Pillow, NumPy, and pypdfium2, the launcher creates `.venv` and installs `requirements.txt`. It does not reinstall when those imports already succeed. `PIPELINE_PYTHON` selects the interpreter when you set it. Otherwise the launcher uses `.venv`, then `python3.13`, then `python3`.

`PYTHONPATH` inside the launcher is `src:models:.`, so `indic_ocr_pipeline`, `models`, `pipeline`, `evaluation`, and `training` import without an editable install.

Check the machine without running stages:

```bash
./run_pipeline.sh --check
```

Preflight checks the Python version, required packages, configs, source PDFs or an existing document manifest, writable `runs/`, and free disk. A missing frozen exam, GPU, Tesseract binary, or model file is a warning at this step. The frozen exam is required when the frozen stage runs. GPU, Tesseract, and model files are required only when `configs/pipeline.yaml` sets `runtime.require_gpu`, `runtime.require_tesseract`, or `runtime.require_ocr_models`.

## Put the inputs in place

Source PDFs go in `data/documents/`, or in the directory you pass with `--input-dir`. Ingestion reads the top level of that directory.

The frozen exam is `data/eval_data/`:

- `data/eval_data/validation.jsonl` — 100 human-verified lines
- `data/eval_data/images/` — the matching line images

Image names in the JSONL are resolved under that directory. The exam is scored once after a checkpoint is chosen. It is not a training file, a validation file, a hard-mining file, or a threshold-tuning file.

Model weights stay outside git. Point at files you already have, or pass a Hugging Face id on the command when you are ready to download.

```bash
export INDICOCR_MODEL_PATH=/path/to/indicocr
export CHITRAPATHAK_MODEL_PATH=/path/to/chitrapathak
export CHECKPOINT=/path/to/parseq.ckpt
export TESSERACT_CMD=/opt/homebrew/bin/tesseract
# Optional. Leave unset to auto-detect CUDA, then MPS, then CPU.
# export MARATHI_OCR_DEVICE=cuda:0
# export CUDA_VISIBLE_DEVICES=0
```

Device selection is automatic. Setup and each run use CUDA when this machine has it, Apple MPS when it does not, and CPU when neither exists. Set `MARATHI_OCR_DEVICE` only to force `cuda`, `cuda:0`, `mps`, or `cpu`. `DATA_ROOT` and `MODEL_ROOT` override data and model directories. `CHECKPOINT` sets `training.base_checkpoint`.

You can put the same values in `.env` (loaded by both scripts, never committed) or in `configs/local.yaml` (also never committed). `configs/local.yaml.example` shows the shape.

`configs/ocr.yaml` keeps `allow_network` false at the dataset-loader layer: the YAML loader rejects a config that turns network access on. Hugging Face downloads happen only when you pass a link or an `org/repo` id on the command, and only through the weight resolver, into `models/cache`.

## One command

```bash
./run_pipeline.sh
```

With no flags this runs every stage: preflight, ingest, render, detect, crop, OCR with the three default adapters, consensus, calibration, export, training, checkpoint scoring, selection, the frozen exam, and the report.

A failed stage stops the run. The log names the stage, the module, the error, and the log path. Run the same command again. Completed stages stay skipped. `FAILED` and `RUNNING` stages are retried. `--resume` is accepted and does not change that behavior.

```bash
./run_pipeline.sh --dry-run
./run_pipeline.sh --from ocr
./run_pipeline.sh --stage dataset
./run_pipeline.sh --stage train
./run_pipeline.sh --stage eval
```

`--from` accepts a stage name or an alias: `training`, `evaluation`, `eval`, `lines`, `crops`, `review`, `calibration`, `dataset`.

## Command reference

| Flag | Effect |
| --- | --- |
| `--check` | Preflight only. |
| `--dry-run` | Print the stage plan and stop. |
| `--resume` | Accepted. Completed stages are already skipped. |
| `--stage dataset` | Ingest through export. |
| `--stage train` | Training only. |
| `--stage eval` | Score checkpoints, select, frozen exam, report. |
| `--from STAGE` | Start at that stage and continue. |
| `--generate` | Dataset section only, same as `--stage dataset`. |
| `--build-golden` | Dataset section, and export writes `data/golden/`. |
| `--train` | Training section only. |
| `--eval` | Evaluation section only. |
| `--language NAME` | Load `configs/languages/NAME.yaml`. |
| `--input-dir DIR` | PDF directory for this invocation. |
| `--model-N NAME` | OCR model N, for N from 1 to 5. Use 2 to 5 models together. |
| `--model-N-path PATH` | Local weights. Used when the path exists. |
| `--model-N-hf ID` | Hugging Face id or URL, used when the local path is missing. |
| `--indicocr PATH_OR_URL` | Override the IndicOCR adapter. |
| `--chitrapathak PATH_OR_URL` | Override the Chitrapathak adapter. |
| `--checkpoint PATH_OR_URL` | PARSeq base weights. Local file wins over a link. |
| `--base-checkpoint PATH` | Local PARSeq checkpoint. Wins over `--base-checkpoint-hf`. |
| `--base-checkpoint-hf ID` | Hugging Face id or link for the PARSeq checkpoint. |
| `--model NAME` | Training backend name. Default `parseq`. |
| `--human-audit-budget N` | Review at most N lines, stratified. Replaces the percentage sample. |
| `--auto-accept-threshold P` | A class may be auto-accepted only when its precision lower bound is at least P. P is between 0 and 1. Omit it and nothing is auto-accepted. |
| `--dataset-size N` | Keep whole documents until about N crops are included. |
| `--train-ratio` | Train share of documents. Together with `--val-ratio`, must leave room for a held-out split. |
| `--val-ratio` | Development validation share. |
| `--offline` | Do not contact Hugging Face or W&B. |
| `--publish` | Upload the selected checkpoint. Refused together with `--offline`. |
| `--fixture` | Isolated synthetic run under `runs/fixture_workspace`. Does not rewrite production manifests. |
| `--root PATH` | Repository root. Default is this checkout. |

Passing more than one of `--generate` / `--build-golden`, `--train`, and `--eval` runs the full pipeline instead of one section.

A single model flag is rejected. Names are lowercase identifiers of up to 32 characters (`a-z`, digits, underscore). Hyphens in a name are turned into underscores.

### Examples

Default three adapters, using paths from the environment:

```bash
./run_pipeline.sh --language marathi --input-dir data/documents --generate
```

Two local models, then a third from Hugging Face if it is not already on disk:

```bash
./run_pipeline.sh \
  --language marathi \
  --input-dir data/documents \
  --model-1 indicocr --model-1-path "$INDICOCR_MODEL_PATH" \
  --model-2 chitrapathak --model-2-path "$CHITRAPATHAK_MODEL_PATH" \
  --model-3 tesseract --model-3-path "$TESSERACT_CMD" \
  --generate \
  --human-audit-budget 500
```

An extra transformers recognizer plus the two adapters you already have:

```bash
./run_pipeline.sh \
  --model-1 indicocr --model-1-path "$INDICOCR_MODEL_PATH" \
  --model-2 tesseract --model-2-path "$TESSERACT_CMD" \
  --model-3 custom --model-3-path /models/custom --model-3-hf org/custom-ocr \
  --generate
```

Build gold after review, train from a local checkpoint, then evaluate. The frozen exam is still scored only after selection:

```bash
./run_pipeline.sh \
  --build-golden \
  --train \
  --eval \
  --base-checkpoint /models/parseq.ckpt
```

Same run when the checkpoint is a Hugging Face link and the local file is absent:

```bash
./run_pipeline.sh --train --eval --base-checkpoint-hf org/parseq-checkpoint
```

Audit 400 lines and allow auto-accept only for a class whose 95% lower bound is at least 0.98. Sensitive lines stay out either way:

```bash
./run_pipeline.sh --generate --build-golden \
  --human-audit-budget 400 \
  --auto-accept-threshold 0.98
```

## Stage graph

```text
preflight
  → ingest          data/output/documents.jsonl
  → render          page images, data/output/pages.jsonl
  → detect          data/output/regions.jsonl
  → crop            data/output/lines.jsonl, data/output/crops.jsonl
  → ocr             data/output/predictions.jsonl
  → compare         data/output/consensus.jsonl
  → calibrate       data/review/calibration/sample.jsonl
                    data/calibration/sample.jsonl
  → export          data/datasets/train.jsonl
                    data/datasets/val.jsonl
                    data/datasets/test.jsonl
                    data/golden/records.jsonl
                    data/calibration/consensus_quality.json
                    data/auto_accepted/records.jsonl
  → train           runs/<run_id>/checkpoints/
  → eval_checkpoints
  → select          BEST_CHECKPOINT
  → frozen          evaluation/frozen100/metrics.json
  → report          reports/final_report.html
```

Ingest does not depend on preflight, so editing `configs/pipeline.yaml` does not rerender PDFs. Changing the OCR configuration invalidates OCR and everything after it. Rendering, detection, and crops stay.

Each stage records a fingerprint of its inputs and config. A later invocation skips the stage when the status is `COMPLETE`, the outputs still validate, and the fingerprint is unchanged. A finished run is sealed. The next invocation creates `runs/<timestamp>_<n>/`. `runs/latest` points at the newest run. The ledger that survives across runs is `runs/pipeline_state.json`.

On failure the log says to rerun `./run_pipeline.sh`. Recovery for training is: approve human gold, pass `--checkpoint` or `--base-checkpoint` with a local file or a Hugging Face link, then run the same command. Recovery for the frozen stage is: restore `data/eval_data`, then run the same command.

## Dataset creation

### Ingest, render, detect, crop

Ingestion discovers PDFs, hashes them, and writes one document row per file. Rendering rasterizes pages with the DPI in `configs/rendering.yaml`. Region detection writes blocks. Line detection writes one stable line id and one crop image per physical line. A crop is a rectangle cut from the rendered page, not a redrawn glyph.

Paths come from `configs/pipeline.yaml`:

| Data | Path |
| --- | --- |
| Source PDFs | `data/documents` |
| Page images | `data/pages` |
| Regions | `data/detected/regions` |
| Full-line crops | `data/crops` |
| Partial crops | `data/crops/partial` |
| Review crops | `data/crops/review` |
| Raw predictions | `data/predictions` |

`safety.overwrite_existing_outputs` is false. Completed renders, crops, and manifests are kept.

### OCR models

With no `--model-N` flags the three adapters run in this order: IndicOCR, Chitrapathak, Tesseract. Their code lives in `models/indicocr`, `models/chitrapathak`, and `models/tesseract`. IndicOCR recognition code, including PARSeq, lives in `models/IndicPhotoOCR`. Weights are not in those directories.

Resolution order for every model:

1. An existing local path, from `--model-N-path`, `--indicocr`, `--chitrapathak`, or the environment.
2. Otherwise the Hugging Face id or URL from `--model-N-hf` or the matching flag.
3. The download is stored once under `models/cache` and reused. The repo id and the snapshot revision are recorded.
4. A missing local path with no Hugging Face id is an error for a model you named on the command. Setup does not try to fix that by downloading.

A direct file URL is saved under `models/cache` by its filename. An `https://huggingface.co/org/repo/resolve/<rev>/<file>` link keeps that revision. An `org/repo` id downloads the snapshot. If the snapshot contains one weight file, that file is used. If it contains several weight files and exactly one `.ckpt`, that `.ckpt` is used. Two or more `.ckpt` files is an error: pass the file URL. Accepted suffixes are `.ckpt`, `.pt`, `.pth`, `.bin`, and `.safetensors`.

Each prediction row stores the model name, revision, prompt, model path, the raw string, the stored transcription, latency, and the backend version. The raw string is kept. Comparison uses a separate key and does not replace the stored transcription.

IndicOCR and Tesseract are not loaded by the generic transformers class. A model whose name is not `indicocr`, `chitrapathak`, or `tesseract` is a transformers image-to-text adapter (`models/transformers_ocr`). It loads only from the local cache (`local_files_only`) after the resolver has placed the snapshot there.

Tesseract’s language comes from the language profile when you pass `--language`. `configs/languages/marathi.yaml` sets `tesseract_lang: mar+eng` and the prompt `Transcribe the line exactly as it appears.` `OCR_LANG` overrides the YAML language for that process. The default in `configs/ocr.yaml` is also `mar+eng` with page segmentation mode 7 (a single line).

### Text comparison

Storage form, used for human labels, is Unicode NFC with the ends trimmed. Punctuation, digits, and internal spaces stay.

The comparison key is NFC, trimmed, with internal whitespace collapsed to a single space. That key is how agreement is decided. It is never written back as the gold transcription.

### Consensus

`data/output/consensus.jsonl` has one row per crop. For the default three models the class names stay compatible with the existing review and tests:

| Situation | `agreement_class` | `status` | `consensus_status` |
| --- | --- | --- | --- |
| All three strings match | `ALL_3_AGREE` | `AUTO_3_AGREE` | `CONSENSUS_3_3` |
| Two match, the third differs | `TWO_AGREE` | `AUTO_2_AGREE` | `CONSENSUS_2_3` |
| All three differ | `ALL_DIFFER` | `HUMAN_REVIEW` | `NO_CONSENSUS` |
| Any engine returned empty | `EMPTY` | `HUMAN_REVIEW` | `EMPTY` |
| The line boundary is marked bad | `BAD_BOUNDARY` | `HUMAN_REVIEW` | `BAD_BOUNDARY` |

For any other model count the unanimous label is `CONSENSUS_N_N` and a strict majority is `CONSENSUS_K_N`. A tie is `NO_CONSENSUS`. Every row has `gold: false`.

`pairwise_similarity` is one minus the normalized Levenshtein distance for every pair, after the comparison key is applied. A third model that missed one character does not look like a third model that produced a different string. The suggested text is the majority string when there is a majority. It is still not gold.

`sensitive` is true when any prediction contains a digit (any Unicode decimal or other-number character), a money mark (`₹ $ € £`), a percent sign, an `@`, or a URL. A sensitive line is always eligible for audit. It is never auto-accepted, including when every model agreed.

### Human audit

Calibration is deterministic. The seed is `3407`.

Without `--human-audit-budget`, the sample uses the rates in `configs/pipeline.yaml`:

| Class | Default rate |
| --- | --- |
| `ALL_3_AGREE` | 5% |
| `TWO_AGREE` | 20% |
| `ALL_DIFFER`, `EMPTY`, `BAD_BOUNDARY` | 100% |

With `--human-audit-budget N`, the sample contains at most N lines. Sensitive lines are taken first and count against N. The remaining budget is weighted toward disagreement and 2-agree, then unanimous lines:

| Class | Share of the remaining budget |
| --- | --- |
| `ALL_DIFFER`, `NO_CONSENSUS` | 0.45 |
| `TWO_AGREE`, `CONSENSUS_2_3` | 0.35 |
| `ALL_3_AGREE` | 0.10 |
| `EMPTY`, `BAD_BOUNDARY` | 0.05 each |

Inside a class, lines are drawn round-robin across documents and across short, medium, and long text, so one document or one length does not consume the budget. The same seed reproduces the same sample. Being selected for audit does not set `gold`.

The sample is written to both `data/review/calibration/sample.jsonl` and `data/calibration/sample.jsonl`.

### Review

The review server is localhost only. It does not run OCR, download assets, or export gold by itself.

```bash
python -m indic_ocr_pipeline.review.app
```

It listens on `127.0.0.1:8765` unless `configs/review.yaml` or `--host` / `--port` say otherwise. `--host` may only be `127.0.0.1` or `localhost`. The process expects `data/output/review_batch.jsonl`. Build that queue from the calibration sample with the existing review scripts (`scripts/sample_review.py`, `scripts/build_review_manifest.py`) before opening the UI.

The screen shows the line image, the page context, every model string, and the suggestion. The suggestion is labeled as not verified. Buttons:

| Button | Key | Decision |
| --- | --- | --- |
| Approve | V | `human_verified`. The typed text must match the candidate you accept. |
| Edit | C | `human_corrected`. The typed text replaces the candidate. |
| Reject | R | The line is not gold. |
| Bad boundary | B | The crop itself is wrong. Not gold. |

The typed text is the human transcription. The verified row stores it as the gold text. Export copies that text to `human_verified_text` on `data/golden/records.jsonl`. Storage form is NFC with the ends trimmed. The reviewer id comes from `configs/review.yaml` (`local_reviewer` by default). Decisions land in `data/output/verified.jsonl` with `review_origin=human`, a reviewer id, and a timestamp.

Reviewer ids `cursor`, `agent`, `assistant`, and `auto` are blocked. A row with one of those ids cannot become gold.

### Quality report and auto-accept

Export joins human decisions to consensus rows and writes `data/calibration/consensus_quality.json` plus `consensus_quality.txt`. For each agreement class the report has:

- population (how many lines fell in that class)
- audited count
- correct count (human text equals the suggested text)
- precision
- a Wilson score interval
- whether auto-accept is allowed

Auto-accept runs only when all of these are true:

1. You passed `--auto-accept-threshold`.
2. The class is a unanimous or majority class, not `EMPTY`, `BAD_BOUNDARY`, `ALL_DIFFER`, or `NO_CONSENSUS`.
3. The interval’s lower bound is at least the threshold.
4. The line is not sensitive.
5. The line is not already human gold.
6. The suggested text is non-empty.

Those rows are written to `data/auto_accepted/records.jsonl` with `verification_method` set to `auto_consensus_<class>`, `review_origin` set to `auto_consensus`, and `gold: false`. They are a separate training source. They are not copied into `data/golden/`, and they are not labeled human.

If you omit `--auto-accept-threshold`, every class stays a suggestion and the auto-accepted file is empty.

### Golden data and splits

`data/golden/records.jsonl` contains only rows with `verification_method=human`. A row is admitted only when:

- the crop id is not in the invalidated Phase 10 archive
- `review_origin` is `human`
- `reviewer_id` is present and not blocked
- `reviewed_at` is present
- the decision is `HUMAN_VERIFIED` or `HUMAN_CORRECTED` (`accept_candidate` and `correct_text` map to those)
- `gold_text` is a non-empty string
- `document_id`, `page_id`, a line id, a crop path, and the original model strings are present

Rejected and bad-boundary rows are listed in `data/review/gold_rejected.jsonl` with the reason. They do not enter training.

Historical Phase 10 review rows were not produced by a person. The first export copies them to `data/review/archive/invalidated_phase10/`, writes `invalidated_ids.json`, and marks them `INVALIDATED_REVIEW`. The original `data/output/verified.jsonl` stays where it is. Those crop ids cannot become gold.

Export then splits human gold by document, so crops from one PDF stay in one split. Defaults in `configs/dataset.yaml` are train 0.8, validation 0.1, test 0.1, seed 3407. Frozen-exam ids are removed before the split. Outputs:

| File | Contents |
| --- | --- |
| `data/datasets/train.jsonl` | Human gold for training. |
| `data/datasets/val.jsonl` | Development validation. Used to pick the checkpoint. |
| `data/datasets/test.jsonl` | Held-out split. Not used to train, to tune the auto-accept threshold, or to pick the checkpoint. |
| `data/golden/records.jsonl` | The same human rows, with `human_verified_text`. |

Default training reads the human manifests only. `assert_training_rows_safe` rejects a row whose `verification_method` starts with `auto_`, a frozen-exam id, or a path that points at the frozen exam.

## Training

Training reads `data/datasets/train.jsonl` and `data/datasets/val.jsonl`. It refuses to start when either manifest is empty, when a frozen-exam id is present, or when the base checkpoint cannot be resolved. That is intentional. Consensus is not a substitute for a person and a checkpoint.

The production backend is PARSeq (`training.backend: parseq`), implemented under `models/IndicPhotoOCR`. Defaults in `configs/training.yaml`: 10 epochs, batch size 8, learning rate `0.0001`, `val_checks: 10`. Leave `training.steps` unset to use epochs times the number of batches. Set `steps` when you want a fixed optimizer-step budget.

Validation is spread across that budget. With 1,000 steps and 10 checks, validation runs at steps 100, 200, …, 1000. Lightning saves a checkpoint at each of those steps. The frozen exam is not loaded in this loop.

```bash
./run_pipeline.sh --train --base-checkpoint /models/parseq.ckpt
```

`--checkpoint` is the same idea and overrides `--base-checkpoint` when both are passed. A local file is used as-is. A Hugging Face link is resolved once into `models/cache`.

Checkpoints land in `runs/<run_id>/checkpoints/`. `training/complete.json` marks the stage finished.

## Evaluation and the frozen exam

Checkpoint selection scores every saved checkpoint on the development validation split. The primary metric is strict character error rate from `evaluation/core_metrics.py`. Lower is better. The scoreboard is `runs/<run_id>/evaluation/checkpoint_scoreboard.json`. The chosen path is written to `runs/<run_id>/BEST_CHECKPOINT`.

Diagnostics kept with that score include content CER, WER, exact match, character accuracy, punctuation, matra, conjunct, numeral, substitutions, deletions, insertions, length buckets, and issue types. The report prints CER, WER, and exact match for the frozen exam as well.

Selection refuses a scoreboard that contains frozen-exam ids. After a checkpoint is chosen, the frozen stage scores `data/eval_data` once and writes `runs/<run_id>/evaluation/frozen100/metrics.json` with `used_for_selection: false`. That file is an exam result. It is not an input to the next training run and it is not an input to `--auto-accept-threshold`.

```bash
./run_pipeline.sh --eval
python -m evaluation score --help
```

The final report is `reports/final_report.html` inside the run directory. It includes crop, prediction, and candidate counts, the train and validation sizes, the scoreboard, and the frozen CER, WER, and exact-match rate.

## Run directory

```text
runs/<run_id>/
  config_snapshot/
  state.json
  logs/pipeline.log
  training/
  checkpoints/
  evaluation/raw_scores.json
  evaluation/checkpoint_scoreboard.json
  evaluation/frozen100/metrics.json
  BEST_CHECKPOINT
  reports/final_report.html
```

`runs/pipeline_state.json` is the cross-run ledger. Do not delete it if you want resume. Do not delete `runs/<run_id>/` if you still want that run’s checkpoints and exam score.

## Configuration files

| File | Role |
| --- | --- |
| `configs/pipeline.yaml` | Canonical paths, safety, seed 3407, calibration rates, training budget, frozen exam path. |
| `configs/run.yaml` | Run overlay, including the frozen-exam path. |
| `configs/ocr.yaml` | Enabled backends, Tesseract language and PSM, IndicOCR and Chitrapathak device and dtype. |
| `configs/rendering.yaml` | Page render settings. |
| `configs/detection.yaml` | Region and line detection. |
| `configs/cropping.yaml` | Crop geometry. |
| `configs/dataset.yaml` | Document split and gold statuses. |
| `configs/training.yaml` | PARSeq hyperparameters and the base checkpoint. |
| `configs/evaluation.yaml` | Primary metric `cer`, minimize, frozen exam path. |
| `configs/review.yaml` | Reviewer id, host, port. |
| `configs/languages/marathi.yaml` | Script, Tesseract language, transcription prompt. |
| `configs/local.yaml` | Machine overrides. Gitignored. |

Add a language by copying `configs/languages/marathi.yaml` to `configs/languages/<name>.yaml` and passing `--language <name>`. Set `tesseract_lang` to a language pack installed on the machine, and set `prompt` to the line the extra transformers adapter should see. You do not edit the consensus or gold code for that language.

## Repository layout

| Path | Role |
| --- | --- |
| `setup.sh` | Create `.venv`, install `requirements.txt`, install this package, check Python, create directories, then run the pipeline. The page reader downloads its own weights on first use. |
| `run_pipeline.sh` | Launcher for the line pipeline. |
| `ocr_service/run.sh` | Launcher for the Indic Vachak page reader. |
| `requirements.txt` | The only Python package list. Pipeline and page reader both use it. |
| `.env.example` | The only environment example. Copy it to `.env`. |
| `.gitignore` | The only ignore file. |
| `pipeline/` | Orchestrator, stage graph, resume, model registry, consensus export, audit budget, quality report. |
| `src/indic_ocr_pipeline/` | Ingest, render, detect, crops, OCR adapters’ call sites, review server, text policy. |
| `models/IndicPhotoOCR/` | IndicOCR and PARSeq code. |
| `models/indicocr/` | IndicOCR adapter. |
| `models/chitrapathak/` | Chitrapathak adapter. |
| `models/tesseract/` | Tesseract adapter. |
| `models/transformers_ocr/` | Extra image-to-text adapter for models you name yourself. |
| `models/cache/` | Hugging Face snapshots and resolved-model metadata. Gitignored by the weight patterns. |
| `evaluation/` | CER, WER, exact match, and the diagnostic reports. |
| `training/` | Split assignment, frozen-id guard, checkpoint selection. |
| `scripts/train/train.py` | Fine-tune on `data/datasets/train.jsonl` without the orchestrator. |
| `scripts/validation/validate.py` | Score a checkpoint on `data/datasets/val.jsonl`. |
| `scripts/eval/eval.py` | Score a checkpoint on `data/eval_data`. |
| `data/golden/` | Human gold only. |
| `data/auto_accepted/` | Optional auto-consensus rows. Never labeled human. |
| `data/calibration/` | Audit sample and the precision report. |
| `data/datasets/` | Train, validation, and held-out JSONL. |
| `data/eval_data/` | Frozen exam. Not committed; keep it on the machine that evaluates. |
| `data/output/` | JSONL ledgers. Not committed. |
| `runs/` | One directory per run. Not committed. |
| `docs/` | Architecture, data contracts, label policy, runbook. |

## What is not in git

`.gitignore` keeps private and generated files off the remote:

- `.env`, `ocr_service/ngrok.yml`, `configs/local.yaml`, `.venv`
- `model_cache/` and `models/cache/`
- `data/`, `input/`, `reports/`, `runs/`, `frozen/`
- `data/output/*.jsonl` and manifest locks
- `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, `checkpoints/`, `models/**/weights/`, `wandb/`

Engine source under `models/` is committed. Weights are not. Clone the repo, run `./setup.sh`, put PDFs and weights on the machine, then `./run_pipeline.sh`.

## Tests

```bash
python -m pytest
python -m ruff check pipeline src/indic_ocr_pipeline training tests
```

`./run_pipeline.sh --fixture` is the isolated synthetic path used by tests. It writes under `runs/fixture_workspace` and does not touch the production ledgers.

The tests that protect the label rules check that three-model class names stay `ALL_3_AGREE`, `TWO_AGREE`, `ALL_DIFFER`, `EMPTY`, and `BAD_BOUNDARY`, that `gold` is false, that a near-miss is more similar than a different third string, that sensitive lines are not auto-accepted, that the audit budget prefers disagreement, that auto-accept waits for the interval lower bound, and that training rejects `auto_*` rows and frozen-exam ids.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `check: failed` | Read the FAIL lines. Install the named package or set the named path, then `./run_pipeline.sh --check` again. |
| `./setup.sh` says a model path is missing | The environment variable is set but the file is not there. Fix the path or unset it. Setup will not download it. |
| `use between 2 and 5 models` | Pass at least two `--model-N` slots, or pass none and keep the default three adapters. |
| `unknown language` | Add `configs/languages/<name>.yaml` or pass `--language marathi`. |
| Stage FAILED | Open `runs/latest/logs/pipeline.log`. Fix the cause. Run `./run_pipeline.sh` again. Completed stages stay skipped. |
| Training failed: no approved rows | A person still has to Approve or Edit lines. Consensus, the auto-accepted file, and the Phase 10 archive are not gold. |
| Training failed: checkpoint missing | Pass `--checkpoint` or `--base-checkpoint` with a local file, or `--base-checkpoint-hf` with a Hugging Face id. |
| Training failed: auto-accepted row | Those rows were copied into a training manifest. Remove them. Default training is human-only. |
| Frozen id in training | Remove that sample from `data/datasets/`. Export already excludes frozen ids; do not merge the exam into train by hand. |
| OCR count mismatch | Predictions must be exactly models × crops, and candidates must equal crops. A missing backend row fails the compare stage instead of inventing a vote. |
| Tesseract language error | Install the `mar` (or the language you configured) traineddata, or set `tesseract_lang` in the language YAML. |
| IndicOCR or Chitrapathak device error | Leave `MARATHI_OCR_DEVICE` unset so the run picks CUDA, MPS, or CPU. Set it only to force one of those. |
| Hugging Face download refused | You passed `--offline`, or the YAML safety flag is doing what it should. Drop `--offline` only for the invocation that should fill `models/cache`. |
| Review app exits immediately | `data/output/review_batch.jsonl` is missing. Build the queue from the calibration sample first. |
| Auto-accept wrote nothing | The threshold was omitted, the lower bound is below it, the class is a disagreement, or every candidate line was sensitive. Read `data/calibration/consensus_quality.txt`. |

## Further reading

- [Runbook](docs/RUNBOOK.md)
- [Pipeline architecture](docs/PIPELINE_ARCHITECTURE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data contracts](docs/DATA_CONTRACTS.md)
- [Data model](docs/DATA_MODEL.md)
- [Label policy](docs/LABEL_POLICY.md)
- [Review policy](docs/REVIEW_POLICY.md)
- [Split policy](docs/SPLIT_POLICY.md)
- [Quality control](docs/QUALITY_CONTROL.md)
- [Offline operation](docs/OFFLINE_OPERATION.md)
- [Pipeline notes](docs/PIPELINE.md)
- [Troubleshooting notes](docs/TROUBLESHOOTING.md)
