# Runbook

## Setup

Python 3.11 or newer. The dataset stages need PyYAML, Pillow, NumPy, and
pypdfium2. OCR needs a local Tesseract with `mar+eng` and local IndicOCR and
Chitrapathak weights. Training and checkpoint inference need PyTorch. CUDA is
used when it is available. Apple MPS is used when CUDA is not. CPU is the
fallback. Nothing in the pipeline downloads weights.

```bash
pip install -r requirements.txt
```

`./run_pipeline.sh` does that install itself the first time the packages are missing.

Optional environment variables:

```bash
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/models
export CHECKPOINT=/path/to/base.ckpt
export CUDA_VISIBLE_DEVICES=0
export INDICOCR_MODEL_PATH=/path/to/indicocr
export CHITRAPATHAK_MODEL_PATH=/path/to/chitrapathak
export PIPELINE_PYTHON=/path/to/python
```

Copy `configs/local.yaml.example` to `configs/local.yaml` for machine paths.
`configs/local.yaml` is gitignored.

## Check, dry run, full run

```bash
./run_pipeline.sh --check
./run_pipeline.sh --dry-run
./run_pipeline.sh
```

## Resume

```bash
./run_pipeline.sh --resume
./run_pipeline.sh
```

Both skip stages whose outputs, inputs, and config hash still match
`runs/pipeline_state.json`. A failed run stays unsealed. The next command
continues `runs/latest` instead of creating a second directory. After a
successful report, the next real change creates `runs/YYYY-MM-DD_NNN` and
leaves the previous directory in place.

Start part-way:

```bash
./run_pipeline.sh --from ocr
./run_pipeline.sh --from training
./run_pipeline.sh --stage dataset
./run_pipeline.sh --stage train
./run_pipeline.sh --stage eval
```

## What the run writes

```text
runs/<run_id>/
  config_snapshot/
  state.json
  logs/pipeline.log
  checkpoints/
  evaluation/checkpoint_scoreboard.json
  evaluation/frozen100/metrics.json
  BEST_CHECKPOINT
  reports/final_report.html
```

`runs/latest` points at the active directory.

## Gold and the frozen exam

Do not treat `AUTO_3_AGREE` or `AUTO_2_AGREE` as labels. Do not import
`data/review/archive/invalidated_phase10/` into training. New gold comes from
human decisions with `review_origin: human`.

`data/eval_data` is the final 100-row exam. Training and checkpoint
selection refuse its ids. It is scored once, after `BEST_CHECKPOINT` is written.

## When a stage fails

The log names the stage, the Python module, the error, and the log file.
Fix the cause and run the same command again:

```bash
./run_pipeline.sh
```

Completed stages are skipped. The failed stage runs again. `--resume` is the
same command.

Training stops when `data/datasets/train.jsonl` has no human-approved rows or
when `training.base_checkpoint` / `CHECKPOINT` is empty and `--checkpoint`
was not given. Pass a local `.ckpt` or a Hugging Face link. That is the
correct stop, not a skipped stage.

## CI fixture

```bash
./run_pipeline.sh --fixture
```

This builds a five-crop corpus under `runs/fixture_workspace`, checks the 3N
prediction invariant, keeps an invalidated review out of gold, selects a
checkpoint on the validation split, and writes a frozen metric file that was
not used for selection. It does not read the real PDFs or the real verified
manifest.

## Tests

```bash
python3 -m pytest
```
