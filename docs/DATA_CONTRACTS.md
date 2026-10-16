# Data contracts

Paths are relative to the repository root unless noted. Three OCR backends
never become three training samples. For N crops the pipeline requires N
candidate rows and 3N prediction rows.

## Ingest

* Input: PDFs in `data/documents/`
* Output: `data/output/documents.jsonl`
* Identity: `document_id`

## Render

* Input: `data/output/documents.jsonl`
* Output: `data/output/pages.jsonl`, `data/pages/`
* Identity: `page_id`

## Region detection

* Input: `data/output/pages.jsonl`
* Output: `data/output/regions.jsonl`
* Identity: `region_id`

## Line detection

* Input: `data/output/regions.jsonl`
* Output: `data/output/lines.jsonl`, `data/output/crops.jsonl`, `data/crops/`
* Identity: `line_id`, `crop_id`

## OCR

* Input: `data/output/crops.jsonl`
* Output: `data/output/predictions.jsonl`
* One row per crop per backend: `indicocr`, `chitrapathak`, `tesseract`
* Count: predictions = 3 × crops

## Comparison

* Input: predictions and crops
* Output: `data/output/consensus.jsonl`
* Count: candidates = crops

Each candidate stores all three strings and one class:

| Class | Status | Review | Gold |
| --- | --- | --- | --- |
| `ALL_3_AGREE` | `AUTO_3_AGREE` | no | no |
| `TWO_AGREE` | `AUTO_2_AGREE` | no | no |
| `ALL_DIFFER` | `HUMAN_REVIEW` | yes | no |
| `EMPTY` | `HUMAN_REVIEW` | yes | no |
| `BAD_BOUNDARY` | `HUMAN_REVIEW` | yes | no |

`proposed_text` is the majority string for the two auto classes. It is not a
label. Dissenting text stays on the row.

## Calibration

* Input: `data/output/consensus.jsonl`
* Output: `data/review/calibration/sample.jsonl`
* Seed `3407`
* Rates: 5% of `ALL_3_AGREE`, 20% of `TWO_AGREE`, 100% of the three review classes
* Selection is `sha256(seed:crop_id)`, so it does not depend on file order
* A sampled row is still not gold

## Gold

A training row must include `reviewer_id`, `reviewed_at`, `decision`,
`gold_text`, `sample_id`, `document_id`, `page_id`, `line_id`, `crop_path`,
and the original OCR strings.

Admitted decisions: `HUMAN_VERIFIED`, `HUMAN_CORRECTED`.

Refused: `HUMAN_REJECTED`, `HUMAN_BAD_BOUNDARY`, missing reviewer, missing
`review_origin: human`, and any crop id in
`data/review/archive/invalidated_phase10/invalidated_ids.json`.

The archive is a copy. `data/output/verified.jsonl` is not deleted. Rows in the
archive are marked `INVALIDATED_REVIEW`.

## Dataset export

* Input: human decisions that pass the gold gate
* Output: `data/datasets/train.jsonl`, `data/datasets/val.jsonl`, `data/datasets/test.jsonl`
* Split is by `document_id`
* Every row has `source: human_review` and `verification_status`
* The frozen exam is rejected if a row references its ids or its directory

## Page-reader captures

Indic Vachak writes finished page readings to `data/ocr_captures/`. `pages/` is the image that was read, `text/` is the raw text, and `captures.jsonl` is the ledger. Each row has `source: model_capture` and `gold: false`. Training does not read this folder. A person still has to accept or correct the text before it can enter `data/datasets/`.

## Training

* Input: `data/datasets/train.jsonl` and `data/datasets/val.jsonl` only
* Output: `runs/<run_id>/checkpoints/`, `runs/<run_id>/training/complete.json`
* Refuses empty gold, raw prediction files, review queues, invalidated reviews,
  and `frozen/marathi100`

## Checkpoint selection

* Input: validation-split predictions for each checkpoint
* Output: `runs/<run_id>/evaluation/checkpoint_scoreboard.json`
* Output: `runs/<run_id>/BEST_CHECKPOINT`
* Primary metric: strict CER, lower wins
* Ties break by checkpoint name
* `frozen_used` must be false

## Frozen exam

* Input: the selected checkpoint and `frozen/marathi100/validation.jsonl`
* Output: `runs/<run_id>/evaluation/frozen100/metrics.json`
* `used_for_selection` is false
* This directory is not a training source, a hard-negative mine, or a
  calibration sample

## Report

* Output: `runs/<run_id>/reports/final_report.html`
* Run config copy: `runs/<run_id>/config_snapshot/`
* Ledger copy: `runs/<run_id>/state.json`
