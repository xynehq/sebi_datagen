# Repository merge plan

This repository is the single Marathi OCR project. It keeps the dataset
pipeline that already lived here and brings in the training and evaluation
code from `IndicPhotoOCR` on the desktop. A third folder, `Marathi_OCR`, was
inspected and left alone: it is an older Surya/Paddle discovery stack, not the
training repository.

## Old repository A — dataset (`sebi_ocr`)

```text
configs/                  stage YAML (render, detect, ocr, review, splits)
src/indic_ocr_pipeline/  ingestion, rendering, detection, crops, OCR, review
scripts/                  one command per dataset stage
data/output/                documents, pages, regions, lines, crops, predictions
data/                     PDFs, page images, crops, predictions
docs/                     label, review, and split policy
tests/                    dataset stage tests
```

Working stage entry points, still called by the orchestrator:

| Stage | Module |
| --- | --- |
| Ingest | `indic_ocr_pipeline.pdf.ingest` |
| Render | `indic_ocr_pipeline.rendering.pages` |
| Regions | `indic_ocr_pipeline.detection.regions` |
| Lines and crops | `indic_ocr_pipeline.detection.lines` |
| OCR | `indic_ocr_pipeline.ocr.run` |
| Human decision commit | `indic_ocr_pipeline.review.decisions` |

`./run_pipeline.sh` is the only launcher. The older `./run` wrapper and
`scripts/run_pipeline.py` were removed so they cannot start a second pipeline.

Current corpus, left in place:

* 20 documents, 145 pages, 4,189 crops
* 12,567 OCR predictions (3 × 4,189)
* 4,189 legacy candidate rows in `data/output/candidates.jsonl`
* 1,288 verified rows. These are archived as `INVALIDATED_REVIEW` and are not gold.

## Old repository B — training and evaluation (`IndicPhotoOCR`)

```text
IndicPhotoOCR/                 PARSeq / strhub recognition package
scripts/evaluation/            canonical CER, WER, and script diagnostics
scripts/run_canonical_eval.py  frozen piecewise evaluation CLI
IndicPhotoOCR/recognition/finetune.py
dataset /validation.jsonl      100-row frozen Marathi exam, plus images/
checkpoints/, results/, wandb/ historical artifacts, not copied
```

Checkpoint class paths require the package name `IndicPhotoOCR`, so that
package was vendored rather than rewritten. The evaluation package was moved
to top-level `evaluation/` and is the only metric implementation the
orchestrator calls.

Not copied: virtualenvs, Weights & Biases runs, Gujarati image dumps,
historical `results/`, and checkpoint binaries. Those stay in the original
folder. This repo points at a base checkpoint through `CHECKPOINT` or
`configs/local.yaml`.

## New layout

```text
run_pipeline.sh
configs/pipeline.yaml          canonical orchestrator config
configs/training.yaml
configs/evaluation.yaml
configs/local.yaml.example
pipeline/                      orchestrator, state, resume, preflight
src/indic_ocr_pipeline/       existing dataset implementation
evaluation/                    canonical metrics and checkpoint inference
training/                      gold-split guards and checkpoint selection
IndicPhotoOCR/                 PARSeq package used by fine-tuning
frozen/marathi100/             frozen exam (100 rows)
data/                          source images and generated dataset files
runs/                          immutable run directories
```

The dataset package was not relocated. Moving it would break the existing
tests and every stage script without changing behavior. The orchestrator
imports it.

## Duplicated logic

| Area | Decision |
| --- | --- |
| Dataset line detection vs IndicPhotoOCR EAST/TextBPN | Keep the PDF line detector. EAST, TextBPN, and the synthetic-data generators were not used by the pipeline and were removed. |
| `indic_ocr_pipeline.scoring.metrics` vs `evaluation.core_metrics` | Different jobs. Scoring ranks review risk. Evaluation measures CER. The orchestrator uses `evaluation.core_metrics` for checkpoints. |
| `jobs.evaluate_holdout` | Review-outcome report for the old runner. Not model evaluation. |
| `./run`, `run_pipeline.command`, `Makefile` | Removed. `./run_pipeline.sh` is the only launcher. |

## Entrypoints

```text
./run_pipeline.sh
    → python -m pipeline.cli
        → pipeline.orchestrator.execute
            → dataset modules, evaluation.harness, training.selection,
              IndicPhotoOCR.recognition.finetune
```

`python -m evaluation` scores pairs or runs frozen inference. The orchestrator
calls `evaluation.harness.score_corpus` directly.

## Stage graph

```text
preflight → ingest → render → detect → crop → ocr → compare
    → calibrate → export → train → eval_checkpoints → select → frozen → report
```

An OCR config change invalidates OCR and everything after it. It does not
invalidate ingest, render, region detection, or line detection.

## Risks

* Historical `data/output/verified.jsonl` looks like human review and is not.
  Export snapshots those crop ids once and refuses them as gold.
* Production training needs a local base checkpoint and real human gold.
  Without those, training stops. That is intentional.
* PARSeq fine-tuning still strips whitespace inside `JsonlDataset`. That
  behavior was not rewritten.
* Frozen image basenames such as `0001.png` are blocked when a training row
  points at the exam path, not merely because a crop file happens to share a
  short name under `data/crops/`.

## Migration

1. Vendor `evaluation/` and `IndicPhotoOCR/`.
2. Copy the frozen exam to `frozen/marathi100/`.
3. Add the orchestrator, gold gate, and launchers.
4. On the first export, copy `data/output/verified.jsonl` to
   `data/review/archive/invalidated_phase10/` and record the crop ids.
   The original manifest stays where it is.
5. New human decisions committed after that snapshot can enter gold when they
   carry `review_origin: human`.
