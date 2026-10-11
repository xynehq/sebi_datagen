# Pipeline architecture

`./run_pipeline.sh` is the only user entry. It sets `PYTHONPATH` and runs
`python -m pipeline.cli`. The shell script does not call other shell scripts.

```text
run_pipeline.sh
  → pipeline.cli
    → pipeline.orchestrator
      → stage runners
```

## Stages

| # | Name | Implementation | Skip rule |
| --- | --- | --- | --- |
| 1 | Preflight | `pipeline.preflight` | Always runs when later work is needed |
| 2 | Ingest | `indic_ocr_pipeline.pdf.ingest` | Existing `documents.jsonl` |
| 3 | Render | `indic_ocr_pipeline.rendering.pages` | Existing `pages.jsonl` |
| 4 | Detect | `indic_ocr_pipeline.detection.regions` | Existing `regions.jsonl` |
| 5 | Crop | `indic_ocr_pipeline.detection.lines` | Lines and crops, same count |
| 6 | OCR | `indic_ocr_pipeline.ocr.run` | Predictions = 3 × crops |
| 7 | Compare | `pipeline.dataset_flow.build_consensus` | `consensus.jsonl` matches the counts |
| 8 | Calibrate | `indic_ocr_pipeline.review.audit_sample` | Calibration sample unchanged |
| 9 | Export | `pipeline.gold` and `training.dataset` | Train/val manifests match the gold input |
| 10 | Train | `IndicPhotoOCR.recognition.finetune` or the fixture writer | Same dataset fingerprint and checkpoint dir |
| 11 | Eval | `evaluation.harness.score_corpus` | Same validation rows and checkpoints |
| 12 | Select | `training.selection.select_best` | Strict CER on the validation split |
| 13 | Frozen | `evaluation.harness` on `frozen/marathi100` | Once, after selection |

Training validates the development set during the run, not only at the end.
The gap is `optimizer_steps // val_checks`. One thousand steps and ten checks
validate at 100, 200, ..., 1000, and each of those steps keeps a checkpoint.
Selection then scores those checkpoints with strict CER. Frozen-100 runs once
after that choice.
| 14 | Report | `pipeline.report` | After frozen metrics exist |

Compare writes `manifests/consensus.jsonl`. It does not rewrite
`manifests/candidates.jsonl`.

## Resume

Ledger: `runs/pipeline_state.json`.

Each stage stores status, start, end, input fingerprint, output fingerprint,
config hash, code version, row counts, and errors. Status values are
`NOT_STARTED`, `RUNNING`, `COMPLETE`, `FAILED`, and `INVALIDATED`.

A completed stage is skipped when its outputs still validate and the input
fingerprint and config hash are unchanged. A config or input change marks that
stage and its downstream stages `INVALIDATED`. Upstream stages stay complete.
Preflight is not a data dependency, so a config edit does not invalidate
ingestion unless the source or path section changed.

Each invocation that still has work uses one run directory, `runs/YYYY-MM-DD_NNN`.
`runs/latest` is a symlink and `runs/latest_id.txt` names it. A finished run
directory is not overwritten. The next invocation creates a new id only when
something must run again.

## Commands

```bash
./run_pipeline.sh
./run_pipeline.sh --check
./run_pipeline.sh --dry-run
./run_pipeline.sh --resume
./run_pipeline.sh --stage dataset
./run_pipeline.sh --stage train
./run_pipeline.sh --stage eval
./run_pipeline.sh --from ocr
./run_pipeline.sh --from training
```

`--fixture` runs the synthetic corpus under `runs/fixture_workspace` and does
not touch `manifests/` or `data/` in the repository root.

## Failure

A failed critical stage is stored as `FAILED`. Later stages do not run. The
terminal output names the stage, module, error, log path, and recovery command.

## Metrics

Checkpoint selection minimizes strict CER from `evaluation.core_metrics`.
Content CER, WER, exact match, character accuracy, substitutions, deletions,
insertions, punctuation, matra, conjunct, numeral, reference-id, length
buckets, and issue types are reported by `evaluation.harness.score_corpus`.
Frozen-100 is scored only after the winner is chosen, and its sample ids are
rejected if they appear in the scoreboard.
