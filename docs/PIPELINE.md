# Pipeline

One command processes every PDF already in the source folder, then opens human review. Completed rows are skipped, so the same command can be run again after adding more PDFs. It does not approve text or export gold.

```bash
./run_pipeline.sh
./run_pipeline.sh --stage train
./run_pipeline.sh --stage eval
```

Put the PDFs in `data/raw/documents/` first. Model paths live in `.env` (see `.env.example`). `--datapoints` caps the review queue and the later train split; whole documents stay in one split. `--stage train` writes that split and does not update model weights. `--stage eval` scores the held-out documents and can be started after train has finished. The review page stays open until you stop it. A person still accepts or corrects each crop. `hindi` and `english` are reserved until a language profile is enabled.

Individual stages remain resumable: completed manifest rows are skipped after their payload is checked, and a crash leaves failed or unfinished rows retryable.

```bash
python3 -m indic_ocr_pipeline.stages --list-stages
python3 -m indic_ocr_pipeline.stages --stage status
python3 scripts/pipeline_status.py
```

| Command | Stage | Phase | Behavior today |
| --- | --- | --- | --- |
| `scripts/ingest_documents.py` | ingest | 2 | Hashes source PDFs into `manifests/documents.jsonl` |
| `scripts/render_pages.py` | render | 3 | Renders each PDF page to `data/rendered/pages/` |
| `scripts/detect_regions.py` | detect | 4 | Writes region boxes to `manifests/regions.jsonl` |
| `scripts/detect_lines.py` | lines | 5 | Writes physical lines and full-line crops |
| `scripts/generate_crops.py` | crops | 5 | Exits 2 |
| `scripts/run_ocr.py` | ocr | 6 | Runs IndicOCR, Chitrapathak, then Tesseract and writes predictions and candidates |
| `scripts/compare_ocr.py` | compare | 10 | Compares existing predictions and writes the review queue |
| `scripts/score_candidates.py` | score | 11 | Same ranking pass as compare. The score is not a probability |
| `scripts/sample_review.py` | review | 12 | Builds stratified `manifests/review_batch.jsonl` |
| `scripts/build_review_manifest.py` | review | 12 | Same as sample_review |
| `scripts/review_app.py` | review | 12 | Offline localhost HTML reviewer |
| `scripts/export_review_package.py` | review | 13 | Exits 2 |
| `scripts/import_review_results.py` | review | 14 | Exits 2 |
| `scripts/build_gold_dataset.py` | gold | 15 | Exits 2 |
| `scripts/build_train_val_split.py` | split | 16 | Exits 2 |
| `scripts/audit_*.py` | audit | 17 | Exits 2 |
| `scripts/pipeline_status.py` | status | 1 | Prints manifest counts |

`--verbose` enables debug logs. Default logs are one line per event, with counters such as `processed`, `completed`, `skipped`, `failed`, `duration_seconds`, and `throughput` when a stage reports them.

OCR backends are selected per process. IndicOCR and Chitrapathak read `device`, `batch_size`, `dtype`, `max_image_size`, and `concurrency` from `configs/ocr.yaml`. `MARATHI_OCR_DEVICE` overrides the shared device and the two GPU backends. Tesseract stays on CPU.

Random crop generation must use `cropping.random_seed` and store that seed on the crop row. The same inputs and seed must reproduce the same boxes.

No stage may open a network connection. `configs/pipeline.yaml` sets `allow_network: false`, and the loader rejects `true`.
