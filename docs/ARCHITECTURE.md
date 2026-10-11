# Architecture

Phase 1 provides the contracts every later stage must honor. Stages do not run OCR or read PDFs yet.

## Authority

| Layer | What it is | Where it lives |
| --- | --- | --- |
| Source truth | Rendered page image from the official PDF | `data/rendered/pages/` and `pages.jsonl` |
| OCR candidate | Text proposed by a local backend | `predictions.jsonl` |
| Candidate consensus | Comparison and review-priority score | `candidates.jsonl` |
| Human decision | Accept, correct, reject, or mark uncertain | `review.jsonl`, `verified.jsonl` |
| Gold label | Text a person confirmed while looking at the crop | `gold.jsonl` |

Tiers stay separate:

| Tier | Meaning | Trainable |
| --- | --- | --- |
| 0 | Source PDF | No |
| 1 | Detected crop | No |
| 2 | OCR candidate | No |
| 3 | High-confidence candidate | No |
| 4 | Human reviewed, including rejections | No |
| 5 | Gold: `human_verified` or `human_corrected` | Yes |

## Stage boundaries

Each stage reads and writes manifests. A stage can be rerun. A completed row is frozen. The same id with the same payload is a resume skip. The same id with different payload is an error. Source PDFs and finished renders are not overwritten unless a config flag and a later stage both allow it. The shipped config keeps overwrite off.

```text
ingest → render → detect → crops → ocr → compare → score → review → gold → split → audit
```

OCR backends are adapters behind `OCRBackend.predict`. IndicOCR, Chitrapathak, and Tesseract run one after another. One backend failing does not delete another backend's predictions. Candidate rows record agreement and leave rank scores empty. Agreement is not a gold label.

The gold record format does not mention a training framework. Chitrapathak can be one OCR assistant during labeling and, later, a model trained on the gold set. Those are different roles.

## What Phase 1 implements

- YAML settings in `configs/`, including an offline guard (`allow_network: false`)
- Dataclass schemas for documents, pages, regions, crops, predictions, candidates, reviews, verified rows, and gold rows
- Hierarchical IDs and a SHA-256 document-id allocator
- Locked, atomic JSONL manifests
- Provenance and leakage checks as pure functions
- `python -m indic_ocr_pipeline.stages --stage status`

## What later phases add

Rendering, OpenCV detection, crop files, real OCR adapters, the HTML review package, gold export, document-level splitting, and the quality-gate command. Those commands already exist and exit 2.
