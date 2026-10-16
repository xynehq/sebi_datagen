# Data model

Manifests are UTF-8 JSONL. Marathi text is stored as Unicode, not as escaped code points. Callers validate a record with its schema before appending it.

## Identifiers

Document IDs look like `MHGR_000001`. The number is allocated once per SHA-256 and is never reused. Filename order does not choose the id. A gap in the manifest is kept; the next new file receives `max(existing) + 1`.

Every other id is a pure function of its parents:

| Entity | Form | Example |
| --- | --- | --- |
| Page | `{document}_P{page:04d}` | `MHGR_000001_P0003` |
| Region | `{page}_R{index:04d}` | `MHGR_000001_P0003_R0001` |
| Line | `{page}_L{index:04d}` | `MHGR_000001_P0003_L0012` |
| Crop | `{line}_C_{type}_{index:02d}` | `MHGR_000001_P0003_L0012_C_left_01` |
| Sample | `SAMPLE_{crop}` | `SAMPLE_MHGR_000001_P0003_L0012_C_full_01` |
| Prediction | `{crop}_OCR_{backend}` | `..._C_full_01_OCR_tesseract` |
| Candidate | `{crop}_CAND` | |
| Review | `{crop}_REV` | |
| Verified | `{crop}_VER` | |

Page, region, line, and crop indexes are 1-based. The same crop always receives the same sample id, so ids do not shift when a run is resumed.

Boxes are `[x0, y0, x1, y1]` in rendered-page pixels. `x1` and `y1` are exclusive. A crop box must lie inside its parent line box and inside the page.

## Manifests

| File | Id field | Record |
| --- | --- | --- |
| `data/output/documents.jsonl` | `document_id` | `DocumentRecord` |
| `data/output/pages.jsonl` | `page_id` | `PageRecord` |
| `data/output/regions.jsonl` | `region_id` | `RegionRecord` |
| `data/output/lines.jsonl` | `line_id` | `RegionRecord` with `region_type` `line` |
| `data/output/crops.jsonl` | `crop_id` | `CropRecord` |
| `data/output/predictions.jsonl` | `prediction_id` | `PredictionRecord` |
| `data/output/candidates.jsonl` | `candidate_id` | `CandidateRecord` |
| `data/output/review.jsonl` | `review_id` | `ReviewRecord` |
| `data/output/verified.jsonl` | `verified_id` | `VerifiedRecord` |
| `data/output/gold.jsonl` | `sample_id` | `GoldRecord` |

`document_date` is the document's own date. `ingested_at`, `rendered_at`, `reviewed_at`, and `verification_timestamp` are ISO-8601 pipeline times.

## Status

Stage status moves `pending → running → complete`, or `running → failed`, or `running → skipped`. `failed` and `skipped` may return to `running`. A `complete` row is frozen. Identity fields (`document_id`, hashes, boxes, `gold_text`, OCR `text`, `final_text`) cannot change once they have a value. `split` may be filled in once, and that assignment cannot be moved to another split.

## Verification status

| Status | Manifest | Gold |
| --- | --- | --- |
| `candidate` | candidates | No |
| `auto_approved_candidate` | candidates, tier 3 | No |
| `human_verified` | verified, then gold | Yes |
| `human_corrected` | verified, then gold | Yes |
| `rejected` | verified | No |

`CandidateRecord` rejects human statuses. `GoldRecord` rejects everything except `human_verified` and `human_corrected`, and it requires tier `tier_5_gold`.

A crop record has geometry and no transcription field. Prediction text is the backend's raw proposal plus a stored `text` field. `confidence` is null when the backend does not provide one.

`consensus_text` on a gold row is audit context. `gold_text` is the label.

## Provenance required on every gold row

`sample_id`, `image_path`, `gold_text`, `document_id`, `page_id`, `page_number`, `region_id`, `parent_line_id`, `crop_id`, `crop_type`, `bbox`, `verification_status`, `source_sha256`, `verification_timestamp`, `original_candidate_texts`.

`original_candidate_texts` keeps the OCR strings after a correction.
