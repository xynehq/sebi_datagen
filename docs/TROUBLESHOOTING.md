# Troubleshooting

## A stage exits with code 2

The command is registered and the stage is not implemented yet. `python3 -m indic_ocr_pipeline.stages --list-stages` shows the phase number. Status is the only stage that runs in Phase 1.

## Configuration fails to load

The loader rejects unknown YAML keys, a DPI outside 150/200/250/300/400, split ratios that do not sum to 1, score weights that do not sum to 1, gold statuses other than `human_verified` and `human_corrected`, and `allow_network: true`. Fix the YAML instead of bypassing the check.

## Manifest conflict

Appending an id that already exists with different content raises `ManifestConflictError` and leaves the file unchanged. That usually means two different outputs were produced for one id. Inspect both payloads. Do not delete the manifest to force a rewrite.

A `complete` row cannot be edited. To retry a `failed` row, a later stage must move it back to `running` under the allowed status transitions.

## Duplicate source PDF

Ingestion will key documents by SHA-256. The same bytes keep the same `MHGR_######` id. Two ids for one hash, or one id for two hashes, is a manifest error.

## OCR confidence is null

Leave it null. Do not invent a probability. Tesseract confidence, when the adapter exists, is stored as a signal and is not treated as the chance the line is correct.

## Review text looks normalized

Gold text is NFC with surrounding whitespace removed. Internal spaces, punctuation, and digits remain. If a comparison shows a single-spaced string and the manifest shows the original spacing, the single-spaced string is the comparison key only.

## Tests

```bash
python3 -m pytest
```

Unit tests use temporary files and in-memory records. They do not need a GPU, a PDF, or a network connection.
