# Marathi OCR evaluation pipeline (canonical / frozen)

This document freezes the production evaluation path after the whole-line vs
piecewise vs boundary forensics work. It is the contract for comparing future
checkpoints on the same 100-image validation set.

## Non-negotiables

- Do **not** modify `validation.jsonl` / `expected_text`.
- Do **not** use GT for segmentation, joining, decoding, or post-correction.
- Do **not** change primary STRICT CER/WER definitions to improve a score.
- Do **not** enable piece overlap in the canonical path (forensics showed
  overlap raised content-no-WS CER on this frozen set).
- Do **not** modify historical `scripts/eval_*.py` / `dump_failures.py`.

## Canonical flow

```text
ORIGINAL IMAGE
      │
      ▼
IMAGE VALIDATION          (path resolve, readable image)
      │
      ▼
HIGH-RES PIECE SEGMENTATION
      │
      ▼
WHITESPACE-AWARE CUTS     (image ink projection only; no GT)
      │
      ▼
NO OVERLAP                (piece_overlap_px = 0)
      │
      ▼
EACH PIECE
      ├── checkpoint img_size preprocess
      ├── same tokenizer / charset
      └── same decoder / EOS / max_label_length
      │
      ▼
OCR PIECES
      │
      ▼
DETERMINISTIC JOIN        (boundary_aware: space only on whitespace_gap)
      │
      ▼
FINAL TEXT
      │
      ▼
STRICT EVALUATION         (primary)
      │
      ├── CER / WER / Exact / Char Acc / Word Acc
      ├── S / D / I
      ├── numerals / punctuation / conjuncts / matras / reference-IDs
      └── length strata + capability subsets (analytical)
      │
      ▼
DIAGNOSTICS (secondary)
      ├── whitespace_normalized view
      ├── content_without_whitespace view
      └── boundary / piece metadata
```

## Frozen inference settings

| Setting | Canonical value |
| --- | --- |
| `mode` | `piecewise` |
| `piece_target_aspect` | `3.0` |
| `join_mode` | `boundary_aware` |
| `piece_overlap_px` | `0` |
| `tta` | `false` |
| `batch_size` | `1` |

Source of truth: `scripts/evaluation/config/canonical_pipeline.yaml`

Pipeline version id: `marathi_piecewise_canonical_v1`

## How to run

### Production entry point (forces canonical settings)

```bash
python -m scripts.run_canonical_eval \
  --model-path checkpoints/data/SOME.ckpt \
  --dataset "/Users/suraj.nagre/Desktop/IndicPhotoOCR/dataset " \
  --output results/some_ckpt_canonical \
  --device auto
```

Optional: `--check-reference-baseline` prints comparison to the documented
Run-D numbers (informative only).

### Equivalent evaluate.py invocation

```bash
python -m scripts.evaluate \
  --canonical \
  --model-path checkpoints/data/SOME.ckpt \
  --dataset "/Users/suraj.nagre/Desktop/IndicPhotoOCR/dataset " \
  --output results/some_ckpt_canonical \
  --device auto \
  --batch-size 1
```

`--canonical` forces the frozen inference settings even if other CLI flags differ.

### Whole-line ablation (not canonical)

```bash
python -m scripts.evaluate \
  --mode whole \
  --model-path ... \
  --dataset ".../dataset " \
  --output results/some_ckpt_whole
```

## Primary vs diagnostic metrics

| View | Role |
| --- | --- |
| STRICT identity | **Primary** benchmark |
| wer_strict | Conventional whitespace-token WER (same as primary `wer`) |
| wer_normalized | NFC + trim + collapse WS, then whitespace-token WER |
| wer_content_aware | OCR diagnostic: content mistakes vs whitespace-boundary-only |
| lexical_wer | NFC/WS-normalized; punctuation characters are separate tokens |
| whitespace_normalized (NFC + collapse/trim WS) | Diagnostic CER only |
| content_without_whitespace | Diagnostic only (glyph residual) |

`wer_content_aware` is **not** NIST-style WER. It does not erase punctuation,
convert Marathi numerals, spell-correct, or rewrite predictions.

Never promote diagnostic views to primary for model ranking without an explicit
version bump and written rationale.

## Reference measurements (Run D, frozen 100)

Documented in the YAML `reference_baseline` block:

| Configuration | Strict CER | Content-no-WS CER |
| --- | ---: | ---: |
| Whole-line | 0.8233 | — |
| Piecewise AR=3, overlap 0 | 0.2456 | 0.1556 |

These numbers characterize the current checkpoint; they are **not** training
targets and must not drive GT-aware postprocessing.

## Experimental knobs (not canonical)

Available on `scripts.evaluate` for research only:

- `--piece-target-aspect` other than 3.0
- `--piece-overlap` > 0
- `--join-mode concat_no_separator`
- `--mode whole`

Forensic / sweep scripts under `scripts/analyze_*.py`,
`scripts/build_aspect_geometry_comparison.py`,
`scripts/run_boundary_forensics.py` are **not** part of the production path.

## Outputs

Each run writes:

- `summary.json` / `summary.md` — metrics + `pipeline` fingerprint
- `per_sample.jsonl` — predictions, piece metadata, diagnostics
- `errors.jsonl` — non-exact samples
- optional plots with `--plot`

## Comparability rule for future checkpoints

Two checkpoint evaluations are comparable only if they share:

1. the same frozen 100-image dataset + GT
2. the same `pipeline_version` / inference settings fingerprint
3. the same `metric_version` / `normalization_version`

If any of those change, bump the version string and document why.
