# Review policy

The reviewer sees the page context, the crop, the crop box, each OCR candidate, the disagreement summary, the proposed consensus, the rank score, and an editable text box. Mouse buttons and these keys are the planned controls:

| Key | Decision |
| --- | --- |
| A | Accept the selected candidate |
| E | Save the edited transcription |
| R | Reject the crop |
| B | Bad boundary |
| U | Uncertain |

Accept stores `final_text` equal to the selected candidate and becomes `human_verified`. Edit stores the typed string and becomes `human_corrected`. Reject, bad boundary, bad image, and non-text store no gold text and become `rejected`. Uncertain stores no gold text and stays out of gold so it can be reviewed again.

The interface must allow text that none of the OCR systems produced.

## Priority

Review risky rows first: disagreement, low rank score, numeral mismatch, suspicious Unicode, bad boundaries, low image quality, empty or very short output, punctuation disagreement, and conjunct or matra disagreement.

Rows where the backends agree, the crop is clean, and the rank score is high are lower priority. They remain `candidate`. `auto_approved_candidate` is allowed by the schema and is still not gold. The first ranking pass does not assign it: three-way agreement can repeat a shared mistake, so every row stays `candidate` until a person reviews it.

A random audit is also drawn from the high-confidence pool. `high_confidence_audit_count` applies the configured fraction and then the minimum and maximum caps. The shipped caps are 500 and 1000. A pool smaller than the minimum is sampled in full. The audit estimates how often the high-confidence tier is actually wrong. Until that error rate is known and acceptable, high-confidence rows stay out of gold.

## Scores

`confidence_score` and `risk_score` are weighted ranks in `[0, 1]`. The weights are in `configs/scoring.yaml` and sum to 1. Each score stores `reasons` from the `ScoreReason` list. The score is not a calibrated probability of correctness. Tesseract word confidence is one input signal, and it is stored as reported. Missing model confidence stays null.

The HTML review package and the importer are phases 12–14. Review results will be JSONL so they can be imported without the rest of the pipeline running.
