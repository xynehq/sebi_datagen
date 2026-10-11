# Label policy

The label is the Unicode text a person confirms while looking at the crop image.

## What the person checks

Every visible character, including matras, conjuncts, punctuation, Marathi digits, Arabic digits, reference numbers, dates, amounts, brackets, hyphens, quotes, and the spacing required by this policy.

## Storage form

Before a gold or verified string is stored, the import step may apply storage normalization explicitly:

1. Unicode NFC.
2. Remove whitespace at the two ends.

The schema refuses a gold string that is not already in that form, so the record constructor does not rewrite it. Punctuation and numerals stay. Internal spaces stay, including a double space when that is what the reviewer kept. U+FFFD cannot be stored as gold. Empty and whitespace-only strings cannot be stored as gold.

## Comparison form

OCR agreement may compare strings after NFC, end-stripping, and collapsing internal whitespace. That comparison key is not written back as the label. The original candidate strings stay in the prediction and candidate records.

Extracted PDF text may be stored on the page record as `pdf_text_layer`. Its role field is fixed to `weak_diagnostic_not_ground_truth`. Government PDFs often have a broken Marathi text layer. The rendered image remains the visual source.

## Partial crops

`assert_partial_label_matches_crop` rejects a partial crop whose gold text is identical to the full parent line's gold text. The visible substring has to be confirmed by the reviewer. The pipeline does not invent that substring from character positions.

When the full line has no gold text yet, a partial crop can still be verified on its own. The reviewer types or accepts the text that is inside that crop.

Full-line crops use `crop_type: full`. Partial types are `left`, `right`, `middle`, and `random`. `multi_line` stays disabled unless `cropping.multi_line` is turned on.

## Corrections

`human_verified` means the reviewer accepted a candidate that matches the image. `human_corrected` means the reviewer typed the text, which may match none of the OCR outputs. Both keep `original_candidate_texts`.

Suspicious labels are flagged by later audits. Those audits do not rewrite the text.
