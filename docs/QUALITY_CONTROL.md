# Quality control

The dataset is judged by correct labels, document diversity, structural diversity, visual diversity, length diversity, provenance, leakage, and label noise. A smaller set of real verified lines is the goal. Synthetic Marathi sentences, invented government wording, and number-swapping to fake uniqueness are out of scope. Image augmentation belongs to a later training repository.

## Audits

Later audit commands flag and do not repair:

- empty, whitespace-only, invalid, or suspicious Unicode
- U+FFFD and accidental English substitutions where Marathi is visible
- duplicate sample ids, duplicate image paths, and the same image with two gold strings
- crop and text mismatches, including a partial crop labeled with the full line
- impossible box dimensions and labels that cannot fit the crop
- SHA-256 duplicates, perceptual near-duplicates, and normalized text duplicates
- the same parent line, document, or source PDF in more than one split
- template repetition, department dominance, document dominance, and extreme partial-crop ratios
- width, height, aspect ratio, DPI, contrast, sharpness, and crop-type counts

Structure measures include exact text, number-normalized text, lexical shape, punctuation shape, character and word n-grams, length, word count, numerals, dates, reference numbers, amounts, and conjunct-heavy lines.

## Quality gate

The gate fails when any of these are true:

- a document or parent line is in more than one split
- invalid Unicode, missing images, or missing labels
- duplicate sample ids, or one image with conflicting gold text
- a candidate, auto-approved candidate, or rejected row is inside gold
- a crop/text mismatch is detected
- source provenance is missing

The gate warns when structure repetition is high, one department or document dominates, the partial-crop ratio is extreme, long lines are scarce, or OCR disagreement is unusually high. Warning thresholds are in `configs/dataset.yaml`. Fail switches cannot be turned off in configuration.

Every gold build will also write `reports/dataset_report.json` and `reports/dataset_report.md`. Those writers are a later phase. The pure checks for leakage, duplicate ids, conflicting labels, SHA-256 collisions, and missing provenance already live in `provenance.py`.

## Pilot

Before a large run, process 5–10 documents and about 500–1,000 crops of more than one document type. Inspect renders, boxes, full crops, partial crops, OCR candidates, the review UI, labels, provenance, and the split. Scale up only after that pilot.
