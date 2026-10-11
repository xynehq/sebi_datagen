# Split policy

Train, validation, and test are assigned by source document. The shipped ratios are 0.8, 0.1, and 0.1, with seed 3407. Every crop from a document, including partial crops of the same line, stays in that document's split.

This split is leakage:

```text
document MHGR_000001, line 10, full crop  → train
document MHGR_000001, line 10, left crop → validation
```

`assert_no_split_leakage` fails when one `document_id` or one `parent_line_id` appears in two splits. The later leakage audit command must exit non-zero in that case. The quality gate also fails when the same source SHA-256 appears in two splits, when sample ids repeat, or when one image path has two gold strings.

Rows with no split yet are ignored by the leakage scan so an unsplit gold file can be checked before the split stage runs. After the split stage, every gold row needs a split.

The split report should count documents, pages, lines, crops, and departments in each split. Department balance is a warning in `configs/dataset.yaml`, not a reason to move individual crops between splits.

Near-duplicate detection (perceptual hash, normalized text, repeated templates) is an audit. It informs the report. It does not move a document into another split by itself, and it does not rewrite numbers to manufacture unique sentences.
