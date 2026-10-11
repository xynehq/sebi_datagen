"""WER diagnostic views for Marathi OCR evaluation.

Reports three clearly named metrics (plus lexical WER):

- wer_strict: conventional whitespace-token WER (unchanged definition)
- wer_normalized: WER after NFC + trim + whitespace-run collapse only
- wer_content_aware: OCR diagnostic separating content mistakes from
  whitespace-boundary-only differences (NOT NIST-style WER)
- lexical_wer: punctuation-split token WER after NFC/whitespace normalize

None of these erase punctuation, convert Marathi numerals, spell-correct,
or use GT to rewrite predictions.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from .core_metrics import (
    _sequence_edit,
    alignment_pairs,
    sample_core_metrics,
    tokenize_words,
)
from .normalization import content_without_whitespace, diagnostic_nfc_whitespace
from .punctuation import load_punctuation_set
from .whitespace_diagnostics import _word_boundary_events


@dataclass(frozen=True)
class SampleWerViews:
    wer_strict: float
    wer_normalized: float
    wer_content_aware: float
    lexical_wer: float
    content_match: bool
    content_error: bool
    whitespace_boundary_error: bool
    word_boundary_merge_count: int
    word_boundary_split_count: int
    content_word_edits: int
    content_reference_words: int
    lexical_edits: int
    lexical_reference_tokens: int
    strict_reference_words: int
    normalized_reference_words: int


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tokenize_lexical(text: str, punct: Optional[Set[str]] = None) -> List[str]:
    """NFC + whitespace-normalize, then split punctuation into own tokens.

    Tokenization (Unicode code-point iteration, not bytes):
    1. Apply diagnostic_nfc_whitespace (NFC, trim, collapse WS runs)
    2. Walk characters:
       - whitespace already collapsed to single spaces → split on space
       - each punctuation character (config set) is its own token
       - all other characters accumulate into a lexical word token

    Devanagari conjuncts/matras stay inside lexical words; only configured
    punctuation characters are separated. Numerals are never remapped.
    """
    punct = punct or load_punctuation_set()
    t = diagnostic_nfc_whitespace(text)
    tokens: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in t:
        if ch.isspace():
            flush()
        elif ch in punct:
            flush()
            tokens.append(ch)
        else:
            buf.append(ch)
    flush()
    return tokens


def _token_wer(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> Tuple[float, int, int]:
    dist, _s, _d, _i = _sequence_edit(ref_tokens, hyp_tokens)
    n = len(ref_tokens)
    wer = (dist / n) if n > 0 else (0.0 if len(hyp_tokens) == 0 else 1.0)
    return wer, dist, n


def _content_spans(nfc_text: str) -> List[Tuple[int, int]]:
    """Map GT whitespace-words to spans in content_without_whitespace index space."""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for word in tokenize_words(nfc_text):
        # word is already without internal spaces; length in content space
        # equals Unicode code-point length of the word string
        start = pos
        end = pos + len(word)
        spans.append((start, end))
        pos = end
    return spans


def _content_aware_word_edits(gt: str, pred: str) -> Tuple[int, int, bool, bool]:
    """Return (edits, n_ref_words, content_match, content_error).

    Procedure:
    A. NFC both strings (keep originals for boundary analysis elsewhere).
    B. Build whitespace-stripped content strings.
    C. Align content strings at character level.
    D. If contents identical → zero content-word edits (boundary-only may differ).
    E. Else mark each GT word whose content span has any non-eq alignment op,
       and count contiguous insertion runs as word insertions.
    """
    gt_n = _nfc(gt)
    pred_n = _nfc(pred)
    c_gt = content_without_whitespace(gt_n)
    c_pred = content_without_whitespace(pred_n)
    spans = _content_spans(gt_n)
    n_ref = len(spans)

    content_match = c_gt == c_pred
    if content_match:
        return 0, n_ref, True, False

    pairs = alignment_pairs(c_gt, c_pred)
    bad_positions: Set[int] = set()
    gt_pos = 0
    insert_runs = 0
    in_insert = False
    for op, _rc, _hc in pairs:
        if op == "eq":
            gt_pos += 1
            in_insert = False
        elif op == "sub":
            bad_positions.add(gt_pos)
            gt_pos += 1
            in_insert = False
        elif op == "del":
            bad_positions.add(gt_pos)
            gt_pos += 1
            in_insert = False
        elif op == "ins":
            if not in_insert:
                insert_runs += 1
                in_insert = True
        else:
            in_insert = False

    bad_words = 0
    for start, end in spans:
        if any(p in bad_positions for p in range(start, end)):
            bad_words += 1

    edits = bad_words + insert_runs
    return edits, n_ref, False, True


def _whitespace_boundary_error(gt: str, pred: str, content_match: bool) -> bool:
    """True when underlying content matches but whitespace boundaries differ.

    Uses NFC+collapsed whitespace strings: if those still differ while content
    matches, boundaries (or non-collapsible WS differences already collapsed)
    differ — e.g. merge/split of words.
    """
    if not content_match:
        return False
    return diagnostic_nfc_whitespace(gt) != diagnostic_nfc_whitespace(pred)


def evaluate_wer_views(
    expected: str,
    prediction: str,
    punct: Optional[Set[str]] = None,
) -> SampleWerViews:
    punct = punct or load_punctuation_set()

    # 1) Strict — conventional whitespace-token WER (identity strings)
    strict_m = sample_core_metrics(expected, prediction)
    wer_strict = strict_m.wer
    strict_ref_words = len(tokenize_words(expected))

    # 2) Normalized — NFC + trim + collapse WS only, then whitespace tokens
    g_n = diagnostic_nfc_whitespace(expected)
    p_n = diagnostic_nfc_whitespace(prediction)
    wer_normalized, _nd, n_norm_words = _token_wer(
        tokenize_words(g_n), tokenize_words(p_n)
    )

    # 3) Content-aware diagnostic
    edits, n_ref, content_match, content_error = _content_aware_word_edits(
        expected, prediction
    )
    wer_content_aware = (
        (edits / n_ref) if n_ref > 0 else (0.0 if edits == 0 else 1.0)
    )
    boundary_err = _whitespace_boundary_error(expected, prediction, content_match)

    # Boundary merge/split heuristics (existing)
    merges, splits = _word_boundary_events(_nfc(expected), _nfc(prediction))

    # 4) Lexical WER — punctuation as separate tokens
    lex_ref = tokenize_lexical(expected, punct)
    lex_hyp = tokenize_lexical(prediction, punct)
    lexical_wer, lex_edits, lex_n = _token_wer(lex_ref, lex_hyp)

    return SampleWerViews(
        wer_strict=wer_strict,
        wer_normalized=wer_normalized,
        wer_content_aware=wer_content_aware,
        lexical_wer=lexical_wer,
        content_match=content_match,
        content_error=content_error,
        whitespace_boundary_error=boundary_err,
        word_boundary_merge_count=merges,
        word_boundary_split_count=splits,
        content_word_edits=edits,
        content_reference_words=n_ref,
        lexical_edits=lex_edits,
        lexical_reference_tokens=lex_n,
        strict_reference_words=strict_ref_words,
        normalized_reference_words=n_norm_words,
    )


def aggregate_wer_views(
    expecteds: Sequence[str],
    predictions: Sequence[str],
    punct: Optional[Set[str]] = None,
) -> dict:
    punct = punct or load_punctuation_set()
    views = [
        evaluate_wer_views(g, p, punct) for g, p in zip(expecteds, predictions)
    ]
    n = len(views) or 1

    # Strict / normalized corpus WER via token edit totals
    strict_ed = 0
    strict_ref = 0
    norm_ed = 0
    norm_ref = 0
    for g, p in zip(expecteds, predictions):
        sm = sample_core_metrics(g, p)
        strict_ed += sm.word_substitutions + sm.word_deletions + sm.word_insertions
        strict_ref += len(tokenize_words(g))
        gn, pn = diagnostic_nfc_whitespace(g), diagnostic_nfc_whitespace(p)
        d, _s, _dlt, _ins = _sequence_edit(tokenize_words(gn), tokenize_words(pn))
        norm_ed += d
        norm_ref += len(tokenize_words(gn))

    content_ed = sum(v.content_word_edits for v in views)
    content_ref = sum(v.content_reference_words for v in views)
    lex_ed = sum(v.lexical_edits for v in views)
    lex_ref = sum(v.lexical_reference_tokens for v in views)

    return {
        "wer_strict": (strict_ed / strict_ref) if strict_ref else 0.0,
        "wer_normalized": (norm_ed / norm_ref) if norm_ref else 0.0,
        "wer_content_aware": (content_ed / content_ref) if content_ref else 0.0,
        "lexical_wer": (lex_ed / lex_ref) if lex_ref else 0.0,
        "content_match_rate": sum(v.content_match for v in views) / n,
        "content_match_count": sum(v.content_match for v in views),
        "content_error_rate": sum(v.content_error for v in views) / n,
        "content_error_count": sum(v.content_error for v in views),
        "whitespace_boundary_error_rate": sum(
            v.whitespace_boundary_error for v in views
        )
        / n,
        "whitespace_boundary_error_count": sum(
            v.whitespace_boundary_error for v in views
        ),
        "word_boundary_merge_count": sum(v.word_boundary_merge_count for v in views),
        "word_boundary_split_count": sum(v.word_boundary_split_count for v in views),
        "total_content_word_edits": content_ed,
        "total_content_reference_words": content_ref,
        "total_lexical_edits": lex_ed,
        "total_lexical_reference_tokens": lex_ref,
        "total_strict_word_edits": strict_ed,
        "total_strict_reference_words": strict_ref,
        "total_normalized_word_edits": norm_ed,
        "total_normalized_reference_words": norm_ref,
        "note": (
            "wer_strict is conventional whitespace-token WER. "
            "wer_normalized applies NFC+trim+collapse-WS only. "
            "wer_content_aware is an OCR diagnostic that separates content "
            "mistakes from whitespace-boundary-only differences; it is not "
            "NIST-style WER. lexical_wer splits punctuation into tokens."
        ),
    }
