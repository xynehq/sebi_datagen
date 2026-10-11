"""Native Indic + ASCII numeral evaluation metrics.

Defaults remain Marathi (Devanagari) digits for backwards compatibility with
``marathi_ocr_eval_v2``. Pass ``native_digits`` / ``script_name`` for Gujarati
or other scripts. ASCII↔native script swaps are never treated as correct.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .core_metrics import align_edit_ops, alignment_pairs, levenshtein

MARATHI_DIGITS = "०१२३४५६७८९"
GUJARATI_DIGITS = "૦૧૨૩૪૫૬૭૮૯"
ASCII_DIGITS = "0123456789"

# Contiguous runs of native Indic and/or ASCII digits
_NUMERIC_TOKEN_RE = re.compile(r"[०-९૦-૯0-9]+")


@dataclass
class NumeralSampleResult:
    has_numeric: bool
    # Primary native-digit accuracy (script of ``native_digits``).
    native_numeral_character_accuracy: Optional[float]
    # Back-compat alias used by Marathi reports / tests.
    marathi_numeral_character_accuracy: Optional[float]
    # Populated when evaluating Gujarati digits (else None).
    gujarati_numeral_character_accuracy: Optional[float]
    numeric_token_exact_accuracy: Optional[float]
    numeric_token_cer: Optional[float]
    numeric_exact_match: Optional[bool]
    numeral_script_error: bool
    native_to_ascii: int = 0
    ascii_to_native: int = 0
    # Back-compat aliases
    marathi_to_ascii: int = 0
    ascii_to_marathi: int = 0
    digit_substitutions: int = 0
    digit_deletions: int = 0
    digit_insertions: int = 0
    # Corpus-level aggregation inputs (native GT digit instances).
    native_correct: int = 0
    native_total: int = 0
    gt_tokens: List[str] = field(default_factory=list)
    pred_tokens: List[str] = field(default_factory=list)
    script_name: str = "marathi"


def extract_numeric_tokens(text: str) -> List[str]:
    return _NUMERIC_TOKEN_RE.findall(text)


def evaluate_numerals(
    expected: str,
    prediction: str,
    native_digits: str = MARATHI_DIGITS,
    script_name: str = "marathi",
) -> NumeralSampleResult:
    native_digits = str(native_digits)
    to_ascii = str.maketrans(native_digits, ASCII_DIGITS)
    to_native = str.maketrans(ASCII_DIGITS, native_digits)
    native_set = set(native_digits)
    ascii_set = set(ASCII_DIGITS)

    def _is_digit(ch: str) -> bool:
        return ch in native_set or ch in ascii_set

    def _script_of(ch: str) -> Optional[str]:
        if ch in native_set:
            return "native"
        if ch in ascii_set:
            return "ascii"
        return None

    gt_tokens = extract_numeric_tokens(expected)
    pred_tokens = extract_numeric_tokens(prediction)
    has_numeric = bool(gt_tokens) or bool(pred_tokens)

    empty = NumeralSampleResult(
        has_numeric=False,
        native_numeral_character_accuracy=None,
        marathi_numeral_character_accuracy=None,
        gujarati_numeral_character_accuracy=None,
        numeric_token_exact_accuracy=None,
        numeric_token_cer=None,
        numeric_exact_match=None,
        numeral_script_error=False,
        gt_tokens=gt_tokens,
        pred_tokens=pred_tokens,
        script_name=script_name,
    )
    if not has_numeric:
        return empty

    pairs = alignment_pairs(expected, prediction)
    nat_correct = nat_total = 0
    n2a = a2n = 0
    d_sub = d_del = d_ins = 0
    script_error = False

    for op, rc, hc in pairs:
        if op == "eq":
            if rc in native_set:
                nat_total += 1
                nat_correct += 1
        elif op == "sub":
            if _is_digit(rc) or _is_digit(hc):
                d_sub += 1
            if rc in native_set:
                nat_total += 1
                rs, hs = _script_of(rc), _script_of(hc)
                if rs == "native" and hs == "ascii":
                    n2a += 1
                    script_error = True
                elif rs == "native" and hs == "native" and rc == hc:
                    nat_correct += 1
            if rc in ascii_set and hc in native_set:
                a2n += 1
                script_error = True
        elif op == "del":
            if _is_digit(rc):
                d_del += 1
            if rc in native_set:
                nat_total += 1
        elif op == "ins":
            if _is_digit(hc):
                d_ins += 1

    for gt_tok, pred_tok in zip(gt_tokens, pred_tokens):
        gt_is_nat = all(c in native_set for c in gt_tok)
        pred_is_asc = all(c in ascii_set for c in pred_tok)
        pred_is_nat = all(c in native_set for c in pred_tok)
        gt_is_asc = all(c in ascii_set for c in gt_tok)
        if gt_is_nat and pred_is_asc:
            if gt_tok.translate(to_ascii) == pred_tok:
                script_error = True
                n2a += 1
        if gt_is_asc and pred_is_nat:
            if gt_tok.translate(to_native) == pred_tok:
                script_error = True
                a2n += 1

    n_tok = len(gt_tokens)
    if n_tok == 0:
        tok_exact_acc = 0.0 if pred_tokens else 1.0
        tok_cer = 0.0 if not pred_tokens else 1.0
        numeric_exact = len(pred_tokens) == 0
    else:
        matched = 0
        total_ed = 0
        total_len = 0
        for i, gt_tok in enumerate(gt_tokens):
            pred_tok = pred_tokens[i] if i < len(pred_tokens) else ""
            if gt_tok == pred_tok:
                matched += 1
            total_ed += levenshtein(gt_tok, pred_tok)
            total_len += len(gt_tok)
        tok_exact_acc = matched / n_tok
        tok_cer = (total_ed / total_len) if total_len > 0 else 0.0
        numeric_exact = (gt_tokens == pred_tokens[:n_tok] and len(pred_tokens) == n_tok)

    nat_acc = (nat_correct / nat_total) if nat_total > 0 else None
    is_guj = (script_name or "").lower() == "gujarati" or native_digits == GUJARATI_DIGITS
    is_mar = (script_name or "").lower() == "marathi" or (
        native_digits == MARATHI_DIGITS and not is_guj
    )

    return NumeralSampleResult(
        has_numeric=True,
        native_numeral_character_accuracy=nat_acc,
        marathi_numeral_character_accuracy=nat_acc if is_mar else None,
        gujarati_numeral_character_accuracy=nat_acc if is_guj else None,
        numeric_token_exact_accuracy=tok_exact_acc,
        numeric_token_cer=tok_cer,
        numeric_exact_match=numeric_exact,
        numeral_script_error=script_error,
        native_to_ascii=n2a,
        ascii_to_native=a2n,
        marathi_to_ascii=n2a,
        ascii_to_marathi=a2n,
        digit_substitutions=d_sub,
        digit_deletions=d_del,
        digit_insertions=d_ins,
        native_correct=nat_correct,
        native_total=nat_total,
        gt_tokens=gt_tokens,
        pred_tokens=pred_tokens,
        script_name=script_name,
    )


@dataclass
class NumeralCorpusMetrics:
    samples_with_numeric: int
    native_numeral_character_accuracy: Optional[float]
    marathi_numeral_character_accuracy: Optional[float]
    gujarati_numeral_character_accuracy: Optional[float]
    numeric_token_exact_accuracy: Optional[float]
    numeric_token_cer: Optional[float]
    numeric_exact_match_rate: Optional[float]
    numeral_script_error_count: int
    native_to_ascii: int
    ascii_to_native: int
    marathi_to_ascii: int
    ascii_to_marathi: int
    # Corpus denominators (native GT digit instances)
    native_correct: int = 0
    native_total: int = 0


def aggregate_numerals(results: Sequence[NumeralSampleResult]) -> NumeralCorpusMetrics:
    """Aggregate numeral metrics.

    Native digit accuracy is **corpus-level**:
    ``Σ native_correct / Σ native_total`` across samples (not mean of sample rates).
    """
    usable = [r for r in results if r.has_numeric]
    if not usable:
        return NumeralCorpusMetrics(
            samples_with_numeric=0,
            native_numeral_character_accuracy=None,
            marathi_numeral_character_accuracy=None,
            gujarati_numeral_character_accuracy=None,
            numeric_token_exact_accuracy=None,
            numeric_token_cer=None,
            numeric_exact_match_rate=None,
            numeral_script_error_count=0,
            native_to_ascii=0,
            ascii_to_native=0,
            marathi_to_ascii=0,
            ascii_to_marathi=0,
            native_correct=0,
            native_total=0,
        )

    total_tok_ed = 0
    total_tok_len = 0
    tok_exact = 0
    tok_total = 0
    num_exact = 0
    script_err = 0
    n2a = a2n = 0
    nat_correct = 0
    nat_total = 0
    script_names = set()

    for r in usable:
        script_names.add(r.script_name)
        nat_correct += int(r.native_correct)
        nat_total += int(r.native_total)
        if r.numeric_exact_match:
            num_exact += 1
        if r.numeral_script_error:
            script_err += 1
        n2a += r.native_to_ascii
        a2n += r.ascii_to_native
        for gt_tok, pred_tok in zip(r.gt_tokens, r.pred_tokens):
            tok_total += 1
            if gt_tok == pred_tok:
                tok_exact += 1
            total_tok_ed += levenshtein(gt_tok, pred_tok)
            total_tok_len += len(gt_tok)
        if len(r.gt_tokens) > len(r.pred_tokens):
            for gt_tok in r.gt_tokens[len(r.pred_tokens) :]:
                tok_total += 1
                total_tok_ed += len(gt_tok)
                total_tok_len += len(gt_tok)

    nat_acc = (nat_correct / nat_total) if nat_total > 0 else None
    is_guj = "gujarati" in script_names
    is_mar = "marathi" in script_names or not is_guj
    tok_acc = (tok_exact / tok_total) if tok_total else None
    tok_cer = (total_tok_ed / total_tok_len) if total_tok_len else None

    return NumeralCorpusMetrics(
        samples_with_numeric=len(usable),
        native_numeral_character_accuracy=nat_acc,
        marathi_numeral_character_accuracy=nat_acc if is_mar else None,
        gujarati_numeral_character_accuracy=nat_acc if is_guj else None,
        numeric_token_exact_accuracy=tok_acc,
        numeric_token_cer=tok_cer,
        numeric_exact_match_rate=num_exact / len(usable),
        numeral_script_error_count=script_err,
        native_to_ascii=n2a,
        ascii_to_native=a2n,
        marathi_to_ascii=n2a,
        ascii_to_marathi=a2n,
        native_correct=nat_correct,
        native_total=nat_total,
    )
