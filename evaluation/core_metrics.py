"""Core OCR metrics: edit distance with S/D/I, CER, WER, exact match."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class EditOps:
    distance: int
    substitutions: int
    deletions: int
    insertions: int


def levenshtein(a: str, b: str) -> int:
    return align_edit_ops(a, b).distance


def align_edit_ops(ref: str, hyp: str) -> EditOps:
    """Wagner–Fischer with backtrace → substitutions/deletions/insertions."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return EditOps(distance=m, substitutions=0, deletions=0, insertions=m)
    if m == 0:
        return EditOps(distance=n, substitutions=0, deletions=n, insertions=0)

    # dp[i][j] = (cost, prev_i, prev_j, op) op in {eq,sub,del,ins}
    dp_cost = [[0] * (m + 1) for _ in range(n + 1)]
    dp_prev = [[(0, 0, "eq")] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp_cost[i][0] = i
        dp_prev[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp_cost[0][j] = j
        dp_prev[0][j] = (0, j - 1, "ins")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost_eq = dp_cost[i - 1][j - 1]
                best = (cost_eq, i - 1, j - 1, "eq")
            else:
                best = (dp_cost[i - 1][j - 1] + 1, i - 1, j - 1, "sub")
            cost_del = dp_cost[i - 1][j] + 1
            if cost_del < best[0]:
                best = (cost_del, i - 1, j, "del")
            cost_ins = dp_cost[i][j - 1] + 1
            if cost_ins < best[0]:
                best = (cost_ins, i, j - 1, "ins")
            dp_cost[i][j] = best[0]
            dp_prev[i][j] = (best[1], best[2], best[3])

    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        pi, pj, op = dp_prev[i][j]
        if op == "sub":
            subs += 1
        elif op == "del":
            dels += 1
        elif op == "ins":
            ins += 1
        if i == 0 and j == 0:
            break
        i, j = pi, pj

    return EditOps(
        distance=dp_cost[n][m],
        substitutions=subs,
        deletions=dels,
        insertions=ins,
    )


def alignment_pairs(ref: str, hyp: str) -> List[Tuple[str, str, str]]:
    """Return list of (op, ref_char_or_empty, hyp_char_or_empty)."""
    n, m = len(ref), len(hyp)
    dp_cost = [[0] * (m + 1) for _ in range(n + 1)]
    dp_prev = [[(0, 0, "eq")] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp_cost[i][0] = i
        dp_prev[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp_cost[0][j] = j
        dp_prev[0][j] = (0, j - 1, "ins")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = (dp_cost[i - 1][j - 1], i - 1, j - 1, "eq")
            else:
                best = (dp_cost[i - 1][j - 1] + 1, i - 1, j - 1, "sub")
            if dp_cost[i - 1][j] + 1 < best[0]:
                best = (dp_cost[i - 1][j] + 1, i - 1, j, "del")
            if dp_cost[i][j - 1] + 1 < best[0]:
                best = (dp_cost[i][j - 1] + 1, i, j - 1, "ins")
            dp_cost[i][j] = best[0]
            dp_prev[i][j] = (best[1], best[2], best[3])

    i, j = n, m
    ops: List[Tuple[str, str, str]] = []
    while i > 0 or j > 0:
        pi, pj, op = dp_prev[i][j]
        if op == "eq" or op == "sub":
            ops.append((op, ref[i - 1], hyp[j - 1]))
        elif op == "del":
            ops.append((op, ref[i - 1], ""))
        else:
            ops.append((op, "", hyp[j - 1]))
        i, j = pi, pj
    ops.reverse()
    return ops


@dataclass(frozen=True)
class SampleCoreMetrics:
    exact_match: bool
    edit_distance: int
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int
    prediction_length: int
    cer: float
    wer: float
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    word_exact_accuracy: float
    length_ratio: float


def tokenize_words(text: str) -> List[str]:
    """Deterministic whitespace tokenization (crop-level WER)."""
    return text.split() if text.strip() else ([] if not text else text.split())


def sample_core_metrics(expected: str, prediction: str) -> SampleCoreMetrics:
    ops = align_edit_ops(expected, prediction)
    ref_len = len(expected)
    pred_len = len(prediction)
    cer = (ops.distance / ref_len) if ref_len > 0 else (0.0 if pred_len == 0 else 1.0)

    ref_words = tokenize_words(expected)
    hyp_words = tokenize_words(prediction)
    w_ed, w_s, w_d, w_i = _sequence_edit(ref_words, hyp_words)
    n_words = len(ref_words)
    wer = (w_ed / n_words) if n_words > 0 else (0.0 if len(hyp_words) == 0 else 1.0)
    correct_words = max(0, n_words - w_s - w_d)
    word_exact = (correct_words / n_words) if n_words > 0 else (1.0 if len(hyp_words) == 0 else 0.0)

    ratio = (pred_len / ref_len) if ref_len > 0 else (1.0 if pred_len == 0 else float("inf"))

    return SampleCoreMetrics(
        exact_match=(expected == prediction),
        edit_distance=ops.distance,
        substitutions=ops.substitutions,
        deletions=ops.deletions,
        insertions=ops.insertions,
        reference_length=ref_len,
        prediction_length=pred_len,
        cer=cer,
        wer=wer,
        word_substitutions=w_s,
        word_deletions=w_d,
        word_insertions=w_i,
        word_exact_accuracy=word_exact,
        length_ratio=ratio if ratio != float("inf") else -1.0,
    )


def _sequence_edit(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int, int]:
    """Edit distance on token sequences; returns (dist, S, D, I)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, 0, m
    if m == 0:
        return n, 0, n, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    prev = [[(0, 0, "eq")] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        prev[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp[0][j] = j
        prev[0][j] = (0, j - 1, "ins")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = (dp[i - 1][j - 1], i - 1, j - 1, "eq")
            else:
                best = (dp[i - 1][j - 1] + 1, i - 1, j - 1, "sub")
            if dp[i - 1][j] + 1 < best[0]:
                best = (dp[i - 1][j] + 1, i - 1, j, "del")
            if dp[i][j - 1] + 1 < best[0]:
                best = (dp[i][j - 1] + 1, i, j - 1, "ins")
            dp[i][j] = best[0]
            prev[i][j] = (best[1], best[2], best[3])
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        pi, pj, op = prev[i][j]
        if op == "sub":
            s += 1
        elif op == "del":
            d += 1
        elif op == "ins":
            ins += 1
        i, j = pi, pj
    return dp[n][m], s, d, ins


@dataclass
class CorpusCoreMetrics:
    cer: float
    cer_repo_compat: float
    character_accuracy: float
    wer: float
    word_accuracy: float
    exact_match_rate: float
    exact_match_count: int
    total_samples: int
    total_edit_distance: int
    total_reference_characters: int
    total_substitutions: int
    total_deletions: int
    total_insertions: int
    total_word_edits: int
    total_reference_words: int
    empty_prediction_count: int
    empty_prediction_rate: float


def corpus_core_metrics(
    expecteds: Sequence[str],
    predictions: Sequence[str],
) -> CorpusCoreMetrics:
    assert len(expecteds) == len(predictions)
    total_ed = 0
    total_ref = 0
    total_ref_compat = 0
    total_s = total_d = total_i = 0
    total_wed = 0
    total_wref = 0
    exact = 0
    empty = 0
    for gt, pred in zip(expecteds, predictions):
        m = sample_core_metrics(gt, pred)
        total_ed += m.edit_distance
        total_ref += m.reference_length
        total_ref_compat += max(m.reference_length, 1)
        total_s += m.substitutions
        total_d += m.deletions
        total_i += m.insertions
        total_wed += m.word_substitutions + m.word_deletions + m.word_insertions
        total_wref += len(tokenize_words(gt))
        if m.exact_match:
            exact += 1
        if pred == "":
            empty += 1

    n = len(expecteds)
    cer = (total_ed / total_ref) if total_ref > 0 else 0.0
    cer_compat = (total_ed / total_ref_compat) if total_ref_compat > 0 else 0.0
    wer = (total_wed / total_wref) if total_wref > 0 else 0.0
    return CorpusCoreMetrics(
        cer=cer,
        cer_repo_compat=cer_compat,
        character_accuracy=1.0 - cer,
        wer=wer,
        word_accuracy=1.0 - wer,
        exact_match_rate=(exact / n) if n else 0.0,
        exact_match_count=exact,
        total_samples=n,
        total_edit_distance=total_ed,
        total_reference_characters=total_ref,
        total_substitutions=total_s,
        total_deletions=total_d,
        total_insertions=total_i,
        total_word_edits=total_wed,
        total_reference_words=total_wref,
        empty_prediction_count=empty,
        empty_prediction_rate=(empty / n) if n else 0.0,
    )
