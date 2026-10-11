"""Length / truncation diagnostics (does not alter primary OCR metrics)."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core_metrics import corpus_core_metrics
from .reporting import _pctile


LENGTH_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("1-25", 1, 25),
    ("26-50", 26, 50),
    ("51-75", 51, 75),
    ("76-100", 76, 100),
    ("101+", 101, None),
)

# Capability-oriented subsets over the same frozen benchmark (not filters).
CAPABILITY_SUBSETS: Tuple[Tuple[str, str, int], ...] = (
    # (name, op, threshold) where op is "le" or "gt"
    ("training_native_subset", "le", 44),
    ("decoder_capacity_subset", "le", 50),
    ("over_decoder_capacity_subset", "gt", 50),
)


def extract_model_length_config(model) -> Dict[str, Any]:
    """Read actual length-related fields from a loaded PARSeq checkpoint.

    Never invents values. Missing fields are reported as None.
    """
    hp = getattr(model, "hparams", None)
    items: Dict[str, Any] = {}
    if hp is not None:
        try:
            items = dict(hp)
        except Exception:
            items = getattr(hp, "__dict__", {}) or {}

    def _hp(name: str):
        if name in items:
            return items[name]
        return getattr(hp, name, None) if hp is not None else None

    img_size = _hp("img_size")
    max_label_length = _hp("max_label_length")
    max_length = _hp("max_length")
    max_len = _hp("max_len")

    # Nested model object (PARSeq.model.max_label_length)
    nested = getattr(model, "model", None)
    nested_max = getattr(nested, "max_label_length", None) if nested is not None else None

    tok = getattr(model, "tokenizer", None)
    tok_max_length = getattr(tok, "max_length", None) if tok is not None else None
    tok_max_len = getattr(tok, "max_len", None) if tok is not None else None
    eos_id = getattr(tok, "eos_id", None) if tok is not None else None
    bos_id = getattr(tok, "bos_id", None) if tok is not None else None
    pad_id = getattr(tok, "pad_id", None) if tok is not None else None

    # Reliable decoder max: prefer hparams.max_label_length, else nested
    reliable_max = None
    if isinstance(max_label_length, int) and max_label_length > 0:
        reliable_max = int(max_label_length)
    elif isinstance(nested_max, int) and nested_max > 0:
        reliable_max = int(nested_max)

    return {
        "img_size": img_size if img_size is not None else None,
        "max_label_length": max_label_length if max_label_length is not None else None,
        "max_length": max_length if max_length is not None else None,
        "max_len": max_len if max_len is not None else None,
        "nested_model_max_label_length": nested_max,
        "tokenizer_max_length": tok_max_length,
        "tokenizer_max_len": tok_max_len,
        "eos_id": eos_id,
        "bos_id": bos_id,
        "pad_id": pad_id,
        "reliable_max_decoded_length": reliable_max,
    }


def format_model_diagnostics(cfg: Dict[str, Any]) -> str:
    def _fmt(v):
        return "not_available" if v is None else repr(v)

    lines = [
        "Model diagnostics",
        "-----------------",
        f"img_size: {_fmt(cfg.get('img_size'))}",
        f"max_label_length: {_fmt(cfg.get('max_label_length'))}",
        f"max_length: {_fmt(cfg.get('max_length'))}",
        f"max_len: {_fmt(cfg.get('max_len'))}",
        f"nested_model_max_label_length: {_fmt(cfg.get('nested_model_max_label_length'))}",
        f"tokenizer_max_length: {_fmt(cfg.get('tokenizer_max_length'))}",
        f"tokenizer_max_len: {_fmt(cfg.get('tokenizer_max_len'))}",
        f"eos_id: {_fmt(cfg.get('eos_id'))}",
        f"bos_id: {_fmt(cfg.get('bos_id'))}",
        f"pad_id: {_fmt(cfg.get('pad_id'))}",
        f"reliable_max_decoded_length: {_fmt(cfg.get('reliable_max_decoded_length'))}",
    ]
    return "\n".join(lines)


def length_ratio(reference_length: int, prediction_length: int) -> Optional[float]:
    if reference_length <= 0:
        return None if prediction_length > 0 else 1.0
    return prediction_length / reference_length


def prediction_reaches_max_length(
    prediction_length: int,
    reliable_max: Optional[int],
) -> Optional[bool]:
    if reliable_max is None:
        return None
    return prediction_length >= reliable_max


def _distribution(values: List[float]) -> dict:
    if not values:
        return {
            "mean": None, "median": None, "p10": None, "p50": None, "p90": None,
            "min": None, "max": None, "count": 0,
        }
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": _pctile(values, 10),
        "p50": _pctile(values, 50),
        "p90": _pctile(values, 90),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def bucket_for_length(ref_len: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if hi is None:
            if ref_len >= lo:
                return name
        elif lo <= ref_len <= hi:
            return name
    return "0"  # empty GT etc.


def compute_length_aggregates(
    per_sample: Sequence[dict],
    reliable_max: Optional[int],
) -> dict:
    gt_lens = [int(r["reference_length"]) for r in per_sample]
    pred_lens = [int(r["prediction_length"]) for r in per_sample]
    ratios = []
    under = over = 0
    at_max = 0
    for r in per_sample:
        ref = int(r["reference_length"])
        pred = int(r["prediction_length"])
        ratio = length_ratio(ref, pred)
        if ratio is not None and ratio >= 0:
            ratios.append(ratio)
        if ref > 0:
            if pred < ref:
                under += 1
            elif pred > ref:
                over += 1
        if r.get("prediction_reaches_max_length") is True:
            at_max += 1

    n = len(per_sample)
    out = {
        "gt_length": _distribution([float(x) for x in gt_lens]),
        "prediction_length": _distribution([float(x) for x in pred_lens]),
        "length_ratio": _distribution(ratios),
        "under_generation_count": under,
        "over_generation_count": over,
        "reliable_max_decoded_length": reliable_max,
        "samples_at_max_length": at_max if reliable_max is not None else None,
        "prediction_reaches_max_length_rate": (
            (at_max / n) if (reliable_max is not None and n) else None
        ),
        "max_length_analysis_available": reliable_max is not None,
    }
    return out


def _summarize_length_group(rows: Sequence[dict]) -> dict:
    """Shared metrics for a GT-length stratum (bucket or capability subset)."""
    if not rows:
        return {
            "sample_count": 0,
            "cer": None,
            "cer_repo_compat": None,
            "wer": None,
            "exact_match_rate": None,
            "character_accuracy": None,
            "word_accuracy": None,
            "mean_reference_length": None,
            "mean_prediction_length": None,
            "mean_length_ratio": None,
            "under_generation_rate": None,
            "deletion_share": None,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "empty_prediction_rate": None,
            "marathi_numeral_accuracy": None,
            "punctuation_f1": None,
            "conjunct_accuracy": None,
            "matra_accuracy": None,
        }

    gts = [r["expected_text"] for r in rows]
    preds = [r["prediction"] for r in rows]
    m = corpus_core_metrics(gts, preds)
    refs = [int(r["reference_length"]) for r in rows]
    preds_l = [int(r["prediction_length"]) for r in rows]
    ratios = [
        length_ratio(a, b)
        for a, b in zip(refs, preds_l)
        if length_ratio(a, b) is not None
    ]
    under = sum(1 for a, b in zip(refs, preds_l) if a > 0 and b < a)
    total_ed = m.total_edit_distance
    del_share = (m.total_deletions / total_ed) if total_ed > 0 else None

    def _mean_field(key: str):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    return {
        "sample_count": len(rows),
        "cer": m.cer,
        "cer_repo_compat": m.cer_repo_compat,
        "wer": m.wer,
        "exact_match_rate": m.exact_match_rate,
        "character_accuracy": m.character_accuracy,
        "word_accuracy": m.word_accuracy,
        "mean_reference_length": statistics.mean(refs),
        "mean_prediction_length": statistics.mean(preds_l),
        "mean_length_ratio": statistics.mean(ratios) if ratios else None,
        "under_generation_rate": under / len(rows),
        "deletion_share": del_share,
        "substitutions": m.total_substitutions,
        "deletions": m.total_deletions,
        "insertions": m.total_insertions,
        "total_edit_distance": total_ed,
        "empty_prediction_rate": m.empty_prediction_rate,
        "marathi_numeral_accuracy": _mean_field("marathi_numeral_accuracy"),
        "punctuation_f1": _mean_field("punctuation_f1"),
        "conjunct_accuracy": _mean_field("conjunct_accuracy"),
        "matra_accuracy": _mean_field("matra_accuracy"),
    }


def compute_gt_length_buckets(per_sample: Sequence[dict]) -> Dict[str, dict]:
    buckets: Dict[str, List[dict]] = {name: [] for name, _, _ in LENGTH_BUCKETS}
    for r in per_sample:
        b = bucket_for_length(int(r["reference_length"]))
        if b in buckets:
            buckets[b].append(r)

    return {name: _summarize_length_group(buckets[name]) for name, _, _ in LENGTH_BUCKETS}


def capability_subset_for_length(ref_len: int) -> List[str]:
    """Return capability subset names that include this GT length."""
    names: List[str] = []
    for name, op, thr in CAPABILITY_SUBSETS:
        if op == "le" and ref_len <= thr:
            names.append(name)
        elif op == "gt" and ref_len > thr:
            names.append(name)
    return names


def compute_capability_subsets(per_sample: Sequence[dict]) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {name: [] for name, _, _ in CAPABILITY_SUBSETS}
    for r in per_sample:
        ref = int(r["reference_length"])
        for name in capability_subset_for_length(ref):
            groups[name].append(r)
    return {name: _summarize_length_group(groups[name]) for name, _, _ in CAPABILITY_SUBSETS}


def compute_all_length_strata(per_sample: Sequence[dict]) -> Dict[str, dict]:
    """Buckets + capability subsets in one report block."""
    out: Dict[str, dict] = {}
    out.update(compute_gt_length_buckets(per_sample))
    out.update(compute_capability_subsets(per_sample))
    return out


def compute_error_composition(
    total_substitutions: int,
    total_deletions: int,
    total_insertions: int,
    total_edit_distance: int,
) -> dict:
    if total_edit_distance <= 0:
        return {
            "substitution_share": None,
            "deletion_share": None,
            "insertion_share": None,
            "total_edit_distance": total_edit_distance,
        }
    return {
        "substitution_share": total_substitutions / total_edit_distance,
        "deletion_share": total_deletions / total_edit_distance,
        "insertion_share": total_insertions / total_edit_distance,
        "total_edit_distance": total_edit_distance,
        "substitutions": total_substitutions,
        "deletions": total_deletions,
        "insertions": total_insertions,
    }


def longest_sample_rows(per_sample: Sequence[dict], n: int = 20) -> List[dict]:
    ranked = sorted(
        per_sample,
        key=lambda r: (-int(r["reference_length"]), int(r.get("index") or 0)),
    )
    out = []
    for r in ranked[:n]:
        out.append({
            "image_filename": r["image_filename"],
            "reference_length": r["reference_length"],
            "prediction_length": r["prediction_length"],
            "length_ratio": r.get("length_ratio"),
            "edit_distance": r["edit_distance"],
            "substitutions": r["substitutions"],
            "deletions": r["deletions"],
            "insertions": r["insertions"],
            "cer": r["cer"],
            "prediction_reaches_max_length": r.get("prediction_reaches_max_length"),
            "expected_text": r.get("expected_text"),
            "prediction": r.get("prediction"),
        })
    return out


def interpret_length_evidence(
    reliable_max: Optional[int],
    at_max_rate: Optional[float],
    mean_pred_len: Optional[float],
    error_composition: dict,
    buckets: Dict[str, dict],
) -> dict:
    """Separate observed facts from diagnostic interpretation.

    Returns conclusion in {A, B, C} with supporting notes.
    """
    facts = []
    if reliable_max is not None:
        facts.append(f"Reliable decoder max_label_length = {reliable_max}")
        facts.append(
            f"prediction_reaches_max_length_rate = {at_max_rate}"
        )
        facts.append(f"mean prediction length = {mean_pred_len}")
    else:
        facts.append("No reliable decoder maximum available")

    facts.append(
        f"deletion_share = {error_composition.get('deletion_share')}"
    )
    facts.append(
        f"substitution_share = {error_composition.get('substitution_share')}"
    )
    facts.append(
        f"insertion_share = {error_composition.get('insertion_share')}"
    )

    for name, m in buckets.items():
        if m.get("sample_count"):
            facts.append(
                f"bucket {name}: n={m['sample_count']} CER={m.get('cer')} "
                f"mean_ratio={m.get('mean_length_ratio')} "
                f"del_share={m.get('deletion_share')}"
            )

    del_share = error_composition.get("deletion_share") or 0.0
    sub_share = error_composition.get("substitution_share") or 0.0
    at_max = at_max_rate or 0.0

    if reliable_max is not None and at_max >= 0.5 and del_share >= 0.4:
        code = "A"
        interpretation = (
            "Strong evidence of output-length limitation: many predictions "
            "hit the known decoder maximum and deletions dominate edits."
        )
    elif reliable_max is not None and at_max < 0.2 and (
        mean_pred_len is not None and mean_pred_len < 0.7 * reliable_max
    ):
        # Predictions stop well below the hard max → early EOS / content errors
        if sub_share >= 0.3 or del_share >= 0.3:
            code = "B"
            interpretation = (
                "No evidence that a hard output-length limit is the primary "
                "failure mode: predictions usually end before the known "
                f"max_label_length={reliable_max}, while substitutions and/or "
                "deletions remain large — consistent with incorrect early "
                "decoding (EOS) and/or content errors, not saturation at the cap."
            )
        else:
            code = "C"
            interpretation = (
                "Mixed: max length is known and rarely hit, but error "
                "composition is not clearly dominated by one op type."
            )
    elif reliable_max is None:
        code = "C"
        interpretation = (
            "Uncertain: predictions are short and deletions may be high, "
            "but truncation cannot be proven without a reliable decoder max."
        )
    else:
        code = "C"
        interpretation = (
            "Mixed/uncertain: a decoder max exists, but saturation rate and "
            "error composition do not strongly support a single explanation."
        )

    return {
        "conclusion_code": code,
        "observed_facts": facts,
        "diagnostic_interpretation": interpretation,
    }
