"""Result writing: per_sample, errors, summary, hashes, plots."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import METRIC_VERSION, NORMALIZATION_VERSION
from .core_metrics import CorpusCoreMetrics


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_manifest_sha256(image_paths: Sequence[str]) -> str:
    """Deterministic hash over sorted relative basenames + file sha256."""
    lines = []
    for p in sorted(image_paths):
        digest = sha256_file(p)
        lines.append(f"{os.path.basename(p)}\t{digest}")
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def git_commit(repo_root: Optional[str | Path] = None) -> Optional[str]:
    try:
        cwd = str(repo_root) if repo_root else None
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _pctile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def length_stats(ratios: List[float], under: int, over: int) -> dict:
    """Backward-compatible ratio-only stats (kept for existing callers)."""
    if not ratios:
        return {
            "mean": None, "median": None, "p10": None, "p50": None, "p90": None,
            "min": None, "max": None,
            "under_generation_count": under,
            "over_generation_count": over,
        }
    finite = [r for r in ratios if r >= 0]
    return {
        "mean": statistics.mean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p10": _pctile(finite, 10),
        "p50": _pctile(finite, 50),
        "p90": _pctile(finite, 90),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "under_generation_count": under,
        "over_generation_count": over,
    }


def latency_stats(latencies_ms: List[float]) -> dict:
    if not latencies_ms:
        return {
            "total_inference_time_s": 0.0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "samples_per_sec": None,
            "images_per_minute": None,
        }
    total_s = sum(latencies_ms) / 1000.0
    n = len(latencies_ms)
    sps = (n / total_s) if total_s > 0 else None
    return {
        "total_inference_time_s": total_s,
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": _pctile(latencies_ms, 95),
        "samples_per_sec": sps,
        "images_per_minute": (sps * 60.0) if sps is not None else None,
    }


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _script_metrics_block(summary: dict) -> tuple:
    """Return (label, metrics_dict) for language-specific section."""
    lang = (summary.get("language") or "marathi").lower()
    if lang == "gujarati" and isinstance(summary.get("gujarati"), dict):
        return "GUJARATI-SPECIFIC", summary["gujarati"]
    if isinstance(summary.get("marathi"), dict):
        return "MARATHI-SPECIFIC", summary["marathi"]
    if isinstance(summary.get("gujarati"), dict):
        return "GUJARATI-SPECIFIC", summary["gujarati"]
    return "SCRIPT-SPECIFIC", {}


def format_summary_md(summary: dict) -> str:
    p = summary.get("primary", {})
    edits = summary.get("edit_counts", {})
    fails = summary.get("failure_counts", {})
    script_label, mar = _script_metrics_block(summary)
    lang = summary.get("language") or "marathi"
    title = "Gujarati OCR Evaluation" if lang == "gujarati" else "Marathi OCR Evaluation"
    lines = [
        title,
        "======================",
        "",
        f"Model / Checkpoint: {summary.get('checkpoint_path')}",
        f"Dataset: {summary.get('dataset_path')}",
        f"Dataset size: {summary.get('dataset_sample_count')}",
        f"Language: {lang}",
        f"Device: {summary.get('device')}",
        f"Batch size: {summary.get('batch_size')}",
        f"TTA: {summary.get('tta')}",
        f"Mode: {summary.get('mode')}",
        f"Timestamp: {summary.get('timestamp')}",
        f"Metric version: {summary.get('metric_version')}",
        f"Normalization version: {summary.get('normalization_version')}",
    ]
    pipe = summary.get("pipeline") or {}
    if pipe:
        lines.extend(
            [
                f"Pipeline version: {pipe.get('pipeline_version')}",
                (
                    f"Pipeline: mode={pipe.get('mode')} "
                    f"aspect={pipe.get('piece_target_aspect')} "
                    f"join={pipe.get('join_mode')} "
                    f"overlap={pipe.get('piece_overlap_px')} "
                    f"canonical_run={summary.get('canonical_run')}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "PRIMARY METRICS",
            "---------------",
            f"CER: {p.get('cer')}",
            f"CER_repo_compat: {p.get('cer_repo_compat')}",
            f"WER (strict): {p.get('wer_strict', p.get('wer'))}",
            f"Exact String Match: {p.get('exact_match_rate')}",
            f"Character Accuracy: {p.get('character_accuracy')}",
            f"Word Accuracy: {p.get('word_accuracy')}",
            "",
            "WER VIEWS",
            "---------",
        ]
    )
    wv = summary.get("wer_views") or {}
    if wv:
        lines.extend(
            [
                f"wer_strict: {wv.get('wer_strict')}",
                f"wer_normalized: {wv.get('wer_normalized')}",
                f"wer_content_aware: {wv.get('wer_content_aware')}",
                f"lexical_wer: {wv.get('lexical_wer')}",
                f"content_match_rate: {wv.get('content_match_rate')}",
                f"whitespace_boundary_error_rate: {wv.get('whitespace_boundary_error_rate')}",
                f"word_boundary_merges: {wv.get('word_boundary_merge_count')}",
                f"word_boundary_splits: {wv.get('word_boundary_split_count')}",
                (
                    "note: wer_content_aware is an OCR diagnostic separating "
                    "content mistakes from whitespace-boundary-only differences; "
                    "it is not NIST-style WER."
                ),
            ]
        )
    else:
        lines.append("(wer_views unavailable)")
    lines.extend(
        [
            "",
            "EDIT COUNTS",
            "-----------",
            f"Substitutions: {edits.get('substitutions')}",
            f"Deletions: {edits.get('deletions')}",
            f"Insertions: {edits.get('insertions')}",
            "",
            "FAILURE COUNTS",
            "--------------",
            f"Empty Predictions: {fails.get('empty_predictions')}",
            f"Inference Errors: {fails.get('inference_errors')}",
            "",
            script_label,
            "-" * len(script_label),
            f"Native Numeral Accuracy: {mar.get('native_numeral_character_accuracy')}",
            f"Marathi Numeral Accuracy: {mar.get('marathi_numeral_character_accuracy')}",
            f"Gujarati Numeral Accuracy: {mar.get('gujarati_numeral_character_accuracy')}",
            f"Numeric Exact Match: {mar.get('numeric_exact_match_rate')}",
            f"Numeric CER: {mar.get('numeric_token_cer')}",
            f"Numeral Script Errors: {mar.get('numeral_script_error_count')}",
            f"Punctuation Precision: {mar.get('punctuation_precision')}",
            f"Punctuation Recall: {mar.get('punctuation_recall')}",
            f"Punctuation F1: {mar.get('punctuation_f1')}",
            f"Punctuation Exact Match: {mar.get('punctuation_exact_match_rate')}",
            f"Conjunct Accuracy: {mar.get('conjunct_occurrence_accuracy')}",
            f"Matra / Vowel-Sign Accuracy: {mar.get('matra_accuracy')}",
            f"Reference-ID Exact Match: {mar.get('reference_id_exact_match_rate')}",
            f"Reference-ID CER: {mar.get('reference_id_cer')}",
            "",
            f"difficulty_source: {summary.get('difficulty_source')}",
            "",
            "PIECEWISE STATS",
            "---------------",
        ]
    )
    pw = summary.get("piecewise_stats") or {}
    if not pw.get("enabled"):
        lines.append("Mode: whole (piecewise disabled)")
    else:
        lines.append(f"Mode: piecewise")
        lines.append(f"piece_target_aspect: {summary.get('piece_target_aspect')}")
        lines.append(f"Mean piece count: {pw.get('mean_piece_count')}")
        lines.append(f"Median piece count: {pw.get('median_piece_count')}")
        lines.append(f"Min/Max piece count: {pw.get('min_piece_count')} / {pw.get('max_piece_count')}")
        lines.append(f"Mean piece width: {pw.get('mean_piece_width')}")
        lines.append(f"Mean piece height: {pw.get('mean_piece_height')}")
        lines.append(f"Mean piece aspect ratio: {pw.get('mean_piece_aspect_ratio')}")
        lines.append(f"Pct samples >1 piece: {pw.get('pct_samples_requiring_gt1_piece')}")
        lines.append(f"Join method: {pw.get('join_method')}")

    lines.extend([
        "",
        "ISSUE TYPE BREAKDOWN",
        "--------------------",
    ])
    for issue, m in (summary.get("issue_type") or {}).items():
        lines.append(
            f"- {issue}: n={m.get('sample_count')} CER={m.get('cer')} "
            f"WER={m.get('wer')} exact={m.get('exact_match')}"
        )
    diff = summary.get("difficulty")
    lines.extend(["", "HARD / NORMAL", "-------------"])
    if not diff:
        lines.append("difficulty metrics: unavailable")
    else:
        for k, m in diff.items():
            lines.append(
                f"- {k}: n={m.get('sample_count')} CER={m.get('cer')} "
                f"WER={m.get('wer')} exact={m.get('exact_match')}"
            )

    # Length diagnostics
    ld = summary.get("length_diagnostics") or {}
    lines.extend(["", "LENGTH DIAGNOSTICS", "------------------"])
    cfg = summary.get("model_length_config") or {}
    lines.append(f"reliable_max_decoded_length: {cfg.get('reliable_max_decoded_length')}")
    if not ld.get("max_length_analysis_available"):
        lines.append("Maximum decoded length: unavailable")
        lines.append("Truncation saturation analysis: unavailable")
    else:
        lines.append(f"samples_at_max_length: {ld.get('samples_at_max_length')}")
        lines.append(
            f"prediction_reaches_max_length_rate: "
            f"{ld.get('prediction_reaches_max_length_rate')}"
        )

    def _fmt_dist(title: str, d: Optional[dict]) -> None:
        d = d or {}
        lines.append(f"{title}:")
        lines.append(
            f"    mean={d.get('mean')} median={d.get('median')} "
            f"p10={d.get('p10')} p50={d.get('p50')} p90={d.get('p90')} "
            f"min={d.get('min')} max={d.get('max')}"
        )

    _fmt_dist("GT length", ld.get("gt_length"))
    _fmt_dist("Prediction length", ld.get("prediction_length"))
    _fmt_dist("Length ratio", ld.get("length_ratio"))
    lines.append(f"Under-generation count: {ld.get('under_generation_count')}")
    lines.append(f"Over-generation count: {ld.get('over_generation_count')}")

    ec = summary.get("error_composition") or {}
    lines.extend(["", "ERROR COMPOSITION", "-----------------"])
    lines.append(f"Substitution share: {ec.get('substitution_share')}")
    lines.append(f"Deletion share: {ec.get('deletion_share')}")
    lines.append(f"Insertion share: {ec.get('insertion_share')}")

    lines.extend(["", "GT LENGTH BUCKETS", "-----------------"])
    lines.append(
        "Length Bucket | Samples | CER | Mean Pred/GT Ratio | Deletion Share"
    )
    for name, m in (summary.get("gt_length_buckets") or {}).items():
        lines.append(
            f"{name} | {m.get('sample_count')} | {m.get('cer')} | "
            f"{m.get('mean_length_ratio')} | {m.get('deletion_share')}"
        )
        if m.get("sample_count"):
            lines.append(
                f"    exact={m.get('exact_match_rate')} "
                f"mean_gt={m.get('mean_reference_length')} "
                f"mean_pred={m.get('mean_prediction_length')} "
                f"under_gen_rate={m.get('under_generation_rate')}"
            )

    lines.extend(["", "CAPABILITY SUBSETS", "------------------"])
    for name, m in (summary.get("capability_subsets") or {}).items():
        lines.append(
            f"{name} | n={m.get('sample_count')} CER={m.get('cer')} "
            f"WER={m.get('wer')} Exact={m.get('exact_match_rate')} "
            f"mean_len_ratio={m.get('mean_length_ratio')} "
            f"del_share={m.get('deletion_share')}"
        )

    ws = summary.get("whitespace_diagnostics") or {}
    if ws:
        lines.extend(["", "WHITESPACE DIAGNOSTICS (secondary)", "---------------------------------"])
        lines.append(f"strict_CER: {ws.get('strict_cer')}")
        lines.append(f"whitespace_normalized_CER: {ws.get('whitespace_normalized_cer')}")
        lines.append(
            f"content_without_whitespace_CER: {ws.get('content_without_whitespace_cer')}"
        )
        lines.append(
            f"strict_exact_rate: {ws.get('strict_exact_match_rate')} | "
            f"ws_norm_exact_rate: {ws.get('whitespace_normalized_exact_match_rate')} | "
            f"content_no_ws_exact_rate: {ws.get('content_without_whitespace_exact_match_rate')}"
        )
        lines.append(
            f"space_ins={ws.get('space_insertion_count')} "
            f"space_del={ws.get('space_deletion_count')} "
            f"space_sub={ws.get('space_substitution_count')} "
            f"ws_errors={ws.get('whitespace_error_count')}"
        )
        lines.append(
            f"word_boundary_merge_count={ws.get('word_boundary_merge_count')} "
            f"word_boundary_split_count={ws.get('word_boundary_split_count')}"
        )

    if summary.get("mode") == "piecewise":
        ps = summary.get("piecewise_stats") or {}
        lines.extend(["", "PIECEWISE", "---------"])
        lines.append(f"join_mode: {summary.get('join_mode')}")
        lines.append(f"mean_piece_count: {ps.get('mean_piece_count')}")
        lines.append(
            f"boundary_whitespace_gap_count: {ps.get('boundary_whitespace_gap_count')}"
        )
        lines.append(
            f"boundary_fallback_gap_count: {ps.get('boundary_fallback_gap_count')}"
        )
        lines.append(
            f"inserted_separator_count: {ps.get('inserted_separator_count')}"
        )

    len_ev = summary.get("length_evidence") or {}
    if len_ev:
        lines.extend(["", "LENGTH EVIDENCE CONCLUSION", "--------------------------"])
        lines.append(f"conclusion_code: {len_ev.get('conclusion_code')}")
        lines.append("observed_facts:")
        for f in len_ev.get("observed_facts") or []:
            lines.append(f"  - {f}")
        lines.append(
            f"diagnostic_interpretation: {len_ev.get('diagnostic_interpretation')}"
        )

    lines.extend(["", "WORST FAILURES", "--------------"])
    for row in summary.get("worst_failures", [])[:20]:
        lines.append(
            f"- {row.get('image_filename')} CER={row.get('cer')} "
            f"issue={row.get('issue_type')}"
        )
        lines.append(f"  GT:   {row.get('expected_text')}")
        lines.append(f"  PRED: {row.get('prediction')}")

    diagnosis = summary.get("diagnosis") or {}
    if diagnosis:
        lines.extend(["", "MODEL QUALITY SUMMARY", "---------------------"])
        for k, v in diagnosis.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: {v}")

    lines.append("")
    return "\n".join(lines)


def write_summary_md(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_summary_md(summary), encoding="utf-8")


def maybe_make_plots(output_dir: Path, per_sample: Sequence[dict], enable: bool) -> None:
    if not enable:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    cers = [r["cer"] for r in per_sample if r.get("cer") is not None]
    wers = [r["wer"] for r in per_sample if r.get("wer") is not None]

    if cers:
        plt.figure()
        plt.hist(cers, bins=20)
        plt.xlabel("CER")
        plt.ylabel("count")
        plt.title("CER distribution")
        plt.tight_layout()
        plt.savefig(plots / "cer_distribution.png")
        plt.close()

    if wers:
        plt.figure()
        plt.hist(wers, bins=20)
        plt.xlabel("WER")
        plt.ylabel("count")
        plt.title("WER distribution")
        plt.tight_layout()
        plt.savefig(plots / "wer_distribution.png")
        plt.close()

    err_counter: Counter = Counter()
    for r in per_sample:
        for t in r.get("error_types") or []:
            err_counter[t] += 1
    if err_counter:
        labels, vals = zip(*err_counter.most_common(15))
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("count")
        plt.title("Error types")
        plt.tight_layout()
        plt.savefig(plots / "error_types.png")
        plt.close()

    # issue_type CER
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in per_sample:
        buckets[r.get("issue_type") or "unknown"].append(r.get("cer") or 0.0)
    if buckets:
        keys = sorted(buckets)
        means = [statistics.mean(buckets[k]) for k in keys]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(keys)), means)
        plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
        plt.ylabel("mean CER")
        plt.title("Issue-type CER")
        plt.tight_layout()
        plt.savefig(plots / "issue_type_cer.png")
        plt.close()

    nums = [
        r["marathi_numeral_accuracy"]
        for r in per_sample
        if r.get("marathi_numeral_accuracy") is not None
    ]
    if nums:
        plt.figure()
        plt.hist(nums, bins=20)
        plt.xlabel("Marathi numeral accuracy")
        plt.ylabel("count")
        plt.title("Numeral accuracy")
        plt.tight_layout()
        plt.savefig(plots / "numeral_accuracy.png")
        plt.close()

    # Reference vs prediction length (per sample index)
    if per_sample:
        idxs = list(range(1, len(per_sample) + 1))
        refs = [r.get("reference_length") or 0 for r in per_sample]
        preds = [r.get("prediction_length") or 0 for r in per_sample]
        plt.figure(figsize=(12, 4))
        plt.plot(idxs, refs, label="reference_length", linewidth=1)
        plt.plot(idxs, preds, label="prediction_length", linewidth=1)
        plt.xlabel("sample")
        plt.ylabel("character count")
        plt.title("Reference vs prediction length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "reference_vs_prediction_length.png")
        plt.close()

        ratios = [
            r.get("length_ratio")
            for r in per_sample
            if r.get("length_ratio") is not None and r.get("length_ratio") >= 0
        ]
        if ratios:
            plt.figure()
            plt.hist(ratios, bins=20)
            plt.xlabel("prediction_length / reference_length")
            plt.ylabel("count")
            plt.title("Length ratio distribution")
            plt.tight_layout()
            plt.savefig(plots / "length_ratio_distribution.png")
            plt.close()


def build_diagnosis(summary: dict) -> dict:
    p = summary.get("primary", {})
    _, mar = _script_metrics_block(summary)
    lang = (summary.get("language") or "marathi").lower()
    strengths = []
    weaknesses = []
    if (p.get("character_accuracy") or 0) >= 0.9:
        strengths.append("High character accuracy on this validation set.")
    if (p.get("exact_match_rate") or 0) >= 0.5:
        strengths.append("Exact-match rate at or above 50%.")
    nat_acc = mar.get("native_numeral_character_accuracy")
    if nat_acc is None:
        nat_acc = mar.get("gujarati_numeral_character_accuracy")
    if nat_acc is None:
        nat_acc = mar.get("marathi_numeral_character_accuracy")
    if (nat_acc or 1) < 0.9:
        label = "Gujarati" if lang == "gujarati" else "Marathi"
        weaknesses.append(f"{label} numerals: accuracy={nat_acc}")
    if mar.get("reference_id_applicable", True) and (
        mar.get("reference_id_exact_match_rate") or 1
    ) < 0.5:
        weaknesses.append(
            f"Reference IDs: exact={mar.get('reference_id_exact_match_rate')}"
        )
    if (mar.get("punctuation_f1") or 1) < 0.9:
        weaknesses.append(f"Punctuation F1={mar.get('punctuation_f1')}")
    if (mar.get("conjunct_occurrence_accuracy") or 1) < 0.9:
        weaknesses.append(f"Conjuncts: accuracy={mar.get('conjunct_occurrence_accuracy')}")
    if (mar.get("matra_accuracy") or 1) < 0.95:
        label = "Vowel signs" if lang == "gujarati" else "Matras"
        weaknesses.append(f"{label}: accuracy={mar.get('matra_accuracy')}")

    worst = summary.get("worst_failures") or []
    common = []
    c = Counter()
    for w in worst:
        for t in w.get("error_types") or []:
            c[t] += 1
    for t, n in c.most_common(5):
        common.append(f"{t} ({n})")

    return {
        "overall": {
            "cer": p.get("cer"),
            "wer": p.get("wer"),
            "exact_match": p.get("exact_match_rate"),
        },
        "strengths": strengths or ["See primary metrics."],
        "weaknesses": weaknesses or ["No major specialized weaknesses flagged by thresholds."],
        "most_common_failures": common,
    }
