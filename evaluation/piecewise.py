"""Piecewise inference: split ORIGINAL-resolution line images BEFORE resize.

Critical order:
  original image → split pieces → existing PARSeq preprocess/OCR → join

Does not alter model weights, img_size, charset, tokenizer, EOS, or max_label_length.
Does NOT use GT / expected_text for splitting or joining.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .inference import InferenceResult, recognise_one

JOIN_CONCAT = "concat_no_separator"
JOIN_BOUNDARY = "boundary_aware"
BOUNDARY_WHITESPACE = "whitespace_gap"
BOUNDARY_FALLBACK = "fallback_gap"


@dataclass
class BoundaryInfo:
    """One internal cut between consecutive pieces."""

    boundary_x: int
    left_context: Tuple[int, int]  # (x0, x1) of left piece
    right_context: Tuple[int, int]
    gap_width: int
    gap_score: float
    boundary_type: str  # whitespace_gap | fallback_gap
    inserted_separator: bool = False


@dataclass
class PieceInfo:
    x0: int
    x1: int
    width: int
    height: int
    aspect_ratio: float
    post_resize_aspect_ratio: float
    prediction: str
    latency_ms: float = 0.0
    inference_error: bool = False
    error_message: Optional[str] = None


@dataclass
class PiecewiseResult:
    prediction: str
    latency_ms: float
    inference_error: bool
    error_message: Optional[str]
    piece_count: int
    pieces: List[PieceInfo] = field(default_factory=list)
    boundaries: List[BoundaryInfo] = field(default_factory=list)
    piecewise_join_method: str = JOIN_BOUNDARY
    prediction_concat_no_separator: str = ""
    prediction_boundary_aware: str = ""
    overlap_used: bool = False
    overlap_px: int = 0
    overlap_merge_diagnostics: List[dict] = field(default_factory=list)
    piece_target_aspect: float = 3.0
    nominal_bounds: List[Tuple[int, int]] = field(default_factory=list)


def _low_ink_gaps(
    gray: np.ndarray,
    min_gap_width: int = 3,
    ink_frac: float = 0.06,
) -> List[Tuple[int, int]]:
    """Vertical low-ink column runs (adapted from scripts/eval_piecewise.py)."""
    ink = (gray < 200).astype(np.int32)
    proj = ink.sum(axis=0)
    thresh = max(1, int(gray.shape[0] * ink_frac))
    is_gap = proj <= thresh
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap, start = True, i
        elif not g and in_gap:
            in_gap = False
            if i - start >= min_gap_width:
                gaps.append((start, i))
    if in_gap and len(is_gap) - start >= min_gap_width:
        gaps.append((start, len(is_gap)))
    return gaps


def _gap_centers(gaps: Sequence[Tuple[int, int]]) -> List[int]:
    return [(a + b) // 2 for a, b in gaps]


def _gap_lookup(gaps: Sequence[Tuple[int, int]]) -> dict:
    """Map each column in a gap → (gap_start, gap_end, width)."""
    lookup: dict = {}
    for a, b in gaps:
        w = b - a
        for x in range(a, b):
            lookup[x] = (a, b, w)
    return lookup


def compute_piece_plan(
    width: int,
    height: int,
    target_aspect: float = 3.0,
    gray: Optional[np.ndarray] = None,
    search_frac: float = 0.35,
) -> Tuple[List[Tuple[int, int]], List[BoundaryInfo]]:
    """Return contiguous [x0, x1) pieces covering [0, width) plus boundary metadata.

    Ideal piece width ≈ height * target_aspect. Cuts prefer nearby low-ink
    whitespace valleys when available. GT is never consulted.
    """
    if width <= 0 or height <= 0:
        return [(0, max(width, 0))], []

    target_w = max(1, int(round(height * float(target_aspect))))
    if width <= int(target_w * 1.25):
        return [(0, width)], []

    gaps: List[Tuple[int, int]] = []
    centers: List[int] = []
    if gray is not None:
        gaps = _low_ink_gaps(gray)
        centers = _gap_centers(gaps)
    gap_at = _gap_lookup(gaps)

    cuts: List[int] = [0]
    cut_meta: List[dict] = []  # parallel to cuts[1:-1] eventually
    x = 0
    search_radius = max(8, int(target_w * search_frac))

    while x + target_w < width:
        ideal = x + target_w
        min_cut = x + max(8, target_w // 4)
        max_cut = min(width - 8, ideal + search_radius)
        candidates = [c for c in centers if min_cut <= c <= max_cut]
        if candidates:
            cut = min(candidates, key=lambda c: abs(c - ideal))
            gap_info = gap_at.get(cut)
            if gap_info is not None:
                ga, gb, gw = gap_info
                btype = BOUNDARY_WHITESPACE
                gscore = float(gw)
            else:
                ga = gb = cut
                gw = 0
                btype = BOUNDARY_FALLBACK
                gscore = 0.0
        else:
            cut = min(max(ideal, min_cut), width - 8)
            gap_info = gap_at.get(cut)
            if gap_info is not None:
                ga, gb, gw = gap_info
                btype = BOUNDARY_WHITESPACE
                gscore = float(gw)
            else:
                ga = gb = cut
                gw = 0
                btype = BOUNDARY_FALLBACK
                gscore = 0.0

        if cut <= cuts[-1]:
            cut = min(cuts[-1] + target_w, width - 8)
            gap_info = gap_at.get(cut)
            if gap_info is not None:
                ga, gb, gw = gap_info
                btype = BOUNDARY_WHITESPACE
                gscore = float(gw)
            else:
                ga = gb = cut
                gw = 0
                btype = BOUNDARY_FALLBACK
                gscore = 0.0

        if cut >= width - 4:
            break
        cuts.append(cut)
        cut_meta.append(
            {
                "boundary_x": cut,
                "gap_width": int(gw),
                "gap_score": float(gscore),
                "boundary_type": btype,
            }
        )
        x = cut

    cuts.append(width)

    pieces: List[Tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if b > a:
            pieces.append((a, b))

    if not pieces:
        return [(0, width)], []
    if pieces[0][0] != 0:
        pieces[0] = (0, pieces[0][1])
    if pieces[-1][1] != width:
        pieces[-1] = (pieces[-1][0], width)

    merged: List[Tuple[int, int]] = [pieces[0]]
    for a, b in pieces[1:]:
        pa, pb = merged[-1]
        if a < pb:
            merged[-1] = (pa, max(pb, b))
        elif a == pb:
            merged.append((a, b))
        else:
            merged[-1] = (pa, b)

    # Rebuild boundary metadata aligned to final contiguous pieces
    boundaries: List[BoundaryInfo] = []
    for i in range(len(merged) - 1):
        left = merged[i]
        right = merged[i + 1]
        bx = left[1]  # contiguous cut
        # Prefer recorded meta at this x; else classify from gap lookup
        meta = next((m for m in cut_meta if m["boundary_x"] == bx), None)
        if meta is None:
            gap_info = gap_at.get(bx)
            if gap_info is not None:
                _, _, gw = gap_info
                btype = BOUNDARY_WHITESPACE
                gscore = float(gw)
            else:
                # Check if cut sits inside any gap
                gw = 0
                btype = BOUNDARY_FALLBACK
                gscore = 0.0
                for ga, gb in gaps:
                    if ga <= bx < gb:
                        gw = gb - ga
                        btype = BOUNDARY_WHITESPACE
                        gscore = float(gw)
                        break
            meta = {
                "boundary_x": bx,
                "gap_width": int(gw),
                "gap_score": float(gscore),
                "boundary_type": btype,
            }
        boundaries.append(
            BoundaryInfo(
                boundary_x=int(meta["boundary_x"]),
                left_context=(left[0], left[1]),
                right_context=(right[0], right[1]),
                gap_width=int(meta["gap_width"]),
                gap_score=float(meta["gap_score"]),
                boundary_type=str(meta["boundary_type"]),
                inserted_separator=False,
            )
        )

    return merged, boundaries


def compute_piece_boundaries(
    width: int,
    height: int,
    target_aspect: float = 3.0,
    gray: Optional[np.ndarray] = None,
    search_frac: float = 0.35,
) -> List[Tuple[int, int]]:
    """Backward-compatible wrapper — returns only piece ranges."""
    pieces, _ = compute_piece_plan(
        width, height, target_aspect=target_aspect, gray=gray, search_frac=search_frac
    )
    return pieces


def join_predictions(
    texts: Sequence[str],
    boundaries: Sequence[BoundaryInfo],
    join_mode: str = JOIN_BOUNDARY,
) -> Tuple[str, List[BoundaryInfo]]:
    """Join piece OCR strings.

    boundary_aware: insert a single space only when boundary_type is
    whitespace_gap (image evidence). Never blind " ".join.
    concat_no_separator: historical "".join behaviour.
    """
    # Fresh copies so callers can compare modes without mutating shared state
    b_out = [
        BoundaryInfo(
            boundary_x=b.boundary_x,
            left_context=b.left_context,
            right_context=b.right_context,
            gap_width=b.gap_width,
            gap_score=b.gap_score,
            boundary_type=b.boundary_type,
            inserted_separator=False,
        )
        for b in boundaries
    ]

    if not texts:
        return "", b_out

    if join_mode == JOIN_CONCAT:
        return "".join(texts), b_out

    if join_mode != JOIN_BOUNDARY:
        raise ValueError(f"Unknown join_mode: {join_mode}")

    out = texts[0]
    for i, b in enumerate(b_out):
        nxt = texts[i + 1] if i + 1 < len(texts) else ""
        insert = b.boundary_type == BOUNDARY_WHITESPACE
        # Avoid double spaces if a piece already carries edge whitespace
        if insert and (out.endswith((" ", "\t")) or nxt.startswith((" ", "\t"))):
            insert = False
        sep = " " if insert else ""
        b.inserted_separator = bool(sep)
        out = out + sep + nxt
    return out, b_out


def expand_bounds_with_overlap(
    nominal: Sequence[Tuple[int, int]],
    overlap_px: int,
    image_width: int,
) -> List[Tuple[int, int]]:
    """Expand contiguous [x0,x1) pieces by overlap_px into neighbors.

    Nominal cuts are unchanged; only the OCR crops grow. First piece still
    starts at 0; last still ends at image_width. GT is never used.
    """
    o = max(0, int(overlap_px))
    if o == 0 or len(nominal) <= 1:
        return [(int(a), int(b)) for a, b in nominal]

    out: List[Tuple[int, int]] = []
    n = len(nominal)
    for i, (a, b) in enumerate(nominal):
        left = a if i == 0 else max(0, a - o)
        right = b if i == n - 1 else min(image_width, b + o)
        # Keep at least 1px; clamp if neighbor too close
        if right <= left:
            right = min(image_width, left + 1)
        out.append((int(left), int(right)))
    return out


def _norm_for_overlap(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", s)


def merge_overlapping_pair(
    left: str,
    right: str,
    *,
    overlap_px: int,
    left_width: int,
    boundary_type: str,
    max_k_cap: int = 40,
) -> Tuple[str, dict]:
    """Deterministic merge of two OCR strings from spatially overlapping crops.

    Priority:
      1. Longest exact suffix/prefix match
      2. Longest NFC-normalized exact suffix/prefix match
      3. Best window with match ratio >= 0.8 (ties → longer k)
      4. Fallback: whitespace_gap → single space; else concat

    Never uses GT.
    """
    L = left or ""
    R = right or ""
    diag: dict = {
        "left_len": len(L),
        "right_len": len(R),
        "overlap_px": overlap_px,
        "method": None,
        "overlap_chars": 0,
        "boundary_type": boundary_type,
    }
    if not L:
        diag["method"] = "take_right"
        return R, diag
    if not R:
        diag["method"] = "take_left"
        return L, diag

    # Estimate max characters that could live in the overlap strip
    dens = (len(L) / max(left_width, 1)) if left_width else 0.1
    est = int(max(2, round(overlap_px * dens * 2))) + 2
    max_k = max(1, min(len(L), len(R), est, max_k_cap))

    # 1) exact
    best_k = 0
    for k in range(1, max_k + 1):
        if L.endswith(R[:k]):
            best_k = k
    if best_k > 0:
        diag["method"] = "exact_suffix_prefix"
        diag["overlap_chars"] = best_k
        return L + R[best_k:], diag

    # 2) NFC-normalized exact
    Ln, Rn = _norm_for_overlap(L), _norm_for_overlap(R)
    best_k = 0
    for k in range(1, min(max_k, len(Ln), len(Rn)) + 1):
        if Ln.endswith(Rn[:k]):
            best_k = k
    if best_k > 0:
        # Apply on normalized strings then return normalized merge
        # (predictions are already NFC-compatible Devanagari in practice)
        diag["method"] = "nfc_exact_suffix_prefix"
        diag["overlap_chars"] = best_k
        return Ln + Rn[best_k:], diag

    # 3) fuzzy ratio windows
    best_k = 0
    best_score = 0.0
    for k in range(1, max_k + 1):
        a, b = L[-k:], R[:k]
        if not a or not b:
            continue
        eq = sum(1 for x, y in zip(a, b) if x == y) / k
        if eq > best_score or (eq == best_score and k > best_k):
            best_score = eq
            best_k = k
    if best_k > 0 and best_score >= 0.8:
        diag["method"] = "fuzzy_ratio_ge_0.8"
        diag["overlap_chars"] = best_k
        diag["fuzzy_score"] = best_score
        # Prefer left's overlapping region, append right remainder
        return L + R[best_k:], diag

    # 4) fallback — same rule as boundary_aware when no text overlap found
    if boundary_type == BOUNDARY_WHITESPACE:
        sep = "" if (L.endswith((" ", "\t")) or R.startswith((" ", "\t"))) else " "
        diag["method"] = "fallback_whitespace_join"
        diag["overlap_chars"] = 0
        return L + sep + R, diag

    diag["method"] = "fallback_concat"
    return L + R, diag


def merge_overlapping_predictions(
    texts: Sequence[str],
    boundaries: Sequence[BoundaryInfo],
    *,
    overlap_px: int,
    piece_widths: Sequence[int],
) -> Tuple[str, List[BoundaryInfo], List[dict]]:
    """Left-to-right merge of overlapping piece OCR strings."""
    b_out = [
        BoundaryInfo(
            boundary_x=b.boundary_x,
            left_context=b.left_context,
            right_context=b.right_context,
            gap_width=b.gap_width,
            gap_score=b.gap_score,
            boundary_type=b.boundary_type,
            inserted_separator=False,
        )
        for b in boundaries
    ]
    if not texts:
        return "", b_out, []
    if overlap_px <= 0:
        # Delegate to boundary_aware for exact baseline reproduction
        joined, b2 = join_predictions(texts, boundaries, JOIN_BOUNDARY)
        return joined, b2, [{"method": "boundary_aware_no_overlap"}]

    out = texts[0]
    diags: List[dict] = []
    for i, b in enumerate(b_out):
        nxt = texts[i + 1] if i + 1 < len(texts) else ""
        lw = piece_widths[i] if i < len(piece_widths) else 1
        merged, diag = merge_overlapping_pair(
            out,
            nxt,
            overlap_px=overlap_px,
            left_width=lw,
            boundary_type=b.boundary_type,
        )
        # Detect whether a space was newly introduced at the join
        if len(merged) > len(out) and merged[len(out) : len(out) + 1] == " ":
            b.inserted_separator = True
        diags.append(diag)
        out = merged
    return out, b_out, diags


def recognise_piecewise(
    model,
    device: str,
    image_path: str,
    target_aspect: float = 3.0,
    model_img_size: Optional[Sequence[int]] = None,
    join_mode: str = JOIN_BOUNDARY,
    overlap_px: int = 0,
) -> PiecewiseResult:
    """Split original image, OCR each piece with existing pipeline, join.

    overlap_px > 0 expands crops across nominal cuts and merges OCR via
    deterministic suffix/prefix overlap (never uses GT).
    """
    import os
    import tempfile
    import time

    t0 = time.perf_counter()
    overlap_px = max(0, int(overlap_px))
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return PiecewiseResult(
            prediction="",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            inference_error=True,
            error_message=f"{type(e).__name__}: {e}",
            piece_count=0,
            piecewise_join_method=join_mode,
            piece_target_aspect=target_aspect,
            overlap_px=overlap_px,
            overlap_used=overlap_px > 0,
        )

    w, h = img.size
    gray = np.array(img.convert("L"))
    nominal, boundaries = compute_piece_plan(
        w, h, target_aspect=target_aspect, gray=gray
    )
    bounds = expand_bounds_with_overlap(nominal, overlap_px, w)

    if model_img_size is None:
        model_img_size = getattr(getattr(model, "hparams", None), "img_size", [32, 128])
    mh, mw = int(model_img_size[0]), int(model_img_size[1])
    post_ar = (mw / mh) if mh else None

    pieces: List[PieceInfo] = []
    texts: List[str] = []
    any_err = False
    err_msgs: List[str] = []

    for x0, x1 in bounds:
        crop = img.crop((x0, 0, x1, h))
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            crop.save(tmp)
            inf: InferenceResult = recognise_one(model, device, tmp, use_tta=False)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        texts.append(inf.prediction)
        if inf.inference_error:
            any_err = True
            if inf.error_message:
                err_msgs.append(inf.error_message)
        pw = x1 - x0
        pieces.append(
            PieceInfo(
                x0=x0,
                x1=x1,
                width=pw,
                height=h,
                aspect_ratio=(pw / h) if h else 0.0,
                post_resize_aspect_ratio=float(post_ar) if post_ar else 0.0,
                prediction=inf.prediction,
                latency_ms=inf.latency_ms,
                inference_error=inf.inference_error,
                error_message=inf.error_message,
            )
        )

    widths = [p.width for p in pieces]
    merge_diags: List[dict] = []
    if overlap_px > 0:
        primary, boundaries_ba, merge_diags = merge_overlapping_predictions(
            texts,
            boundaries,
            overlap_px=overlap_px,
            piece_widths=widths,
        )
        pred_concat = "".join(texts)  # raw concat of overlapping OCR (debug)
        pred_boundary = primary
        join_method = f"overlap_merge_{overlap_px}px"
    else:
        pred_concat, _ = join_predictions(texts, boundaries, JOIN_CONCAT)
        pred_boundary, boundaries_ba = join_predictions(texts, boundaries, JOIN_BOUNDARY)
        if join_mode == JOIN_CONCAT:
            primary = pred_concat
            join_method = JOIN_CONCAT
        elif join_mode == JOIN_BOUNDARY:
            primary = pred_boundary
            join_method = JOIN_BOUNDARY
        else:
            raise ValueError(f"Unknown join_mode: {join_mode}")

    latency = (time.perf_counter() - t0) * 1000.0
    return PiecewiseResult(
        prediction=primary,
        latency_ms=latency,
        inference_error=any_err,
        error_message="; ".join(err_msgs) if err_msgs else None,
        piece_count=len(pieces),
        pieces=pieces,
        boundaries=boundaries_ba,
        piecewise_join_method=join_method,
        prediction_concat_no_separator=pred_concat,
        prediction_boundary_aware=pred_boundary,
        overlap_used=overlap_px > 0,
        overlap_px=overlap_px,
        overlap_merge_diagnostics=merge_diags,
        piece_target_aspect=target_aspect,
        nominal_bounds=list(nominal),
    )


def piecewise_result_to_inference(pw: PiecewiseResult) -> InferenceResult:
    return InferenceResult(
        prediction=pw.prediction,
        latency_ms=pw.latency_ms,
        inference_error=pw.inference_error,
        error_message=pw.error_message,
    )


def pieces_as_dicts(pw: PiecewiseResult) -> List[dict]:
    return [asdict(p) for p in pw.pieces]


def boundaries_as_dicts(pw: PiecewiseResult) -> List[dict]:
    return [
        {
            "boundary": b.boundary_x,
            "boundary_x": b.boundary_x,
            "left_context": list(b.left_context),
            "right_context": list(b.right_context),
            "gap_width": b.gap_width,
            "gap_score": b.gap_score,
            "boundary_type": b.boundary_type,
            "inserted_separator": b.inserted_separator,
        }
        for b in pw.boundaries
    ]
