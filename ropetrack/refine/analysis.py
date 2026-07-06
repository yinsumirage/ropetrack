"""Numpy-only helpers for rope residual closure and correlation analysis.

Shared by apply_rope_refinement.py (summary stats), the sliced scorer, and
the dead-zone analysis. No torch import so CPU scoring jobs stay light.
"""

from __future__ import annotations

import math

import numpy as np

from ropetrack.rope import FINGER_ORDER


def json_sanitize(obj):
    """Replace non-finite floats with None so json.dumps emits strict JSON.

    In-memory stats keep float('nan') as the degenerate-value sentinel; this
    is applied only at the serialization boundary.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {key: json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(value) for value in obj]
    return obj


def rope_abs_residual(pred_rope_norm: np.ndarray, target_rope_norm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Masked |pred - target| in normalized rope units; invalid entries are NaN."""
    pred = np.asarray(pred_rope_norm, dtype=np.float64)
    target = np.asarray(target_rope_norm, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: pred={pred.shape} target={target.shape} valid={mask.shape}"
        )
    residual = np.abs(pred - target)
    residual[~mask] = np.nan
    return residual


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def summarize_rope_residuals(
    base_residual: np.ndarray,
    refined_residual: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Residual-closure summary: did the correction actually satisfy the rope?

    closure_frac = 1 - refined/base (1.0 means the rope constraint is fully
    closed, 0.0 means untouched, negative means the correction moved away).
    """
    base = np.asarray(base_residual, dtype=np.float64)
    refined = np.asarray(refined_residual, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    base = np.where(mask, base, np.nan)
    refined = np.where(mask, refined, np.nan)

    base_mean = _finite_mean(base)
    refined_mean = _finite_mean(refined)
    closure = 1.0 - refined_mean / base_mean if base_mean and np.isfinite(base_mean) and base_mean > 0 else float("nan")

    with np.errstate(invalid="ignore"):
        improved = refined < base
    finger_stats = {}
    for finger_idx, finger in enumerate(FINGER_ORDER):
        finger_stats[finger] = {
            "base_mean_abs": _finite_mean(base[:, finger_idx]),
            "refined_mean_abs": _finite_mean(refined[:, finger_idx]),
        }

    valid_entries = np.isfinite(base) & np.isfinite(refined)
    return {
        "base": {"mean_abs": base_mean, "median_abs": _finite_median(base)},
        "refined": {"mean_abs": refined_mean, "median_abs": _finite_median(refined)},
        "closure_frac": closure,
        "frac_fingers_improved": float(improved[valid_entries].mean()) if valid_entries.any() else float("nan"),
        "per_finger": finger_stats,
        "num_valid_fingers": int(valid_entries.sum()),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if denom <= 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    return pearson(_average_ranks(x), _average_ranks(y))


def quantile_bucket_edges(values: np.ndarray, num_buckets: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite values to bucket")
    quantiles = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    return np.quantile(finite, quantiles)


def bucket_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bucket index per value (NaN -> -1)."""
    values = np.asarray(values, dtype=np.float64)
    idx = np.searchsorted(edges, values, side="right").astype(np.int64)
    idx[~np.isfinite(values)] = -1
    return idx
