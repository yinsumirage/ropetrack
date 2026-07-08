"""Cache IO and sample-id alignment helpers for the refine pipeline.

Everything joins by sample_id with loud failure (CLAUDE.md rule 7);
positional joins are forbidden in alignment-critical paths.
"""

from __future__ import annotations

import numpy as np

from ropetrack.io import load_pred_json, read_json


def validate_cache(cache: dict[str, np.ndarray]) -> None:
    """Shape/key contract of a refiner eval cache (built by apply_rope_refinement)."""
    missing = [key for key in ("sample_id", "base_hand_pose", "base_rope_norm", "rope_valid") if key not in cache]
    if "input_rope_norm" not in cache and "gt_rope_norm" not in cache:
        missing.append("input_rope_norm or gt_rope_norm")
    if missing:
        raise ValueError(f"cache missing required keys: {', '.join(missing)}")

    expected = len(cache["sample_id"])
    sample_keys = ["base_hand_pose", "base_rope_norm", "rope_valid", "input_rope_norm", "gt_rope_norm"]
    bad = {key: len(cache[key]) for key in sample_keys if key in cache and len(cache[key]) != expected}
    if bad:
        raise ValueError(f"cache first dimension mismatch: expected {expected}, got {bad}")

    shapes = {
        "base_hand_pose": (expected, 45),
        "base_rope_norm": (expected, 5),
        "rope_valid": (expected, 5),
    }
    if "input_rope_norm" in cache:
        shapes["input_rope_norm"] = (expected, 5)
    if "gt_rope_norm" in cache:
        shapes["gt_rope_norm"] = (expected, 5)
    for key, shape in shapes.items():
        actual = tuple(np.shape(cache[key]))
        if actual != shape:
            raise ValueError(f"{key} shape must be {shape}, got {actual}")


def dense_rope(values, valid) -> list[float]:
    """None-free per-finger values (invalid slots become 0.0)."""
    return [float(value) if is_valid and value is not None else 0.0 for value, is_valid in zip(values, valid, strict=True)]


def load_sample_order(run_meta, fallback_ids: list[str]) -> list[str]:
    """Sample order from a run_meta.json, loud on any inconsistency.

    - run_meta is None -> the documented fallback order is returned;
    - run_meta points at a missing file -> FileNotFoundError (a typo must not
      silently change the join order);
    - the file lacks 'sample_order' -> ValueError.
    """
    if run_meta is None:
        return fallback_ids
    from pathlib import Path

    meta_path = Path(run_meta)
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta does not exist: {meta_path}")
    meta = read_json(meta_path)
    if "sample_order" not in meta:
        raise ValueError(f"run_meta missing sample_order: {meta_path}")
    return list(meta["sample_order"])


def load_prediction_joints(pred_dir) -> list:
    """xyz rows from a benchmark export's pred.json."""
    from pathlib import Path

    xyz, _ = load_pred_json(Path(pred_dir) / "pred.json")
    return xyz


def align_rows_by_sample_id(wanted_ids, have_ids) -> np.ndarray:
    """Permutation mapping have_ids rows onto wanted_ids order; loud on gaps."""
    index = {str(sid): i for i, sid in enumerate(have_ids)}
    missing = [str(sid) for sid in wanted_ids if str(sid) not in index]
    if missing:
        raise ValueError(f"rows missing sample_ids: {missing[:5]} (total {len(missing)})")
    return np.asarray([index[str(sid)] for sid in wanted_ids])
