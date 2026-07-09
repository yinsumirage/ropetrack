"""Causal sequence primitives for temporal rope refinement."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import operator

import numpy as np

from ropetrack.refine.alpha_student import STUDENT_FEATURE_DIM, features_from_cache


@dataclass(frozen=True)
class SequenceSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]


def sequence_frame(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).replace("\\", "/").split("/")
    if len(parts) < 2 or not parts[-2] or not parts[-1].isdigit():
        raise ValueError(f"invalid temporal sample_id: {sample_id}")
    return parts[-2], int(parts[-1])


def deterministic_sequence_split(
    sample_ids, val_fraction: float, seed: int
) -> SequenceSplit:
    fraction = float(val_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1")

    sequences = np.asarray([sequence_frame(sample_id)[0] for sample_id in sample_ids])
    names = sorted(
        set(sequences.tolist()),
        key=lambda name: (
            hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
            name,
        ),
    )
    n_val = max(1, math.ceil(len(names) * fraction))
    if n_val >= len(names):
        raise ValueError("sequence split must leave at least one training sequence")

    val_sequences = tuple(names[:n_val])
    train_sequences = tuple(names[n_val:])
    return SequenceSplit(
        train_idx=np.flatnonzero(np.isin(sequences, train_sequences)),
        val_idx=np.flatnonzero(np.isin(sequences, val_sequences)),
        train_sequences=train_sequences,
        val_sequences=val_sequences,
    )


def _positive_int(name: str, value: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _contiguous_segments(sample_ids, raw_frame_step: int):
    ids = np.asarray(sample_ids)
    if ids.ndim != 1:
        raise ValueError(f"sample_ids must be one-dimensional, got {ids.shape}")

    parsed = [sequence_frame(sample_id) for sample_id in ids]
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate temporal (sequence, frame) rows")

    grouped: dict[str, list[tuple[int, int]]] = {}
    for row, (sequence, frame) in enumerate(parsed):
        grouped.setdefault(sequence, []).append((frame, row))

    segments: list[list[tuple[int, int]]] = []
    for sequence in sorted(grouped):
        rows = sorted(grouped[sequence])
        start = 0
        for index in range(1, len(rows)):
            if rows[index][0] - rows[index - 1][0] != raw_frame_step:
                segments.append(rows[start:index])
                start = index
        segments.append(rows[start:])
    return ids, segments


def build_causal_windows(
    sample_ids,
    features,
    history_length: int,
    raw_frame_step: int,
    history_step: int,
):
    history_length = _positive_int("history_length", history_length)
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    history_step = _positive_int("history_step", history_step)
    if history_step % raw_frame_step:
        raise ValueError("history_step must be divisible by raw_frame_step")

    ids, segments = _contiguous_segments(sample_ids, raw_frame_step)
    feature_rows = np.asarray(features)
    if feature_rows.ndim == 0 or len(feature_rows) != len(ids):
        rows = 0 if feature_rows.ndim == 0 else len(feature_rows)
        raise ValueError(f"feature/sample row mismatch: {rows} != {len(ids)}")

    windows = np.zeros(
        (len(ids), history_length, *feature_rows.shape[1:]), dtype=feature_rows.dtype
    )
    valid = np.zeros((len(ids), history_length), dtype=bool)

    for segment in segments:
        row_by_frame = dict(segment)
        for frame, output_row in segment:
            source_rows = [
                row_by_frame[target]
                for offset in range(history_length - 1, -1, -1)
                if (target := frame - offset * history_step) in row_by_frame
            ]
            windows[output_row, : len(source_rows)] = feature_rows[source_rows]
            valid[output_row, : len(source_rows)] = True

    return windows, valid, valid.sum(axis=1).astype(np.int64)


def temporal_features(cache: dict[str, np.ndarray], raw_frame_step: int = 1) -> np.ndarray:
    base = np.asarray(features_from_cache(cache), dtype=np.float32)
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    ids, segments = _contiguous_segments(cache["sample_id"], raw_frame_step)
    input_rope = np.asarray(cache["input_rope_norm"], dtype=np.float32)

    if base.shape != (len(ids), STUDENT_FEATURE_DIM):
        raise ValueError(
            f"base feature/sample shape mismatch: {base.shape} != "
            f"({len(ids)}, {STUDENT_FEATURE_DIM})"
        )
    if input_rope.shape != (len(ids), 5):
        raise ValueError(f"input_rope_norm must be {(len(ids), 5)}, got {input_rope.shape}")

    delta = np.zeros((len(ids), 5), dtype=np.float32)
    for segment in segments:
        for (_, previous_row), (_, current_row) in zip(segment, segment[1:]):
            delta[current_row] = input_rope[current_row] - input_rope[previous_row]

    return np.concatenate([base, delta], axis=1).astype(np.float32, copy=False)
