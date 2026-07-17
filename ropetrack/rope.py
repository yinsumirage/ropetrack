from __future__ import annotations

from math import isfinite, sqrt
from typing import Any

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# shared per-finger palette (thumb..pinky), kept next to FINGER_ORDER so the
# color<->finger pairing cannot drift between visualization scripts
FINGER_COLORS = ("#d14b45", "#2f7ed8", "#20845a", "#b77716", "#7b4bd1")

FINGER_CHAINS = {
    "freihand": (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    ),
    "ho3d": (
        (0, 13, 14, 15, 16),
        (0, 1, 2, 3, 17),
        (0, 4, 5, 6, 18),
        (0, 10, 11, 12, 19),
        (0, 7, 8, 9, 20),
    ),
    "egodex": (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    ),
}
FINGER_CHAINS["arctic"] = FINGER_CHAINS["freihand"]
FINGER_CHAINS["hot3d"] = FINGER_CHAINS["freihand"]


def canonical_rope_dataset(dataset: str) -> str:
    name = dataset.lower()
    if name in {"ho3d", "ho3d_v2", "ho3d_v3"}:
        return "ho3d"
    if name == "freihand":
        return "freihand"
    if name in {"egodex", "arctic", "hot3d"}:
        return name
    raise ValueError(f"unsupported rope dataset: {dataset}")


def point_distance(a, b) -> float | None:
    vals = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    if not all(isfinite(v) for v in vals):
        return None
    return sqrt(sum(v * v for v in vals))


def chain_length(joints, chain: tuple[int, ...]) -> float | None:
    total = 0.0
    for left, right in zip(chain, chain[1:], strict=False):
        dist = point_distance(joints[left], joints[right])
        if dist is None:
            return None
        total += dist
    return total


def rope_distances_for_joints(dataset: str, joints) -> list[float | None]:
    chains = FINGER_CHAINS[canonical_rope_dataset(dataset)]
    if len(joints) < 21:
        raise ValueError(f"expected at least 21 joints, got {len(joints)}")
    return [point_distance(joints[chain[0]], joints[chain[-1]]) for chain in chains]


def normalize_rope_distance(
    distance: float | None, chain_m: float | None, fist_ratio: float = 0.5, clamp: bool = True
) -> float | None:
    if distance is None or chain_m is None or chain_m <= 1e-9:
        return None
    lmin = float(fist_ratio) * chain_m
    denom = max(chain_m - lmin, 1e-9)
    value = (distance - lmin) / denom
    return max(0.0, min(1.0, value)) if clamp else value


def pred_rope_norm_for_dataset(dataset: str, joints, chain_m, valid, fist_ratio: float, clamp: bool = True) -> list[float]:
    """Normalized rope values for predicted joints against label chain lengths.

    Invalid or chain-less fingers become 0.0 (dense output, mirrors the label
    convention); ``clamp=False`` matches the optimizer's unclamped objective.
    """
    distances = rope_distances_for_joints(dataset, joints)
    return [
        float(normalize_rope_distance(distance, chain, fist_ratio=fist_ratio, clamp=clamp)) if is_valid and chain is not None else 0.0
        for distance, chain, is_valid in zip(distances, chain_m, valid, strict=True)
    ]


def rope_values_for_joints(dataset: str, joints, fist_ratio: float = 0.5) -> dict[str, list[Any]]:
    chains = FINGER_CHAINS[canonical_rope_dataset(dataset)]
    if len(joints) < 21:
        raise ValueError(f"expected at least 21 joints, got {len(joints)}")

    rope_dist, rope_chain, rope_norm, rope_valid = [], [], [], []
    for chain in chains:
        dist = point_distance(joints[chain[0]], joints[chain[-1]])
        length = chain_length(joints, chain)
        valid = dist is not None and length is not None and length > 1e-9
        if not valid:
            rope_dist.append(None)
            rope_chain.append(None)
            rope_norm.append(None)
            rope_valid.append(False)
            continue

        rope_dist.append(float(dist))
        rope_chain.append(float(length))
        rope_norm.append(float(normalize_rope_distance(dist, length, fist_ratio=fist_ratio)))
        rope_valid.append(True)

    return {
        "rope_dist_m": rope_dist,
        "rope_chain_m": rope_chain,
        "rope_norm": rope_norm,
        "rope_valid": rope_valid,
    }


def build_rope_row(dataset: str, sample_id: str, joints, fist_ratio: float = 0.5, source: str = "gt_joints") -> dict:
    values = rope_values_for_joints(dataset, joints, fist_ratio=fist_ratio)
    return {
        "sample_id": sample_id,
        "dataset": canonical_rope_dataset(dataset),
        "finger_order": FINGER_ORDER,
        "source": source,
        "normalization": {
            "mode": "chain_length_fist_ratio",
            "fist_ratio": float(fist_ratio),
        },
        **values,
    }
