#!/usr/bin/env python3
"""Audit the HO3D (v3) training split before building the train pipeline.

The training-split extension (hard images, rope labels, GT-bbox export,
teacher generation) rests on format assumptions that only the real data can
confirm. This CPU-only script samples metas across sequences and reports:

- train.txt line count/format and per-sequence frame counts;
- image file extension and presence (v3 uses .jpg, v2 .png);
- meta pkl key sets and shapes: handJoints3D [21, 3], handPose [48],
  handBeta [10], handTrans [3], camMat [3, 3];
- whether handBoundingBox exists in TRAIN metas (evaluation metas have it,
  training metas may not - the export falls back to projected-joint bboxes);
- NaN/degenerate annotation rates in the sample.

Run it on a CPU node and attach the JSON report to the experience note
before any pipeline code relies on these assumptions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ropetrack.refine.analysis import json_sanitize  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
EXPECTED_SHAPES = {
    "handJoints3D": (21, 3),
    "handPose": (48,),
    "handBeta": (10,),
    "handTrans": (3,),
    "camMat": (3, 3),
}


def read_split_ids(split_file: Path) -> list[str]:
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [sid for sid in ids[:1000] if "/" not in sid]
    if bad:
        raise ValueError(f"split ids are not 'SEQ/frame' formatted, e.g. {bad[:3]}")
    return ids


def resolve_image(root: Path, split_dir: str, sample_id: str) -> tuple[Path | None, str | None]:
    seq, frame = sample_id.split("/")
    for ext in IMAGE_EXTENSIONS:
        path = root / split_dir / seq / "rgb" / f"{frame}{ext}"
        if path.is_file():
            return path, ext
    return None, None


def audit_meta(path: Path) -> dict:
    with path.open("rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    report: dict = {"keys": sorted(meta.keys()), "shape_ok": {}, "nan": {}}
    for key, expected in EXPECTED_SHAPES.items():
        if key not in meta or meta[key] is None:
            report["shape_ok"][key] = None
            continue
        value = np.asarray(meta[key], dtype=np.float64)
        report["shape_ok"][key] = bool(value.shape == expected)
        report["nan"][key] = bool(~np.isfinite(value).all())
    report["has_handBoundingBox"] = "handBoundingBox" in meta and meta["handBoundingBox"] is not None
    return report


def run_audit(root: Path, split_dir: str, split_file: Path, sample_count: int) -> dict:
    ids = read_split_ids(split_file)
    seq_counts = Counter(sid.split("/")[0] for sid in ids)
    picks = [ids[i] for i in np.linspace(0, len(ids) - 1, min(sample_count, len(ids)), dtype=int)]

    key_sets = Counter()
    extension_counts = Counter()
    missing_images, missing_metas = [], []
    shape_fail: dict[str, int] = Counter()
    nan_counts: dict[str, int] = Counter()
    bbox_present = 0
    audited = 0

    for sample_id in picks:
        seq, frame = sample_id.split("/")
        image_path, ext = resolve_image(root, split_dir, sample_id)
        if image_path is None:
            missing_images.append(sample_id)
        else:
            extension_counts[ext] += 1
        meta_path = root / split_dir / seq / "meta" / f"{frame}.pkl"
        if not meta_path.is_file():
            missing_metas.append(sample_id)
            continue
        report = audit_meta(meta_path)
        audited += 1
        key_sets[tuple(report["keys"])] += 1
        bbox_present += int(report["has_handBoundingBox"])
        for key, ok in report["shape_ok"].items():
            if ok is not True:
                shape_fail[key] += 1
        for key, is_nan in report["nan"].items():
            if is_nan:
                nan_counts[key] += 1

    return {
        "root": str(root),
        "split_dir": split_dir,
        "num_split_ids": len(ids),
        "num_sequences": len(seq_counts),
        "frames_per_sequence_min_max": [min(seq_counts.values()), max(seq_counts.values())],
        "num_sampled": len(picks),
        "num_metas_audited": audited,
        "image_extensions": dict(extension_counts),
        "missing_images": missing_images[:10],
        "missing_metas": missing_metas[:10],
        "meta_key_sets": [{"keys": list(keys), "count": count} for keys, count in key_sets.most_common(3)],
        "frac_with_handBoundingBox": bbox_present / max(audited, 1),
        "shape_failures": dict(shape_fail),
        "nan_annotation_counts": dict(nan_counts),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an HO3D training split layout and annotations.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--split-dir", default="train", help="Subdirectory holding sequences (HO3D v3 uses 'train').")
    parser.add_argument("--split-file", type=Path, default=None, help="Defaults to <input-root>/train.txt.")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    split_file = args.split_file if args.split_file is not None else args.input_root / "train.txt"
    report = run_audit(args.input_root, args.split_dir, split_file, args.sample_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_sanitize(report), indent=2), encoding="utf-8")
    print(json.dumps(json_sanitize(report), indent=2))
    return args.output


if __name__ == "__main__":
    main()
