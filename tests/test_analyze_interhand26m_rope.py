import numpy as np

from scripts.analyze_interhand26m_rope import (
    align_cache, bbox_iou, pair_geometry, quantile_labels, quantile_representatives,
)


def test_pair_geometry_tracks_overlap_and_other_hand_inside_crop():
    rows = [
        {
            "frame_group_id": "frame", "mano_side": side, "bbox_xyxy": box,
            "joint_valid": [1] * 21, "intrinsic": np.eye(3).tolist(),
        }
        for side, box in (("left", [0, 0, 2, 2]), ("right", [1, 0, 3, 2]))
    ]
    xyz = np.ones((2, 21, 3), dtype=np.float64)
    xyz[0, :, :2] = [1.5, 1.0]
    xyz[1, :, :2] = [1.0, 1.0]
    result = pair_geometry(rows, xyz)
    np.testing.assert_allclose(result["bbox_iou"], [1 / 3, 1 / 3])
    np.testing.assert_allclose(result["other_joint_inside_fraction"], [1, 1])
    assert bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0


def test_quantile_labels_keep_missing_values_explicit():
    labels, edges = quantile_labels(np.asarray([0.0, 1.0, 2.0, 3.0, np.nan]), buckets=2)
    assert labels.tolist() == ["Q1", "Q1", "Q2", "Q2", "missing"]
    assert edges == [1.5]


def test_align_cache_leaves_global_finger_order_alone(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez(
        path,
        sample_id=np.asarray(["b", "a"]),
        base_rope_norm=np.asarray([[2], [1]]),
        finger_order=np.asarray(["thumb", "index", "middle", "ring", "pinky"]),
    )
    cache = align_cache(path, ["a", "b"])
    np.testing.assert_array_equal(cache["base_rope_norm"], [[1], [2]])
    np.testing.assert_array_equal(cache["finger_order"], ["thumb", "index", "middle", "ring", "pinky"])


def test_quantile_representatives_return_observed_ranked_samples():
    values = np.asarray([9.0, 1.0, 5.0, 3.0, 7.0])
    assert quantile_representatives(np.arange(5), values, (0.0, 0.5, 1.0)) == [1, 2, 0]
