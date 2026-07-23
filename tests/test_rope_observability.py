import hashlib

import numpy as np

from ropetrack.refine.rope_observability import (
    balanced_rows,
    representation,
    verify_external_hashes,
)


def test_balanced_rows_round_robins_episodes():
    episodes = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    rows = balanced_rows(episodes, np.arange(9), 6, 7)
    assert len(rows) == 6
    assert set(episodes[rows]) == {"a", "b", "c"}


def test_combined_representation_gives_blocks_equal_scale():
    reference = {
        "base": np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        "rope": np.asarray([[0.0], [2.0]], dtype=np.float32),
    }
    query = {
        "base": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "rope": np.asarray([[1.0]], dtype=np.float32),
    }
    query_rep, reference_rep = representation(query, reference, "base_rope")
    assert query_rep.shape == (1, 3)
    assert reference_rep.shape == (2, 3)
    np.testing.assert_allclose(query_rep, 0.0, atol=1e-7)


def test_external_inputs_are_hash_locked(tmp_path):
    source = tmp_path / "source.bin"
    manifest = tmp_path / "manifest.jsonl"
    source.write_bytes(b"source")
    manifest.write_bytes(b"manifest")
    protocol = {
        "external": {
            "anchor": {
                "episode_manifest": str(manifest),
                "source": {"cache": str(source)},
                "sha256": {
                    "episode_manifest": hashlib.sha256(b"manifest").hexdigest(),
                    "cache": hashlib.sha256(b"source").hexdigest(),
                },
            }
        }
    }
    assert verify_external_hashes(protocol)["anchor"]["cache"] == hashlib.sha256(
        b"source"
    ).hexdigest()
