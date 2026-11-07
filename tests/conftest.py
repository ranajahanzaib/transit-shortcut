"""Explicitly synthetic fixtures, isolated from the production dataset cache."""

import numpy as np
import pandas as pd
import pytest

from transitshortcut.data import FLAGS, LENGTHS, PARAMS, digest, inventory, write_json


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSITSHORTCUT_HOME", str(tmp_path))
    rng = np.random.default_rng(7)
    rows = []
    for split, n in [("train", 90), ("validation", 30), ("test", 30)]:
        for i in range(n):
            label = ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"][i % 3]
            missing = label == "NO_SIGNAL" or i == 0
            row = {
                "star_id": f"SYNTHETIC_{split}_{i}",
                "target_id": f"{split}-{i}",
                "split": split,
                "disposition": label,
                "mission": "kepler",
                "cadence": "long",
                "has_transit_params": not missing,
                "label_source": "SYNTHETIC_TEST_ONLY",
            }
            row.update({p: np.nan if missing else 1.0 for p in PARAMS})
            row.update({f: np.nan if missing else 0.0 for f in FLAGS})
            for field, length in LENGTHS.items():
                a = (1 + rng.normal(0, 0.01, length)).astype("<f4")
                if field != "flux_raw" and missing:
                    a[:] = 0
                row[field] = a.tobytes()
            rows.append(row)
    frame = pd.DataFrame(rows)
    folder = tmp_path / "data/synthetic-test"
    folder.mkdir(parents=True)
    frame.to_parquet(folder / "snapshot.parquet", index=False)
    manifest = {
        "dataset_id": "synthetic-test",
        "repository": "SYNTHETIC TEST FIXTURE",
        "revision": "fixture",
        "config": {
            "seed": 7,
            "mode": "sample",
            "revision": "fixture",
            "requested_sizes": {"train": 90, "validation": 30, "test": 30},
            "max_flux_mb": 1,
        },
        "summary": inventory(frame),
        "snapshot_sha256": digest(folder / "snapshot.parquet"),
    }
    write_json(folder / "manifest.json", manifest)
    write_json(tmp_path / "data/current.json", {"dataset_id": "synthetic-test"})
    return frame, manifest, tmp_path
