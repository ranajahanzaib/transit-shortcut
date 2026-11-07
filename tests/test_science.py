import joblib
import numpy as np
import pandas as pd
import pytest

from transitshortcut.data import decode, inventory, sample_indices, validate
from transitshortcut.experiments import metrics, pipeline, read_predictions, train
from transitshortcut.features import GROUPS, extract, shortcut, signal


def test_binary_decode_and_malformed():
    a = np.arange(20000, dtype="<f4")
    np.testing.assert_array_equal(decode(a.tobytes(), "flux_raw"), a)
    for bad in (b"bad", a, None, b"\x00" * 8):
        with pytest.raises(ValueError):
            decode(bad, "flux_raw")
    with pytest.raises(ValueError):
        decode(np.full(20000, np.nan, dtype="<f4").tobytes(), "flux_raw")


def test_sampler_is_deterministic_and_preserves_groups():
    meta = pd.DataFrame(
        {
            "mission": ["kepler"] * 90 + ["tess"] * 10,
            "disposition": ["PLANET"] * 50 + ["NO_SIGNAL"] * 40 + ["PLANET"] * 10,
        }
    )
    a = sample_indices(meta, 20, 42)
    assert a == sample_indices(meta, 20, 42)
    assert len(set(a)) == 20
    assert set(meta.iloc[a].mission) == {"kepler", "tess"}
    assert a != sample_indices(meta, 20, 41)


def test_quality_and_nonexclusive_zero_artifact(cohort):
    f, _, _ = cohort
    assert validate(f)["cross_split_ids"] == 0
    s = inventory(f)
    assert s["artifact_counts"]["zero_folded"] > s["classes"]["NO_SIGNAL"]
    row = f.iloc[0].to_dict()
    assert shortcut(row)["flux_global_zero"] == 1
    assert row["disposition"] == "PLANET"
    row.pop("period_days")
    assert shortcut(row)["missing_period_days"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([f, f.iloc[:1]]))
    with pytest.raises(ValueError, match="Required"):
        validate(f.drop(columns="mission"))


def test_feature_allowlists_and_raw_only(cohort):
    f, _, _ = cohort
    row = f.iloc[0].to_dict()
    a = shortcut(row)
    assert set(a) == {c for columns in GROUPS.values() for c in columns}
    before = signal(row["flux_raw"])
    row.update(disposition="NO_SIGNAL", confidence_score=1, mission="tess")
    assert signal(row["flux_raw"]) == before
    assert not {"disposition", "confidence_score", "label_source", "star_id"} & set(a)
    for raw in (
        np.ones(20000),
        np.zeros(20000),
        np.r_[np.nan, np.ones(19999)],
        np.r_[-1e6, np.ones(19999)],
    ):
        values = signal(raw.astype("<f4").tobytes())
        assert np.isfinite(list(values.values())).all()
        assert values == signal(raw.astype("<f4").tobytes())


def test_train_isolation_serialization_and_ablation(cohort):
    f, _, path = cohort
    result = train("multiclass")
    assert result["models"]["shortcut"]["test"]["accuracy"] > 0.6
    pred = read_predictions(result)
    assert len(pred) == 30
    assert set(pred.split) == {"test"}
    assert result["classes"] == ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"]
    a, _ = extract(f)
    model = joblib.load(path / "artifacts" / result["id"] / "shortcut.joblib")
    assert len(model.predict(a)) == len(f)
    # Encoders are train-fitted: a category appearing only in test remains unseen.
    f.loc[f.split == "test", "mission"] = "unseen"
    a.loc[f.split == "test", "mission"] = "unseen"
    m = pipeline(a, 7)
    m.fit(a[f.split == "train"], f.loc[f.split == "train", "disposition"])
    encoder = m.named_steps["preprocess"].named_transformers_["categories"]
    assert "unseen" not in encoder.categories_[0]
    assert len(m.predict(a[f.split == "test"])) == 30
    binary = train("cross-mission")
    assert binary["classes"] == ["PLANET", "FALSE_POSITIVE"]
    assert not binary["models"]["signal"]["missions"]["tess"]["available"]


def test_missing_class_metrics():
    m = metrics(["PLANET"], ["PLANET"], ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"])
    assert m["per_class"]["NO_SIGNAL"]["recall"] is None
    assert m["balanced_accuracy"] == 1
    assert m["warning"]


def test_positive_and_negative_artifact_reliance():
    # Only folded-zero group is informative; the test needs refit sensitivity, not causal claims.
    n = 120
    y = np.array(["PLANET", "FALSE_POSITIVE"] * (n // 2))
    x = pd.DataFrame(
        {
            "mission": ["kepler"] * n,
            "cadence": ["long"] * n,
            "flux_global_zero": (y == "PLANET").astype(int),
            "constant": np.ones(n),
        }
    )
    m = pipeline(x, 7).fit(x.iloc[:80], y[:80])
    reduced = x.drop(columns="flux_global_zero")
    ablated = pipeline(reduced, 7).fit(reduced.iloc[:80], y[:80])
    assert metrics(y[80:], m.predict(x.iloc[80:]), ["PLANET", "FALSE_POSITIVE"])["accuracy"] == 1
    assert (
        metrics(y[80:], ablated.predict(reduced.iloc[80:]), ["PLANET", "FALSE_POSITIVE"])[
            "accuracy"
        ]
        == 0.5
    )
    # Constant suspicious feature: no apparent reliance after removing it.
    x["flux_global_zero"] = 0
    neutral = pipeline(x, 7).fit(x.iloc[:80], y[:80])
    assert np.array_equal(neutral.predict(x.iloc[80:]), ablated.predict(reduced.iloc[80:]))


def test_group_sampler_preserves_whole_split_allocation():
    from transitshortcut.data import grouped_sample

    meta = pd.DataFrame(
        {"mission": ["kepler"] * 1000, "disposition": ["PLANET", "NO_SIGNAL"] * 500}
    )
    selected, groups = grouped_sample(meta, 100, 42, [200] * 5)
    assert len(groups) == 1
    assert selected == grouped_sample(meta, 100, 42, [200] * 5)[0]
    assert meta.iloc[selected].disposition.value_counts().to_dict() == {
        "PLANET": 50,
        "NO_SIGNAL": 50,
    }


def test_raw_downsampling_preserves_deep_dip(cohort):
    from transitshortcut.api import flux

    # Cached immutable snapshot is shared for read-only requests; temporarily inject a dip for this test.
    from transitshortcut.data import load

    cached, _ = load()
    raw = np.ones(20000, dtype="<f4")
    raw[12345] = 0.8
    cached.loc[cached.target_id == "test-0", "flux_raw"] = raw.tobytes()
    result = flux("test-0", "flux_raw", 100)["data"]
    assert 12345 in result["x"]
    assert min(result["y"]) < 0.81
