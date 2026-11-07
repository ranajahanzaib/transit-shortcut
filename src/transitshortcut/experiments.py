"""Train-fitted baselines, isolated evaluation, and refit ablations."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CLASSES, load, load_manifest, root, write_json
from .features import GROUPS, extract

LIMITATIONS = [
    "Metrics describe the selected cohort, not the full dataset or independent observations.",
    "Correlation, refit sensitivity, and physical generalization are different claims.",
    "NO_SIGNAL is documented as Kepler-only; TESS/K2 three-class coverage is incomplete.",
    "Zero-filled folded views can also occur for PLANET and FALSE_POSITIVE.",
    "Raw sample indices are not physical time; features are simple photometric proxies.",
    "Prediction scores are uncalibrated model probability estimates.",
    "This application does not discover or validate planets.",
]


def pipeline(frame: pd.DataFrame, seed: int) -> Pipeline:
    cats = [c for c in ("mission", "cadence") if c in frame]
    nums = [c for c in frame if c not in cats]
    transforms = []
    if cats:
        transforms.append(("categories", OneHotEncoder(handle_unknown="ignore"), cats))
    if nums:
        transforms.append(
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                nums,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transforms)),
            ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]
    )


def metrics(y: list | pd.Series, pred: list | np.ndarray, classes: list[str]) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    support = {c: int(sum(y == c)) for c in classes}
    if len(y) == 0:
        return {"n": 0, "available": False, "support": support}
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    present = [i for i, n in enumerate(s) if n]
    return {
        "n": len(y),
        "available": True,
        "support": support,
        "warning": "Small sample or incomplete class coverage" if min(s) < 5 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(np.mean(r[present])),
        "macro_f1": float(np.mean(f[present])),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "class_order": classes,
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "per_class": {
            c: {
                "precision": float(p[i]) if sum(pred == c) else None,
                "recall": float(r[i]) if s[i] else None,
                "f1": float(f[i]) if s[i] else None,
                "support": int(s[i]),
            }
            for i, c in enumerate(classes)
        },
    }


def train(experiment: str) -> dict:
    if experiment not in {"multiclass", "cross-mission"}:
        raise ValueError("Unknown experiment")
    frame, manifest = load()
    cache = root() / "data" / manifest["dataset_id"] / "features-v1.joblib"
    if cache.exists():
        a, b = joblib.load(cache)
    else:
        a, b = extract(frame)
        joblib.dump((a, b), cache)
    eligible = (
        frame.disposition != "NO_SIGNAL"
        if experiment == "cross-mission"
        else pd.Series(True, index=frame.index)
    )
    classes = CLASSES[:2] if experiment == "cross-mission" else CLASSES
    masks = {s: (frame.split == s) & eligible for s in ("train", "validation", "test")}
    if set(frame.loc[masks["train"], "disposition"]) != set(classes):
        raise ValueError(
            "Training cohort does not contain all task classes; prepare a larger sample"
        )
    seed = manifest["config"]["seed"]
    run_id = f"{experiment}-{uuid.uuid4().hex[:10]}"
    folder = root() / "artifacts" / run_id
    folder.mkdir(parents=True)
    test = frame.loc[masks["test"]].copy()
    predictions = test[["target_id", "star_id", "mission", "disposition", "split"]].copy()
    result = {
        "id": run_id,
        "experiment": experiment,
        "dataset_id": manifest["dataset_id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_manifest": manifest,
        "classes": classes,
        "models": {},
        "ablations": [],
        "limitations": LIMITATIONS,
        "configuration": {
            "seed": seed,
            "C": 1.0,
            "max_iter": 2000,
            "fit_partition": "train",
            "tuning": "none",
        },
    }
    code = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    result["code_version"] = code.stdout.strip() or "unknown"
    source = Path(__file__).parent
    result["source_sha256"] = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(source.glob("*.py")))
    ).hexdigest()
    for name, features in [("shortcut", a), ("signal", b)]:
        model = pipeline(features, seed)
        model.fit(features.loc[masks["train"]], frame.loc[masks["train"], "disposition"])
        joblib.dump(model, folder / f"{name}.joblib")
        pred = model.predict(features.loc[masks["test"]])
        probabilities = model.predict_proba(features.loc[masks["test"]])
        predictions[f"{name}_prediction"] = pred
        for i, c in enumerate(model.classes_):
            predictions[f"{name}_score_{c}"] = probabilities[:, i]
        val_pred = (
            model.predict(features.loc[masks["validation"]]) if masks["validation"].any() else []
        )
        result["models"][name] = {
            "features": list(features),
            "train_n": int(masks["train"].sum()),
            "test": metrics(test.disposition, pred, classes),
            "validation": metrics(frame.loc[masks["validation"], "disposition"], val_pred, classes),
            "missions": {
                m: metrics(
                    test.loc[test.mission == m, "disposition"],
                    pred[(test.mission == m).to_numpy()],
                    classes,
                )
                for m in ("kepler", "k2", "tess")
            },
        }
    base = joblib.load(folder / "shortcut.joblib")
    base_prob = base.predict_proba(a.loc[masks["test"]])
    for i, (group, columns) in enumerate(GROUPS.items()):
        reduced = a.drop(columns=columns)
        model = pipeline(reduced, seed)
        model.fit(reduced.loc[masks["train"]], frame.loc[masks["train"], "disposition"])
        joblib.dump(model, folder / f"ablation-{i}.joblib")
        pred = model.predict(reduced.loc[masks["test"]])
        prob = model.predict_proba(reduced.loc[masks["test"]])
        changed = pred != predictions.shortcut_prediction.to_numpy()
        delta = np.max(np.abs(prob - base_prob), axis=1)
        predictions[f"ablation_{i}_prediction"] = pred
        predictions[f"ablation_{i}_score_delta"] = delta
        for j, c in enumerate(model.classes_):
            predictions[f"ablation_{i}_score_{c}"] = prob[:, j]
        score = metrics(test.disposition, pred, classes)
        result["ablations"].append(
            {
                "index": i,
                "group": group,
                "removed_features": columns,
                "metrics": score,
                "macro_f1_drop": result["models"]["shortcut"]["test"]["macro_f1"]
                - score["macro_f1"],
                "changed_n": int(changed.sum()),
                "changed_fraction": float(changed.mean()),
                "mean_max_score_delta": float(delta.mean()),
                "by_mission": {
                    str(m): {
                        "n": int(sum(test.mission == m)),
                        "changed": int(changed[(test.mission == m).to_numpy()].sum()),
                    }
                    for m in test.mission.unique()
                },
                "by_class": {
                    c: {
                        "n": int(sum(test.disposition == c)),
                        "changed": int(changed[(test.disposition == c).to_numpy()].sum()),
                    }
                    for c in classes
                },
            }
        )
    result["ablations"].sort(key=lambda x: x["macro_f1_drop"], reverse=True)
    result["audit_completed_at"] = datetime.now(UTC).isoformat()
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    result["predictions_sha256"] = hashlib.sha256(
        (folder / "predictions.parquet").read_bytes()
    ).hexdigest()
    write_json(folder / "result.json", result)
    return result


def runs() -> list[dict]:
    manifest = load_manifest()
    return sorted(
        [
            json.loads(p.read_text())
            for p in (root() / "artifacts").glob("*/result.json")
            if json.loads(p.read_text())["dataset_id"] == manifest["dataset_id"]
        ],
        key=lambda r: r["timestamp"],
        reverse=True,
    )


def get_run(run_id: str) -> dict:
    for run in runs():
        if run["id"] == run_id:
            return run
    raise FileNotFoundError("Completed experiment not found for active dataset")


def read_predictions(run: dict) -> pd.DataFrame:
    path = root() / "artifacts" / run["id"] / "predictions.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != run["predictions_sha256"]:
        raise ValueError("Prediction artifact checksum mismatch")
    return pd.read_parquet(path)


def markdown(run: dict) -> str:
    lines = [
        f"# TransitShortcut audit: {run['id']}",
        f"\nTask: {run['experiment']}",
        f"\nCompleted: {run['timestamp']}",
        f"\nCode: {run['code_version']} (working source SHA256 {run['source_sha256']})",
        f"\nDataset: {run['dataset_manifest']['repository']} @ {run['dataset_manifest']['revision']}",
        "\n## Results\n",
        "Model | Test n | Accuracy | Balanced accuracy | Macro F1",
        "---|---:|---:|---:|---:",
    ]
    for name, model in run["models"].items():
        m = model["test"]
        lines.append(
            f"{name} | {m['n']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f}"
        )
    lines += ["\n## Refit sensitivity\n"]
    for a in run["ablations"]:
        lines.append(
            f"- Drop {a['group']}: macro F1 {a['metrics']['macro_f1']:.4f}; {a['changed_n']}/{a['metrics']['n']} changed predictions."
        )
    config = run["dataset_manifest"]["config"]
    sizes = config["requested_sizes"]
    lines += [
        "\n## Reproduce\n",
        "```sh",
        f"uv run transitshortcut data prepare --mode {config['mode']} --seed {config['seed']} --revision {config['revision']} --train-size {sizes['train']} --validation-size {sizes['validation']} --test-size {sizes['test']} --max-flux-mb {config['max_flux_mb']}",
        f"uv run transitshortcut train --experiment {run['experiment']}",
        "```",
        "\n## Limitations\n",
        *[f"- {v}" for v in run["limitations"]],
        "\n## Complete saved evidence\n",
        "```json",
        json.dumps(run, indent=2),
        "```",
    ]
    return "\n".join(lines)
