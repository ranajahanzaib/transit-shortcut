"""Two explicit feature allowlists; raw features never inspect metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .data import FLAGS, LENGTHS, PARAMS, decode

GROUPS = {
    "Mission/cadence": ["mission", "cadence"],
    "Transit availability": ["has_transit_params", *[f"missing_{p}" for p in PARAMS]],
    "Folded zeros": [
        f"{f}_{suffix}" for f in LENGTHS if f != "flux_raw" for suffix in ("zero", "zero_fraction")
    ],
    "Flag missingness": [f"missing_{f}" for f in FLAGS],
}


def shortcut(row: dict) -> dict:
    out = {
        "mission": str(row["mission"]),
        "cadence": str(row["cadence"]),
        "has_transit_params": int(bool(row["has_transit_params"])),
    }
    out.update({f"missing_{f}": int(pd.isna(row.get(f))) for f in PARAMS + FLAGS})
    for f in LENGTHS:
        if f == "flux_raw":
            continue
        a = decode(row[f], f)
        out[f"{f}_zero"] = int(np.all(a == 0))
        out[f"{f}_zero_fraction"] = float(np.mean(a == 0))
    return out


def signal(raw: object) -> dict[str, float]:
    x = decode(raw, "flux_raw").astype(float)
    finite = np.isfinite(x)
    missing = float(1 - finite.mean())
    x = np.where(finite, x, np.nanmedian(x[finite]))
    x -= np.median(x)
    mad = float(np.median(np.abs(x)))
    scale = max(mad * 1.4826, 1e-8)
    z = np.clip(x / scale, -100, 100)
    local = z - median_filter(z, size=101)
    dips = local < -2
    edges = np.diff(np.r_[False, dips, False].astype(int))
    widths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    out = {
        "std": float(x.std()),
        "mad": mad,
        "missing_fraction": missing,
        "negative_tail": float(-np.quantile(z, 0.01)),
        "asymmetry": float(np.quantile(z, 0.99) + np.quantile(z, 0.01)),
        "dip_fraction": float(dips.mean()),
        "dip_width_median": float(np.median(widths)) if len(widths) else 0.0,
        "local_contrast": float(-np.quantile(local, 0.01)),
    }
    for q in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99):
        out[f"quantile_{q}"] = float(np.quantile(x, q))
    for w in (5, 21, 51):
        out[f"window_dip_{w}"] = float(-np.min(np.convolve(z, np.ones(w) / w, "valid")))
    for lag in (1, 10, 50, 200):
        a, b = z[:-lag], z[lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        out[f"autocorr_{lag}"] = float(a @ b / denom) if denom > 1e-12 else 0.0
    return out


def extract(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.to_dict("records")
    return (
        pd.DataFrame([shortcut(r) for r in rows]),
        pd.DataFrame([signal(r["flux_raw"]) for r in rows]),
    )
