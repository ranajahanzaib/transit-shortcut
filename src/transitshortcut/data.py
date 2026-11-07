"""Revision-pinned, column-projected remote Parquet sampling."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "bingbangboom/exoplanet-transit-detection"
LENGTHS = {
    "flux_raw": 20000,
    "flux_global": 201,
    "flux_local": 81,
    "flux_odd": 201,
    "flux_even": 201,
}
CLASSES = ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"]
PARAMS = ["period_days", "epoch_bjd", "duration_hrs", "depth_ppm", "planet_radius_earth"]
FLAGS = ["fp_flag_not_transit", "fp_flag_stellar", "fp_flag_centroid", "fp_flag_ephemeris"]


def root() -> Path:
    return Path(os.environ.get("TRANSITSHORTCUT_HOME", "."))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def decode(value: object, field: str) -> np.ndarray:
    if field not in LENGTHS or not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{field}: expected binary little-endian float32 array")
    if len(value) != LENGTHS[field] * 4:
        raise ValueError(f"{field}: expected {LENGTHS[field] * 4} bytes, got {len(value)}")
    array = np.frombuffer(value, dtype="<f4")
    if not np.isfinite(array).any():
        raise ValueError(f"{field}: no finite observations")
    return array


def sample_indices(meta: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Proportional allocation with one per stratum when budget permits."""
    if n < 1:
        raise ValueError("Sample sizes must be positive")
    n = min(n, len(meta))
    groups = list(meta.groupby(["mission", "disposition"], sort=True).indices.values())
    sizes = np.array([len(g) for g in groups])
    allocation = np.zeros(len(groups), dtype=int)
    if n >= len(groups):
        allocation[:] = 1
    for _ in range(n - allocation.sum()):
        deficit = n * sizes / sizes.sum() - allocation
        deficit[allocation >= sizes] = -np.inf
        allocation[int(np.argmax(deficit))] += 1
    rng = np.random.default_rng(seed)
    return sorted(int(i) for g, k in zip(groups, allocation) for i in rng.choice(g, k, False))


def grouped_sample(
    meta: pd.DataFrame, n: int, seed: int, group_sizes: list[int]
) -> tuple[list[int], list[int]]:
    """Select seeded row-group clusters, then stratify within their candidate pool.

    This is a cluster sample, not a uniform row sample. Record its candidate pool.
    """
    desired = meta.iloc[sample_indices(meta, n, seed)].groupby(["mission", "disposition"]).size()
    offsets = np.r_[0, np.cumsum(group_sizes)]
    rng = np.random.default_rng(seed)
    candidates: list[int] = []
    selected_groups: list[int] = []
    for group in rng.permutation(len(group_sizes)):
        selected_groups.append(int(group))
        candidates.extend(range(int(offsets[group]), int(offsets[group + 1])))
        counts = meta.iloc[candidates].groupby(["mission", "disposition"]).size()
        if all(counts.get(k, 0) >= count for k, count in desired.items()):
            break
    pool = meta.iloc[sorted(candidates)]
    chosen = []
    for key, count in desired.items():
        eligible = pool[(pool.mission == key[0]) & (pool.disposition == key[1])].index.to_numpy()
        chosen.extend(int(i) for i in rng.choice(eligible, int(count), replace=False))
    return sorted(chosen), sorted(selected_groups)


def validate(frame: pd.DataFrame) -> dict:
    required = {
        "star_id",
        "mission",
        "cadence",
        "disposition",
        "has_transit_params",
        "split",
        *PARAMS,
        *FLAGS,
        *LENGTHS,
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Required columns absent: {sorted(missing)}")
    if not set(frame.disposition) <= set(CLASSES):
        raise ValueError("Unrecognized dispositions")
    if frame.star_id.isna().any() or frame.star_id.duplicated().any():
        duplicates = frame.loc[frame.star_id.duplicated(False), ["star_id", "split"]]
        raise ValueError(
            f"Duplicate/cross-partition star IDs: {duplicates.to_dict('records')[:20]}"
        )
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("All three original splits are required")
    nonfinite = {}
    for field in LENGTHS:
        arrays = [decode(v, field) for v in frame[field]]
        nonfinite[field] = sum(int((~np.isfinite(a)).sum()) for a in arrays)
    return {"duplicate_ids": 0, "cross_split_ids": 0, "nonfinite_values": nonfinite}


def inventory(frame: pd.DataFrame) -> dict:
    patterns = pd.DataFrame({"mission": frame.mission, "disposition": frame.disposition})
    patterns["missing_transit"] = frame[PARAMS[:3]].isna().any(axis=1).astype(int)
    patterns["zero_folded"] = [
        int(all(np.all(decode(r[f], f) == 0) for f in LENGTHS if f != "flux_raw"))
        for r in frame.to_dict("records")
    ]
    patterns["missing_flags"] = frame[FLAGS].isna().all(axis=1).astype(int)
    groups = []
    for (mission, label), g in patterns.groupby(["mission", "disposition"]):
        groups.append(
            {
                "mission": mission,
                "class": label,
                "n": len(g),
                **{c: float(g[c].mean()) for c in patterns.columns[2:]},
            }
        )
    return {
        "classes": frame.disposition.value_counts().to_dict(),
        "missions": frame.mission.value_counts().to_dict(),
        "groups": groups,
        "artifact_counts": {c: int(patterns[c].sum()) for c in patterns.columns[2:]},
    }


def prepare(
    mode: str, seed: int, sizes: dict[str, int], revision: str = "main", max_flux_mb: int = 512
) -> dict:
    from huggingface_hub import HfApi, HfFileSystem

    if mode not in {"sample", "full"}:
        raise ValueError("mode must be sample or full")
    sha = HfApi().dataset_info(REPO, revision=revision).sha
    config = {
        "mode": mode,
        "seed": seed,
        "requested_sizes": sizes,
        "revision": sha,
        "max_flux_mb": max_flux_mb,
        "sampler": "seeded-row-group-clusters-stratified-v2",
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = root() / "data" / key
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if digest(folder / "snapshot.parquet") != manifest["snapshot_sha256"]:
            raise ValueError("Cached snapshot checksum mismatch")
        write_json(root() / "data/current.json", {"dataset_id": key})
        return manifest
    fs = HfFileSystem(block_size=64 * 1024)
    frames, inspected, selected, transfer = [], {}, {}, 0
    candidate_groups = {}
    source_duplicates: dict[str, str] = {}
    for split, filename in [
        ("train", "train.parquet"),
        ("validation", "val.parquet"),
        ("test", "test.parquet"),
    ]:
        with fs.open(f"datasets/{REPO}@{sha}/{filename}", "rb", block_size=64 * 1024) as stream:
            parquet = pq.ParquetFile(stream)
            columns = ["star_id", "mission", "disposition"]
            required = {*columns, "cadence", "has_transit_params", *PARAMS, *FLAGS, *LENGTHS}
            if missing := required - set(parquet.schema_arrow.names):
                raise ValueError(f"Remote schema missing columns: {sorted(missing)}")
            print(f"{split}: discovering {parquet.metadata.num_rows} metadata rows", flush=True)
            meta = parquet.read(columns=columns).to_pandas()
            inspected[split] = len(meta)
            for star in meta.star_id:
                if pd.isna(star) or star in source_duplicates:
                    raise ValueError(f"Duplicate source star ID {star}; split isolation unsafe")
                source_duplicates[star] = split
            if mode == "full":
                indices = list(range(len(meta)))
                candidate_groups[split] = list(range(parquet.num_row_groups))
            else:
                indices, candidate_groups[split] = grouped_sample(
                    meta,
                    sizes[split],
                    seed,
                    [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)],
                )
            selected[split] = indices
            offset = 0
            blocks = []
            for group in range(parquet.num_row_groups):
                rg = parquet.metadata.row_group(group)
                local = [i - offset for i in indices if offset <= i < offset + rg.num_rows]
                if local:
                    cost = sum(
                        rg.column(j).total_compressed_size
                        for j in range(rg.num_columns)
                        if rg.column(j).path_in_schema in LENGTHS
                    )
                    transfer += cost
                    if mode == "sample" and transfer > max_flux_mb * 1024**2:
                        raise ValueError(
                            f"Selected row groups require >{max_flux_mb} MiB compressed flux. "
                            "No full-download fallback. Increase --max-flux-mb explicitly, "
                            "reduce sample sizes, or choose --mode full."
                        )
                    print(
                        f"{split}: row group {group}, {len(local)} selected; flux budget used {transfer / 1024**2:.1f} MiB",
                        flush=True,
                    )
                    block = parquet.read_row_group(group).take(pa.array(local)).to_pandas()
                    block["source_row"] = [i + offset for i in local]
                    blocks.append(block)
                offset += rg.num_rows
            part = pd.concat(blocks, ignore_index=True)
            part["split"] = split
            part["target_id"] = [f"{split}-{i}" for i in part.source_row]
            frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    quality = validate(frame)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "snapshot.parquet", index=False, row_group_size=128)
    manifest = {
        "dataset_id": key,
        "repository": REPO,
        "revision": sha,
        "prepared_at": datetime.now(UTC).isoformat(),
        "config": config,
        "inspected_rows": inspected,
        "selected_rows": selected,
        "candidate_row_groups": candidate_groups,
        "sampling_limitation": "Seeded row-group cluster sample, stratified to whole-split mission/class allocation. Within-stratum representativeness is not guaranteed.",
        "selected_identifiers": frame[["target_id", "star_id", "split"]].to_dict("records"),
        "actual_sizes": frame.split.value_counts().to_dict(),
        "quality": quality,
        "compressed_flux_row_group_bytes": transfer,
        "summary": inventory(frame),
        "snapshot_sha256": digest(folder / "snapshot.parquet"),
    }
    write_json(manifest_path, manifest)
    write_json(root() / "data/current.json", {"dataset_id": key})
    return manifest


def load_manifest() -> dict:
    pointer = root() / "data/current.json"
    if not pointer.exists():
        raise FileNotFoundError("Dataset not prepared. Run transitshortcut data prepare.")
    key = json.loads(pointer.read_text())["dataset_id"]
    folder = root() / "data" / key
    return json.loads((folder / "manifest.json").read_text())


@lru_cache(maxsize=1)
def _snapshot(path: str, mtime_ns: int, size: int, checksum: str) -> pd.DataFrame:
    """Cache immutable snapshot data, never trained model state."""
    file = Path(path)
    if digest(file) != checksum:
        raise ValueError("Dataset checksum mismatch")
    return pd.read_parquet(file)


def load() -> tuple[pd.DataFrame, dict]:
    manifest = load_manifest()
    path = root() / "data" / manifest["dataset_id"] / "snapshot.parquet"
    stat = path.stat()
    frame = _snapshot(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size, manifest["snapshot_sha256"]
    )
    return frame, manifest
