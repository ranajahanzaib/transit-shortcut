from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data import LENGTHS, decode, load, load_manifest
from .experiments import get_run, markdown, metrics, read_predictions, runs
from .features import shortcut, signal

app = FastAPI(title="TransitShortcut", version="0.1.0")


class Health(BaseModel):
    status: str
    dataset_prepared: bool
    models_trained: bool
    dataset_id: str | None = None


class Payload(BaseModel):
    data: Any


class Page(BaseModel):
    total: int
    offset: int
    items: list[dict[str, Any]]
    metrics: dict[str, Any] | None = None
    error_breakdown: dict[str, int] | None = None


def clean(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


@app.exception_handler(FileNotFoundError)
def missing(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def invalid(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health", response_model=Health)
def health() -> dict:
    try:
        m = load_manifest()
        return {
            "status": "ready",
            "dataset_prepared": True,
            "models_trained": bool(runs()),
            "dataset_id": m["dataset_id"],
        }
    except FileNotFoundError:
        return {"status": "setup", "dataset_prepared": False, "models_trained": False}


@app.get("/api/dataset/summary", response_model=Payload)
def summary() -> dict:
    return {"data": load_manifest()}


@app.get("/api/experiments", response_model=Payload)
def experiments() -> dict:
    try:
        return {"data": runs()}
    except FileNotFoundError:
        return {"data": []}


@app.get("/api/experiments/{run_id}", response_model=Payload)
def experiment(run_id: str) -> dict:
    return {"data": get_run(run_id)}


@app.get("/api/experiments/{run_id}/predictions", response_model=Page)
def predictions(
    run_id: str,
    mission: Literal["kepler", "k2", "tess", ""] | None = None,
    true_class: Literal["PLANET", "FALSE_POSITIVE", "NO_SIGNAL", ""] | None = None,
    predicted_class: Literal["PLANET", "FALSE_POSITIVE", "NO_SIGNAL", ""] | None = None,
    correct: bool | None = None,
    disagreement: bool | None = None,
    model: Literal["shortcut", "signal"] = "shortcut",
    discovery: Literal["", "shortcut-only", "signal-only", "both-wrong"] = "",
    sort_ablation: int | None = Query(None, ge=0, le=3),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    run = get_run(run_id)
    frame = read_predictions(run)
    if mission:
        frame = frame[frame.mission == mission]
    if true_class:
        frame = frame[frame.disposition == true_class]
    if predicted_class:
        frame = frame[frame[f"{model}_prediction"] == predicted_class]
    if correct is not None:
        frame = frame[(frame[f"{model}_prediction"] == frame.disposition) == correct]
    if disagreement is not None:
        frame = frame[(frame.shortcut_prediction != frame.signal_prediction) == disagreement]
    a = frame.shortcut_prediction == frame.disposition
    b = frame.signal_prediction == frame.disposition
    if discovery == "shortcut-only":
        frame = frame[a & ~b]
    elif discovery == "signal-only":
        frame = frame[~a & b]
    elif discovery == "both-wrong":
        frame = frame[~a & ~b]
    if sort_ablation is not None:
        frame = frame.sort_values(f"ablation_{sort_ablation}_score_delta", ascending=False)
    a = frame.shortcut_prediction == frame.disposition
    b = frame.signal_prediction == frame.disposition
    return {
        "total": len(frame),
        "offset": offset,
        "items": frame.iloc[offset : offset + limit].to_dict("records"),
        "metrics": {
            m: metrics(frame.disposition, frame[f"{m}_prediction"], run["classes"])
            for m in ("shortcut", "signal")
        },
        "error_breakdown": {
            "both_correct": int((a & b).sum()),
            "both_wrong": int((~a & ~b).sum()),
            "disagree": int((frame.shortcut_prediction != frame.signal_prediction).sum()),
            "different_errors": int(
                (~a & ~b & (frame.shortcut_prediction != frame.signal_prediction)).sum()
            ),
        },
    }


@app.get("/api/targets", response_model=Page)
def targets(
    mission: Literal["kepler", "k2", "tess", ""] | None = None,
    true_class: Literal["PLANET", "FALSE_POSITIVE", "NO_SIGNAL", ""] | None = None,
    query: str = "",
    pattern: Literal["all", "zero", "incomplete"] = "all",
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    frame, _ = load()
    if mission:
        frame = frame[frame.mission == mission]
    if true_class:
        frame = frame[frame.disposition == true_class]
    frame = frame[frame.star_id.str.contains(query, case=False, regex=False)]
    if pattern == "incomplete":
        frame = frame[~frame.has_transit_params]
    if pattern == "zero":
        frame = frame[[bool(np.all(decode(v, "flux_global") == 0)) for v in frame.flux_global]]
    return {
        "total": len(frame),
        "offset": offset,
        "items": frame.iloc[offset : offset + limit][
            ["target_id", "star_id", "mission", "disposition", "split"]
        ].to_dict("records"),
    }


def target_row(target_id: str) -> dict:
    frame, _ = load()
    rows = frame[frame.target_id == target_id]
    if rows.empty:
        raise HTTPException(404, "Target not found")
    return rows.iloc[0].to_dict()


@app.get("/api/targets/{target_id}", response_model=Payload)
def target(target_id: str) -> dict:
    row = target_row(target_id)
    metadata = {k: None if pd.isna(v) else v for k, v in row.items() if k not in LENGTHS}
    return {
        "data": {
            "metadata": clean(metadata),
            "shortcut_features": shortcut(row),
            "signal_features": signal(row["flux_raw"]),
        }
    }


@app.get("/api/targets/{target_id}/flux", response_model=Payload)
def flux(
    target_id: str,
    view: Literal["flux_raw", "flux_global", "flux_local", "flux_odd", "flux_even"] = "flux_raw",
    max_points: int = Query(2000, ge=100, le=20000),
) -> dict:
    array = decode(target_row(target_id)[view], view)
    unavailable = view != "flux_raw" and bool(np.all(array == 0))
    # Min/max envelope preserves extrema within every sample-index bin.
    indices = np.arange(len(array))
    if len(array) > max_points:
        chosen = {0, len(array) - 1}
        for chunk in np.array_split(indices, (max_points - 2) // 2):
            valid = chunk[np.isfinite(array[chunk])]
            if len(valid):
                chosen.update(
                    [int(valid[np.argmin(array[valid])]), int(valid[np.argmax(array[valid])])]
                )
        indices = np.array(sorted(chosen))
    return {
        "data": {
            "view": view,
            "unavailable": unavailable,
            "annotation": "Unavailable phase-folded representation; stored as zeros. These are not observed zero flux."
            if unavailable
            else None,
            "x_label": "Sample index"
            if view == "flux_raw"
            else "Folded bin index (relative position)",
            "original_n": len(array),
            "x": indices.tolist(),
            "y": [float(v) if np.isfinite(v) else None for v in array[indices]],
        }
    }


@app.get("/api/targets/{target_id}/predictions", response_model=Payload)
def target_predictions(target_id: str, run_id: str) -> dict:
    run = get_run(run_id)
    pred = read_predictions(run)
    pred = pred[pred.target_id == target_id]
    if pred.empty:
        raise HTTPException(404, "Target is not in this experiment's evaluated test cohort")
    return {"data": pred.iloc[0].to_dict()}


@app.get("/api/audit/artifacts", response_model=Payload)
def artifacts() -> dict:
    return {"data": load_manifest()["summary"]}


@app.get("/api/audit/ablations", response_model=Payload)
def ablations(run_id: str) -> dict:
    return {"data": get_run(run_id)["ablations"]}


@app.get("/api/reports/{run_id}")
def report(run_id: str, format: Literal["json", "markdown", "csv"] = "json") -> Response:
    run = get_run(run_id)
    if format == "csv":
        content, media, ext = read_predictions(run).to_csv(index=False), "text/csv", "csv"
    elif format == "markdown":
        content, media, ext = markdown(run), "text/markdown", "md"
    else:
        content, media, ext = json.dumps(run, indent=2), "application/json", "json"
    return Response(
        content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{run_id}.{ext}"'},
    )


web = Path(__file__).resolve().parents[2] / "web/dist"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
