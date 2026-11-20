"""Instant re-analysis of a finished run with a new threshold / minimum cluster size.

The model output does not depend on the threshold: every finished run keeps the full per-pixel probability map
(probability.tif) and class map (classes.tif). Re-clustering them takes about a second, with no download and no
model inference, and gives exactly the detections a fresh run with the new settings would give.
"""
from __future__ import annotations

import json
import time

import numpy as np
import rasterio

from ..imagery.geo import Grid
from . import outputs as out
from .runner import run_dir


def recluster(run_id: str, stats: dict, threshold: float, min_px: int) -> tuple[dict, dict]:
    """Returns (new_stats, detections FeatureCollection) and rewrites detections.geojson / report.json."""
    t0 = time.time()
    d = run_dir(run_id)
    with rasterio.open(d / "probability.tif") as src:
        prob = src.read(1)
        grid = Grid(crs=src.crs.to_string(), transform=src.transform, width=src.width, height=src.height)
    with rasterio.open(d / "classes.tif") as src:
        cls = src.read(1)
    valid = prob >= 0                                  # masked pixels were stored as -1
    cls = np.where(valid, cls, 0).astype(np.uint8)
    det = out.detections(prob, valid, cls, grid, threshold, min_px)
    hit = (prob >= threshold) & valid
    px_m2 = abs(grid.transform.a * grid.transform.e)
    new = {**stats, "threshold": threshold, "min_cluster_pixels": min_px, "debris_pixels": int(hit.sum()),
           "debris_area_m2": round(float(hit.sum() * px_m2), 1), "clusters": len(det["features"]),
           "reprocess_s": round(time.time() - t0, 2)}
    out.write_json(det, d / "detections.geojson")
    rp = d / "report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        rep["stats"] = new
        rep.setdefault("params", {}).update({"threshold": threshold, "min_cluster_pixels": min_px})
        rep["top_detections"] = [f["properties"] for f in det["features"][:25]]
        out.write_json(rep, rp)
    return new, det
