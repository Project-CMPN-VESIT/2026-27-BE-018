"""One detection run: pick scene -> fetch pixels -> model -> masks -> map products -> stats."""
from __future__ import annotations

import logging
import shutil
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import shape

from ..config import get_settings
from ..db import Run, now, session
from ..imagery.base import ProviderError, SceneInfo
from ..imagery.geo import aoi_mask, area_km2
from ..imagery.masks import build_mask
from ..imagery.registry import collection_of, get_provider, scene_cache
from ..ml.engine import MARIDA_CLASSES, ModelEngine
from . import outputs as out

log = logging.getLogger(__name__)


class RunCancelled(Exception):
    pass


def pick_best_scene(scenes: list[SceneInfo]) -> Optional[SceneInfo]:
    """Prefer scenes that cover the AOI, then least cloudy, then most recent."""
    if not scenes:
        return None

    def score(s: SceneInfo) -> tuple:
        cov = s.coverage if s.coverage is not None else 1.0
        cloud = s.cloud_cover if s.cloud_cover is not None else 50.0
        return (cov < 0.5, round(cloud + (1 - cov) * 100, 1), -_ts(s.datetime))

    return sorted(scenes, key=score)[0]


def _opt(params: dict, key: str, default):
    v = params.get(key)
    return default if v is None else v


def _ts(iso: str) -> float:
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


class _Progress:
    """Maps sub-step progress into the run's 0..1 bar and writes it to the DB at most twice a second."""

    def __init__(self, run_id: str):
        self.run_id, self._last = run_id, 0.0

    def set(self, status: str, frac: float, msg: str, force: bool = False) -> None:
        t = time.time()
        if not force and t - self._last < 0.5:
            return
        self._last = t
        with session() as db:
            r = db.get(Run, self.run_id)
            if r is None:
                raise RunCancelled()
            r.status, r.progress, r.message = status, float(min(max(frac, 0), 1)), msg[:500]
            db.commit()

    def span(self, status: str, lo: float, hi: float):
        return lambda f, m: self.set(status, lo + (hi - lo) * f, m)


def run_dir(run_id: str) -> Path:
    return get_settings().runs_dir / run_id


def execute(run_id: str, engine: ModelEngine) -> None:
    s = get_settings()
    t_start = time.time()
    with session() as db:
        r = db.get(Run, run_id)
        if r is None:
            return
        r.status, r.started_at, r.progress, r.error = "searching", now(), 0.0, None
        db.commit()
        geom_json, provider_name, scene_id, params = r.geometry, r.provider, r.scene_id, dict(r.params or {})
    pg = _Progress(run_id)
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    try:
        aoi = shape(geom_json)
        provider = get_provider(provider_name)
        threshold = float(params.get("threshold", s.debris_threshold))

        # 1) scene selection
        chosen: Optional[SceneInfo] = None
        if not scene_id:
            end = date.fromisoformat(params.get("end_date") or date.today().isoformat())
            start = date.fromisoformat(params.get("start_date") or (end - timedelta(days=30)).isoformat())
            pg.set("searching", 0.02, f"searching scenes {start} .. {end}", force=True)
            scenes = provider.search(aoi, start, end, float(params.get("max_cloud", 30)), s.scene_search_limit)
            chosen = pick_best_scene(scenes)
            if chosen is None:
                raise ProviderError(f"no Sentinel-2 scene with <= {params.get('max_cloud', 30)}% cloud between "
                                    f"{start} and {end} - widen the date range or raise the cloud limit")
            scene_id = chosen.id

        # 2) pixels (served from the on-disk scene cache when this exact area + scene was fetched before)
        t0 = time.time()
        cache = scene_cache() if provider_name != "upload" else None
        ckey = cache.key(provider_name, collection_of(provider_name), scene_id, aoi) if cache else ""
        data = cache.get(ckey) if cache else None
        from_cache = data is not None
        if from_cache:
            pg.set("fetching", 0.40, "imagery loaded from local cache", force=True)
        else:
            pg.set("fetching", 0.05, "downloading Sentinel-2 bands", force=True)
            data = provider.load(scene_id, aoi, pg.span("fetching", 0.05, 0.45))
            if cache:
                pg.set("fetching", 0.45, "saving imagery to local cache", force=True)
                cache.put(ckey, data)
        fetch_s = time.time() - t0
        scene = data.scene if data.scene.id else (chosen or data.scene)
        if chosen and chosen.cloud_cover is not None and scene.cloud_cover is None:
            scene.cloud_cover = chosen.cloud_cover
        with session() as db:
            r = db.get(Run, run_id)
            r.scene_id, r.scene = scene_id, scene.to_dict()
            db.commit()

        # 3) masks first, so the model only runs on tiles that contain analysable water
        inside = aoi_mask(aoi, data.grid)
        mask = build_mask(data.reflectance, data.scl, inside, mask_clouds=_opt(params, "mask_clouds", s.mask_clouds),
                          mask_land=_opt(params, "mask_land", s.mask_land), max_hole_fill_px=s.max_hole_fill_px)

        # 4) model
        pg.set("inferring", 0.45, "running GeoLitterNet", force=True)
        res = engine.predict(data.reflectance, pg.span("inferring", 0.45, 0.88), needed=mask.valid)

        # 5) products
        pg.set("rendering", 0.9, "building heatmap and detections", force=True)
        t1 = time.time()
        prob, valid, grid = res.debris_prob, mask.valid, data.grid
        min_px = int(params.get("min_cluster_pixels", s.min_cluster_pixels))
        det = out.detections(prob, valid, res.class_map, grid, threshold, min_px)
        layers = {
            "heatmap": {"url": f"/api/runs/{run_id}/files/heatmap.png",
                        "coordinates": out.render_heatmap(prob, valid, grid, d / "heatmap.png", s.heatmap_floor)},
            "rgb": {"url": f"/api/runs/{run_id}/files/rgb.png",
                    "coordinates": out.render_rgb(data.reflectance, inside, grid, d / "rgb.png")},
            "classes": {"url": f"/api/runs/{run_id}/files/classes.png",
                        "coordinates": out.render_classes(res.class_map, valid, grid, d / "classes.png")},
            "detections": {"url": f"/api/runs/{run_id}/files/detections.geojson"},
        }
        out.write_json(det, d / "detections.geojson")
        tags = {"model": "GeoLitterNet v3 seed42", "model_sha256": engine.sha256, "scene": scene_id,
                "provider": provider_name, "threshold": threshold}
        out.write_geotiff(np.where(valid, prob, -1).astype(np.float32), grid, d / "probability.tif", -1.0,
                          ["P(Marine Debris)"], tags)
        out.write_geotiff(np.where(valid, res.class_map, 255).astype(np.uint8), grid, d / "classes.tif", 255,
                          ["MARIDA class index"], {**tags, "classes": "|".join(MARIDA_CLASSES)})

        # 6) stats
        px_m2 = abs(grid.transform.a * grid.transform.e)
        n_valid = int(valid.sum())
        hit = (prob >= threshold) & valid
        cls_counts = np.bincount(res.class_map[valid], minlength=len(MARIDA_CLASSES)) if n_valid else np.zeros(15)
        fr = mask.fractions(inside)
        warnings = []
        cov = scene.coverage
        if cov is not None and cov < 0.95:
            warnings.append(f"The scene covers only {cov:.0%} of the area.")
        if fr["cloud"] > 0.5:
            warnings.append(f"{fr['cloud']:.0%} of the area is cloud or cloud shadow; detections there are suppressed.")
        if n_valid == 0:
            warnings.append("No usable open-water pixels in this area (all land, cloud or no data).")
        elif fr["land"] > 0.5:
            warnings.append("Most of the area is land; the model only analyses water.")
        stats = {
            "aoi_km2": round(area_km2(aoi), 3), "analysed_km2": round(n_valid * px_m2 / 1e6, 3),
            "fractions": fr, "mask_method": mask.method, "threshold": threshold, "min_cluster_pixels": min_px,
            "debris_pixels": int(hit.sum()), "debris_area_m2": round(float(hit.sum() * px_m2), 1),
            "clusters": len(det["features"]),
            "max_prob": round(float(prob[valid].max()), 4) if n_valid else 0.0,
            "mean_prob": round(float(prob[valid].mean()), 5) if n_valid else 0.0,
            "class_fractions": {c: round(float(n) / max(n_valid, 1), 5) for c, n in zip(MARIDA_CLASSES, cls_counts) if n},
            "grid": {"crs": grid.crs, "width": grid.width, "height": grid.height, "res_m": abs(grid.transform.a)},
            "timing_s": {"fetch": round(fetch_s, 1), "inference": round(res.seconds, 1),
                         "render": round(time.time() - t1, 1), "total": round(time.time() - t_start, 1)},
            "tiles": res.tiles, "tiles_skipped": res.tiles_skipped, "from_cache": from_cache,
            "model_sha256": engine.sha256, "warnings": warnings,
        }
        out.write_json({"run_id": run_id, "scene": scene.to_dict(), "params": params, "stats": stats,
                        "top_detections": [f["properties"] for f in det["features"][:25]]}, d / "report.json")
        with session() as db:
            r = db.get(Run, run_id)
            r.status, r.progress, r.message = "done", 1.0, f"{stats['clusters']} debris clusters"
            r.stats, r.layers, r.finished_at = stats, layers, now()
            db.commit()
        log.info("run %s done in %.1fs: %s clusters", run_id, stats["timing_s"]["total"], stats["clusters"])
    except RunCancelled:
        shutil.rmtree(d, ignore_errors=True)
    except ProviderError as e:
        _fail(run_id, str(e))
    except Exception as e:  # unexpected: keep the trace in the log, show a short message to the user
        log.exception("run %s failed", run_id)
        _fail(run_id, f"internal error: {type(e).__name__}: {e}")


def _fail(run_id: str, msg: str) -> None:
    with session() as db:
        r = db.get(Run, run_id)
        if r is not None:
            r.status, r.error, r.message, r.finished_at = "failed", msg[:2000], "failed", now()
            db.commit()
