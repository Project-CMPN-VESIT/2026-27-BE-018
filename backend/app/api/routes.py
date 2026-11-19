"""HTTP API (mounted under /api)."""
from __future__ import annotations

import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from shapely.geometry import mapping
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import AOI, Run, get_db, latest_run_for
from ..imagery.base import ProviderError
from ..imagery.geo import area_km2, parse_geometry
from ..imagery.registry import get_provider, providers, upload_provider
from ..ml.engine import MARIDA_CLASSES, TILE
from ..ml.preprocess import BANDS, INDEX_NAMES
from ..pipeline.runner import run_dir
from ..schemas import AOIIn, AOIPatch, AOIRunIn, RunIn, SceneSearchIn

RUN_FILES = {
    "heatmap.png": "image/png", "rgb.png": "image/png", "classes.png": "image/png",
    "detections.geojson": "application/geo+json", "probability.tif": "image/tiff",
    "classes.tif": "image/tiff", "report.json": "application/json",
}


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    key = get_settings().api_key
    if key and x_api_key != key:
        raise HTTPException(401, "missing or invalid X-API-Key")


public = APIRouter(prefix="/api")
router = APIRouter(prefix="/api", dependencies=[Depends(require_key)])


# ---------------------------------------------------------------- system
@public.get("/health")
def health(request: Request) -> dict[str, Any]:
    eng = request.app.state.engine
    return {"status": "ok" if eng.loaded else "degraded", "model_loaded": eng.loaded, "device": str(eng.device)}


@router.get("/info")
def info(request: Request) -> dict[str, Any]:
    s, eng, jobs = get_settings(), request.app.state.engine, request.app.state.jobs
    provs = []
    for name, p in providers().items():
        ok, why = p.status()
        provs.append({"name": name, "label": p.label, "available": ok, "detail": why})
    return {
        "model": {"name": "GeoLitterNet v3", "checkpoint": s.model_path.name, "sha256": eng.sha256,
                  "parameters": eng.n_params, "classes": MARIDA_CLASSES, "debris_class": MARIDA_CLASSES[0],
                  "input_bands": BANDS, "physics_indices": INDEX_NAMES, "tile_px": TILE, "resolution_m": 10,
                  "device": str(eng.device)},
        "providers": provs, "default_provider": s.default_provider,
        "defaults": {"threshold": s.debris_threshold, "max_cloud": 30, "days_back": 30,
                     "min_cluster_pixels": s.min_cluster_pixels, "mask_land": s.mask_land, "mask_clouds": s.mask_clouds},
        "limits": {"max_aoi_km2": s.max_aoi_km2, "max_upload_mb": s.max_upload_mb},
        "queue": {"running": jobs.current, "waiting": jobs.q.qsize()},
    }


def _aoi_geom(geometry: dict) -> Any:
    try:
        g = parse_geometry(geometry)
    except ValueError as e:
        raise HTTPException(422, str(e))
    a = area_km2(g)
    lim = get_settings().max_aoi_km2
    if a > lim:
        raise HTTPException(422, f"area is {a:.0f} km2; the limit is {lim:.0f} km2 (GLN_MAX_AOI_KM2)")
    if a < 0.01:
        raise HTTPException(422, "area is too small (minimum 0.01 km2)")
    return g


# ---------------------------------------------------------------- scenes
@router.post("/scenes/search")
def search_scenes(body: SceneSearchIn) -> dict[str, Any]:
    g = _aoi_geom(body.geometry)
    p = get_provider(body.provider or get_settings().default_provider)
    scenes = p.search(g, body.start_date, body.end_date, body.max_cloud, body.limit)
    return {"provider": p.name, "count": len(scenes), "scenes": [s.to_dict() for s in scenes]}


# ---------------------------------------------------------------- uploads
@router.post("/uploads", status_code=201)
def upload_image(file: UploadFile = File(...), acquired: str = Form(""), dn_offset: float = Form(0.0)) -> dict:
    if not (file.filename or "").lower().endswith((".tif", ".tiff")):
        raise HTTPException(422, "upload a GeoTIFF (.tif / .tiff)")
    limit = get_settings().max_upload_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=get_settings().uploads_dir) as tmp:
        size = 0
        while chunk := file.file.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > limit:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {get_settings().max_upload_mb} MB")
            tmp.write(chunk)
    meta = upload_provider().register(Path(tmp.name), file.filename or "upload.tif", acquired or None, dn_offset)
    return {**meta, "scene_id": f"upload:{meta['id']}"}


@router.get("/uploads")
def list_uploads() -> list[dict]:
    return [{**m, "scene_id": f"upload:{m['id']}"} for m in upload_provider().list()]


@router.delete("/uploads/{uid}", status_code=204)
def delete_upload(uid: str) -> None:
    upload_provider().delete(uid)


# ---------------------------------------------------------------- runs
def _create_run(db: Session, request: Request, *, geometry: Optional[dict], provider: str, scene_id: Optional[str],
                params: dict, aoi_id: Optional[str]) -> Run:
    get_provider(provider)  # fail fast if unavailable
    if geometry is None:
        if provider == "upload" and scene_id:
            fp = upload_provider().footprint(scene_id)
            geometry = mapping(fp)
        else:
            raise HTTPException(422, "geometry (or aoi_id) is required")
    g = _aoi_geom(geometry)
    run = Run(geometry=mapping(g), provider=provider, scene_id=scene_id, aoi_id=aoi_id,
              params={k: v for k, v in params.items() if v is not None})
    db.add(run)
    db.commit()
    request.app.state.jobs.submit(run.id)
    return run


@router.post("/runs", status_code=202)
def create_run(body: RunIn, request: Request, db: Session = Depends(get_db)) -> dict:
    geometry = body.geometry
    if body.aoi_id:
        aoi = db.get(AOI, body.aoi_id)
        if aoi is None:
            raise HTTPException(404, "saved area not found")
        geometry = geometry or aoi.geometry
    params = {"start_date": body.start_date.isoformat() if body.start_date else None,
              "end_date": body.end_date.isoformat() if body.end_date else None, "max_cloud": body.max_cloud,
              "threshold": body.threshold, "min_cluster_pixels": body.min_cluster_pixels,
              "mask_land": body.mask_land, "mask_clouds": body.mask_clouds}
    run = _create_run(db, request, geometry=geometry, provider=body.provider or get_settings().default_provider,
                      scene_id=body.scene_id, params=params, aoi_id=body.aoi_id)
    return run.to_dict()


@router.get("/runs")
def list_runs(aoi_id: Optional[str] = None, limit: int = 50, db: Session = Depends(get_db)) -> list[dict]:
    q = select(Run).order_by(Run.created_at.desc()).limit(min(max(limit, 1), 500))
    if aoi_id:
        q = q.where(Run.aoi_id == aoi_id)
    return [r.to_dict() for r in db.scalars(q)]


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request, db: Session = Depends(get_db)) -> dict:
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    d = r.to_dict()
    if r.status == "queued":
        d["queue_position"] = request.app.state.jobs.position(run_id)
    return d


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request, db: Session = Depends(get_db)) -> None:
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    if request.app.state.jobs.current == run_id:
        raise HTTPException(409, "this run is being processed; delete it when it finishes")
    db.delete(r)
    db.commit()
    shutil.rmtree(run_dir(run_id), ignore_errors=True)


@router.get("/runs/{run_id}/files/{name}")
def run_file(run_id: str, name: str):
    if name not in RUN_FILES:
        raise HTTPException(404, "unknown file")
    p = run_dir(run_id) / name
    if not p.exists():
        raise HTTPException(404, "file not available (run not finished?)")
    dl = name.endswith((".tif", ".json", ".geojson"))
    return FileResponse(p, media_type=RUN_FILES[name], filename=f"gln_{run_id}_{name}" if dl else None,
                        headers={"Cache-Control": "private, max-age=86400"})


# ---------------------------------------------------------------- saved areas
@router.get("/aois")
def list_aois(db: Session = Depends(get_db)) -> list[dict]:
    return [a.to_dict(latest_run_for(db, a.id)) for a in db.scalars(select(AOI).order_by(AOI.created_at.desc()))]


@router.post("/aois", status_code=201)
def create_aoi(body: AOIIn, db: Session = Depends(get_db)) -> dict:
    g = _aoi_geom(body.geometry)
    a = AOI(name=body.name.strip(), geometry=mapping(g), area_km2=area_km2(g), notes=body.notes)
    db.add(a)
    db.commit()
    return a.to_dict()


@router.get("/aois/{aoi_id}")
def get_aoi(aoi_id: str, db: Session = Depends(get_db)) -> dict:
    a = db.get(AOI, aoi_id)
    if a is None:
        raise HTTPException(404, "saved area not found")
    return a.to_dict(latest_run_for(db, a.id))


@router.patch("/aois/{aoi_id}")
def patch_aoi(aoi_id: str, body: AOIPatch, db: Session = Depends(get_db)) -> dict:
    a = db.get(AOI, aoi_id)
    if a is None:
        raise HTTPException(404, "saved area not found")
    if body.name is not None:
        a.name = body.name.strip()
    if body.notes is not None:
        a.notes = body.notes
    db.commit()
    return a.to_dict(latest_run_for(db, a.id))


@router.delete("/aois/{aoi_id}", status_code=204)
def delete_aoi(aoi_id: str, delete_runs: bool = False, db: Session = Depends(get_db)) -> None:
    a = db.get(AOI, aoi_id)
    if a is None:
        raise HTTPException(404, "saved area not found")
    if delete_runs:
        ids = [r.id for r in db.scalars(select(Run).where(Run.aoi_id == aoi_id))]
        db.execute(delete(Run).where(Run.aoi_id == aoi_id))
        for rid in ids:
            shutil.rmtree(run_dir(rid), ignore_errors=True)
    db.delete(a)
    db.commit()


@router.post("/aois/{aoi_id}/runs", status_code=202)
def run_aoi_latest(aoi_id: str, body: AOIRunIn, request: Request, db: Session = Depends(get_db)) -> dict:
    """Detect on the best (least cloudy) scene of the last `days_back` days for a saved area."""
    a = db.get(AOI, aoi_id)
    if a is None:
        raise HTTPException(404, "saved area not found")
    end = date.today()
    params = {"start_date": (end - timedelta(days=body.days_back)).isoformat(), "end_date": end.isoformat(),
              "max_cloud": body.max_cloud, "threshold": body.threshold}
    run = _create_run(db, request, geometry=a.geometry, provider=body.provider or get_settings().default_provider,
                      scene_id=None, params=params, aoi_id=a.id)
    return run.to_dict()


__all__ = ["router", "public", "ProviderError"]
