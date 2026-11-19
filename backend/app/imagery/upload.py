"""User-supplied Sentinel-2 GeoTIFFs (offline use, own downloads, or testing).

Accepted layouts (auto-detected):
  * band descriptions naming the bands (B1/B01 ... B12, optional SCL) - any order;
  * 11 bands in MARIDA order  B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12;
  * 12 bands L2A order        B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12            (B9 dropped);
  * 13 bands L1C order        B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12        (B9, B10 dropped).
Values may be reflectance (0..1) or DN (reflectance x 10000, optionally with a DN offset).
Scene id format: "upload:<uuid>".
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from .base import BAND_IDS, ImageryProvider, ProviderError, ProgressFn, SceneData, SceneInfo, read_to_grid
from .geo import grid_for, to_crs

log = logging.getLogger(__name__)
ORDER_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
ORDER_13 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


def _norm_name(desc: Optional[str]) -> Optional[str]:
    if not desc:
        return None
    d = desc.strip().upper()
    if d in ("SCL", "SCENE_CLASSIFICATION"):
        return "SCL"
    m = re.search(r"B0?(\d{1,2}A?)\b", d)
    if not m:
        return None
    n = m.group(1)
    return "B8A" if n == "8A" else f"B{int(n):02d}"


def _band_map(src) -> dict[str, int]:
    """name -> 1-based band index in the file."""
    named = {_norm_name(d): i + 1 for i, d in enumerate(src.descriptions)}
    named.pop(None, None)
    if all(b in named for b in BAND_IDS):
        return {b: named[b] for b in BAND_IDS + (["SCL"] if "SCL" in named else [])}
    n = src.count
    if n == 11:
        return {b: i + 1 for i, b in enumerate(BAND_IDS)}
    if n in (12, 13):
        order = ORDER_12 if n == 12 else ORDER_13
        return {b: order.index(b) + 1 for b in BAND_IDS}
    raise ProviderError(f"unsupported band layout: {n} bands without recognisable band names "
                        "(need 11 MARIDA-order, 12 L2A-order or 13 L1C-order bands, or named bands)")


class UploadProvider(ImageryProvider):
    name = "upload"
    label = "Uploaded GeoTIFF"

    def __init__(self, uploads_dir: Path):
        self.dir = Path(uploads_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> tuple[bool, str]:
        return True, "upload a Sentinel-2 GeoTIFF"

    # ---------------- registry ----------------
    def _meta_path(self, uid: str) -> Path:
        return self.dir / f"{uid}.json"

    def register(self, tmp_path: Path, filename: str, acquired: Optional[str] = None, dn_offset: float = 0.0) -> dict:
        uid = uuid.uuid4().hex
        dst = self.dir / f"{uid}.tif"
        shutil.move(str(tmp_path), dst)
        try:
            with rasterio.open(dst) as src:
                if src.crs is None:
                    raise ProviderError("GeoTIFF has no coordinate reference system")
                bmap = _band_map(src)
                foot = to_crs(box(*src.bounds), 4326, src.crs.to_string())
                # decide DN vs reflectance from a decimated read of the red band
                ov = src.read(bmap["B04"], out_shape=(min(src.height, 256), min(src.width, 256)),
                              resampling=Resampling.nearest).astype(np.float32)
                ov = ov[np.isfinite(ov) & (ov > 0)]
                p99 = float(np.percentile(ov, 99)) if ov.size else 0.0
                is_dn = p99 > 2.0
                res = abs(src.transform.a)
                meta = {"id": uid, "filename": filename, "bands": bmap, "crs": src.crs.to_string(),
                        "width": src.width, "height": src.height, "res": res,
                        "projected": bool(src.crs.is_projected), "footprint": mapping(foot),
                        "scale": 1e-4 if is_dn else 1.0, "offset": (dn_offset * 1e-4) if is_dn else 0.0,
                        "value_type": "DN" if is_dn else "reflectance",
                        "acquired": acquired or src.tags().get("TIFFTAG_DATETIME") or "",
                        "uploaded": datetime.now(timezone.utc).isoformat(), "has_scl": "SCL" in bmap}
        except ProviderError:
            dst.unlink(missing_ok=True)
            raise
        except Exception as e:
            dst.unlink(missing_ok=True)
            raise ProviderError(f"not a readable GeoTIFF: {e}") from e
        self._meta_path(uid).write_text(json.dumps(meta, indent=2))
        return meta

    def get(self, uid: str) -> dict:
        p = self._meta_path(uid)
        if not p.exists():
            raise ProviderError(f"upload not found: {uid}")
        return json.loads(p.read_text())

    def list(self) -> list[dict]:
        out = [json.loads(p.read_text()) for p in self.dir.glob("*.json")]
        return sorted(out, key=lambda m: m["uploaded"], reverse=True)

    def delete(self, uid: str) -> None:
        self.get(uid)
        (self.dir / f"{uid}.tif").unlink(missing_ok=True)
        self._meta_path(uid).unlink(missing_ok=True)

    def footprint(self, scene_id: str) -> BaseGeometry:
        return shape(self.get(scene_id.removeprefix("upload:"))["footprint"])

    def _info(self, m: dict, aoi: Optional[BaseGeometry]) -> SceneInfo:
        foot = shape(m["footprint"])
        cov = None
        if aoi is not None and aoi.area > 0:
            cov = round(float(foot.intersection(aoi).area / aoi.area), 4)
        return SceneInfo(id=f"upload:{m['id']}", provider=self.name, datetime=m.get("acquired") or m["uploaded"],
                         cloud_cover=None, platform="upload", coverage=cov,
                         extras={"filename": m["filename"], "value_type": m["value_type"], "has_scl": m["has_scl"]})

    # ---------------- provider API ----------------
    def search(self, aoi: BaseGeometry, start: date, end: date, max_cloud: float, limit: int) -> list[SceneInfo]:
        return [self._info(m, aoi) for m in self.list() if shape(m["footprint"]).intersects(aoi)][:limit]

    def load(self, scene_id: str, aoi: BaseGeometry, progress: Optional[ProgressFn] = None) -> SceneData:
        m = self.get(scene_id.removeprefix("upload:"))
        foot = shape(m["footprint"])
        area = aoi.intersection(foot)
        if area.is_empty:
            raise ProviderError("the area does not overlap the uploaded image")
        crs = m["crs"] if (m["projected"] and m["res"] <= 20) else None
        grid = grid_for(area, crs)
        path = str(self.dir / f"{m['id']}.tif")
        refl = np.empty((11, *grid.shape), np.float32)
        for k, b in enumerate(BAND_IDS):
            v = read_to_grid(path, grid, Resampling.bilinear, band=m["bands"][b], fill=0)
            ok = np.isfinite(v) & (v > 0)
            refl[k] = np.where(ok, v * m["scale"] + m["offset"], np.nan)
            if progress:
                progress((k + 1) / 12, f"band {b}")
        refl[:, np.isnan(refl).any(0)] = np.nan
        scl = None
        if m.get("has_scl"):
            scl = read_to_grid(path, grid, Resampling.nearest, band=m["bands"]["SCL"], dtype="uint8")
        if progress:
            progress(1.0, "loaded")
        if np.isnan(refl[0]).all():
            raise ProviderError("no valid pixels in the uploaded image over this area")
        return SceneData(reflectance=refl, scl=scl, grid=grid, scene=self._info(m, aoi))
