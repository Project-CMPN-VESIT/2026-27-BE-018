"""Sentinel-2 L2A from Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED).

Needs a Google Cloud project registered for Earth Engine and a service-account JSON key
(see docs/GEE_SETUP.md). Pixels are pulled with ee.data.computePixels in 512 px blocks on the same
10 m UTM grid the rest of the product uses. S2_SR_HARMONIZED already removes the baseline-04.00
offset, so reflectance = DN / 10000.

Scene id format: "<system:index>[,<system:index>...]" (granules of one pass).
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .base import ImageryProvider, ProviderError, ProgressFn, SceneData, SceneInfo
from .geo import grid_for

log = logging.getLogger(__name__)
EE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BLOCK = 512


class GeeProvider(ImageryProvider):
    name = "gee"
    label = "Sentinel-2 L2A (Google Earth Engine)"

    def __init__(self, enabled: bool, project: str, service_account: str, key_file: Optional[Path], collection: str):
        self.enabled, self.project, self.sa, self.key_file, self.collection = enabled, project, service_account, key_file, collection
        self._ready = False
        self._err = ""
        self._lock = threading.Lock()

    def _init(self):
        with self._lock:
            if self._ready:
                return
            if not self.enabled:
                raise ProviderError("Google Earth Engine is disabled (set GLN_GEE_ENABLED=true)")
            try:
                import ee  # noqa: WPS433
            except ImportError as e:
                raise ProviderError("earthengine-api is not installed") from e
            try:
                if self.key_file:
                    if not Path(self.key_file).exists():
                        raise ProviderError(f"GEE key file not found: {self.key_file}")
                    creds = ee.ServiceAccountCredentials(self.sa or None, str(self.key_file))
                    ee.Initialize(creds, project=self.project or None)
                else:
                    ee.Initialize(project=self.project or None)   # uses `earthengine authenticate` user credentials
            except ProviderError:
                raise
            except Exception as e:
                self._err = str(e)
                raise ProviderError(f"Earth Engine initialisation failed: {e}") from e
            self._ready = True

    def status(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "disabled - set GLN_GEE_ENABLED=true and provide a service-account key"
        if not self.project:
            return False, "GLN_GEE_PROJECT is not set"
        if self.key_file and not Path(self.key_file).exists():
            return False, f"key file not found: {self.key_file}"
        return True, "configured" if not self._err else f"error: {self._err}"

    def _ee(self):
        self._init()
        import ee
        return ee

    def search(self, aoi: BaseGeometry, start: date, end: date, max_cloud: float, limit: int) -> list[SceneInfo]:
        ee = self._ee()
        region = ee.Geometry(mapping(aoi))
        col = (ee.ImageCollection(self.collection).filterBounds(region)
               .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
               .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud)))
        try:
            info = col.limit(300).map(lambda im: ee.Feature(im.geometry(), {
                "idx": im.get("system:index"), "t": im.get("system:time_start"),
                "cloud": im.get("CLOUDY_PIXEL_PERCENTAGE"), "sat": im.get("SPACECRAFT_NAME"),
                "tile": im.get("MGRS_TILE")})).getInfo()
        except Exception as e:
            raise ProviderError(f"Earth Engine search failed: {e}") from e
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for f in info.get("features", []):
            p = f["properties"]
            t = datetime.fromtimestamp(p["t"] / 1000, tz=timezone.utc)
            groups[(t.date().isoformat(), p.get("sat") or "")].append({**p, "dt": t, "geom": f["geometry"]})
        out = []
        for (_, sat), fs in groups.items():
            foot = unary_union([shape(f["geom"]) for f in fs])
            out.append(SceneInfo(
                id=",".join(sorted(f["idx"] for f in fs)), provider=self.name,
                datetime=min(f["dt"] for f in fs).isoformat(),
                cloud_cover=round(float(np.mean([f["cloud"] for f in fs if f["cloud"] is not None] or [np.nan])), 2),
                platform=sat.lower(), tiles=sorted({f.get("tile") or "" for f in fs} - {""}),
                coverage=round(float(foot.intersection(aoi).area / aoi.area), 4) if aoi.area else None,
                extras={"collection": self.collection}))
        out.sort(key=lambda s: s.datetime, reverse=True)
        return out[:limit]

    def load(self, scene_id: str, aoi: BaseGeometry, progress: Optional[ProgressFn] = None) -> SceneData:
        ee = self._ee()
        ids = [s for s in scene_id.split(",") if s]
        if not ids:
            raise ProviderError("empty scene id")
        imgs = [ee.Image(f"{self.collection}/{i}") for i in ids]
        img = ee.ImageCollection(imgs).mosaic().select(EE_BANDS + ["SCL"]).toUint16()
        grid = grid_for(aoi)
        H, W = grid.shape
        refl = np.full((11, H, W), np.nan, np.float32)
        scl = np.zeros((H, W), np.uint8)
        blocks = [(r, c) for r in range(0, H, BLOCK) for c in range(0, W, BLOCK)]
        for k, (r, c) in enumerate(blocks):
            h, w = min(BLOCK, H - r), min(BLOCK, W - c)
            x0 = grid.transform.c + c * grid.transform.a
            y0 = grid.transform.f + r * grid.transform.e
            req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
                   "grid": {"dimensions": {"width": w, "height": h},
                            "affineTransform": {"scaleX": grid.transform.a, "shearX": 0, "translateX": x0,
                                                "shearY": 0, "scaleY": grid.transform.e, "translateY": y0},
                            "crsCode": grid.crs}}
            try:
                arr = ee.data.computePixels(req)
            except Exception as e:
                raise ProviderError(f"Earth Engine pixel request failed: {e}") from e
            dn = np.stack([arr[b].astype(np.float32) for b in EE_BANDS])
            valid = (dn > 0).all(0)
            refl[:, r:r + h, c:c + w] = np.where(valid, dn / 10000.0, np.nan)
            scl[r:r + h, c:c + w] = arr["SCL"].astype(np.uint8)
            if progress:
                progress((k + 1) / len(blocks), f"block {k + 1}/{len(blocks)}")
        if np.isnan(refl[0]).all():
            raise ProviderError("the selected scene has no valid pixels over this area")
        try:
            meta = ee.Image(f"{self.collection}/{ids[0]}").toDictionary(
                ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "SPACECRAFT_NAME", "MGRS_TILE"]).getInfo()
        except Exception:
            meta = {}
        t = meta.get("system:time_start")
        scene = SceneInfo(id=",".join(sorted(ids)), provider=self.name,
                          datetime=datetime.fromtimestamp(t / 1000, tz=timezone.utc).isoformat() if t else "",
                          cloud_cover=meta.get("CLOUDY_PIXEL_PERCENTAGE"),
                          platform=str(meta.get("SPACECRAFT_NAME", "")).lower(),
                          tiles=[meta["MGRS_TILE"]] if meta.get("MGRS_TILE") else [],
                          extras={"collection": self.collection})
        return SceneData(reflectance=refl, scl=scl, grid=grid, scene=scene)
