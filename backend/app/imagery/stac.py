"""Sentinel-2 L2A from a STAC API (default: Element84 Earth Search, open COGs on AWS - no account needed).

A "scene" in this product = all granules (MGRS tiles) of one satellite pass over the AOI, mosaicked.
Scene id format: "<item_id>[,<item_id>...]".
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

import numpy as np
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .base import BAND_IDS, ImageryProvider, ProviderError, ProgressFn, SceneData, SceneInfo, mosaic_fill, read_to_grid
from .geo import grid_for

log = logging.getLogger(__name__)

# candidate asset keys per band (Earth Search v1 names first, then Planetary Computer / CDSE style)
ASSET_KEYS: dict[str, list[str]] = {
    "B01": ["coastal", "B01"], "B02": ["blue", "B02"], "B03": ["green", "B03"], "B04": ["red", "B04"],
    "B05": ["rededge1", "B05"], "B06": ["rededge2", "B06"], "B07": ["rededge3", "B07"], "B08": ["nir", "B08"],
    "B8A": ["nir08", "B8A"], "B11": ["swir16", "B11"], "B12": ["swir22", "B12"], "SCL": ["scl", "SCL"],
}


def _asset(item, band: str):
    for k in ASSET_KEYS[band]:
        if k in item.assets:
            return item.assets[k]
    raise ProviderError(f"scene {item.id} has no asset for band {band}")


def _scale_offset(item, asset) -> tuple[float, float]:
    """DN -> reflectance.

    Order matters: Earth Search v1 `sentinel-2-l2a` re-encodes COGs with the baseline-04.00 offset already removed
    and sets `earthsearch:boa_offset_applied: true`, yet still advertises `raster:bands.offset = -0.1`. Applying that
    offset again would push water reflectance below zero, so the flag wins over the band metadata.
    """
    rb = (asset.extra_fields.get("raster:bands") or [{}])[0] or {}
    scale = float(rb.get("scale", 1e-4))
    props = item.properties
    if props.get("earthsearch:boa_offset_applied") is True:
        return scale, 0.0
    if "offset" in rb:
        return scale, float(rb["offset"])
    try:
        baseline = float(props.get("s2:processing_baseline", "0"))
    except ValueError:
        baseline = 0.0
    return scale, (-0.1 if baseline >= 4.0 else 0.0)   # baseline >= 04.00 adds a +1000 DN offset


def _epsg(item) -> Optional[str]:
    p = item.properties
    if p.get("proj:epsg"):
        return f"EPSG:{p['proj:epsg']}"
    if p.get("proj:code"):
        return str(p["proj:code"])
    return None


class StacProvider(ImageryProvider):
    name = "stac"

    def __init__(self, url: str, collection: str, timeout_s: int = 60):
        self.url, self.collection, self.timeout_s = url, collection, timeout_s
        self.label = f"Sentinel-2 L2A (STAC: {url.split('//')[-1].split('/')[0]})"
        self._client: Optional[Client] = None

    def _cl(self) -> Client:
        if self._client is None:
            try:
                self._client = Client.open(self.url, timeout=self.timeout_s)
            except Exception as e:  # network / DNS / TLS
                raise ProviderError(f"cannot reach STAC API {self.url}: {e}") from e
        return self._client

    def status(self) -> tuple[bool, str]:
        return True, "public, no credentials required"

    # ---------------- search ----------------
    def _items(self, aoi: BaseGeometry, start: date, end: date, max_cloud: float) -> list:
        s = self._cl().search(collections=[self.collection], intersects=mapping(aoi),
                              datetime=f"{start.isoformat()}/{end.isoformat()}",
                              query={"eo:cloud_cover": {"lte": max_cloud}}, max_items=300)
        try:
            return list(s.items())
        except Exception as e:
            raise ProviderError(f"STAC search failed: {e}") from e

    def search(self, aoi: BaseGeometry, start: date, end: date, max_cloud: float, limit: int) -> list[SceneInfo]:
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for it in self._items(aoi, start, end, max_cloud):
            groups[(it.datetime.date().isoformat(), it.properties.get("platform", ""))].append(it)
        scenes = []
        for (_, platform), items in groups.items():
            items.sort(key=lambda i: i.id)
            foot = unary_union([shape(i.geometry) for i in items])
            cov = float(foot.intersection(aoi).area / aoi.area) if aoi.area > 0 else 0.0
            clouds = [i.properties.get("eo:cloud_cover") for i in items if i.properties.get("eo:cloud_cover") is not None]
            thumb = next((i.assets["thumbnail"].href for i in items if "thumbnail" in i.assets), None)
            scenes.append(SceneInfo(
                id=",".join(i.id for i in items), provider=self.name,
                datetime=min(i.datetime for i in items).isoformat(),
                cloud_cover=round(float(np.mean(clouds)), 2) if clouds else None,
                platform=platform, tiles=sorted({str(i.properties.get("s2:mgrs_tile") or i.properties.get("grid:code") or "") for i in items} - {""}),
                coverage=round(cov, 4), thumbnail=thumb,
                extras={"collection": self.collection, "items": [i.id for i in items]},
            ))
        scenes.sort(key=lambda s: s.datetime, reverse=True)
        return scenes[:limit]

    # ---------------- load ----------------
    def load(self, scene_id: str, aoi: BaseGeometry, progress: Optional[ProgressFn] = None) -> SceneData:
        ids = [s for s in scene_id.split(",") if s]
        if not ids:
            raise ProviderError("empty scene id")
        try:
            items = list(self._cl().search(collections=[self.collection], ids=ids).items())
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"STAC lookup failed: {e}") from e
        if not items:
            raise ProviderError(f"scene not found in {self.collection}: {scene_id}")
        items = [i for i in items if shape(i.geometry).intersects(aoi)] or items
        epsgs = Counter(e for e in (_epsg(i) for i in items) if e)
        grid = grid_for(aoi, epsgs.most_common(1)[0][0] if epsgs else None)

        refl = np.full((11, *grid.shape), np.nan, dtype=np.float32)
        scl = np.zeros(grid.shape, dtype=np.uint8)
        n_jobs = len(items) * 12
        done = [0]

        def tick(msg: str):
            done[0] += 1
            if progress:
                progress(done[0] / n_jobs, msg)

        for item in items:
            def read_band(b: str) -> tuple[str, np.ndarray]:
                a = _asset(item, b)
                if b == "SCL":
                    arr = read_to_grid(a.href, grid, Resampling.nearest, dtype="uint8")
                else:
                    dn = read_to_grid(a.href, grid, Resampling.bilinear, dtype="float32")
                    scale, offset = _scale_offset(item, a)
                    arr = np.where(dn > 0, dn * scale + offset, np.nan).astype(np.float32)
                tick(f"{item.id}: {b}")
                return b, arr

            with ThreadPoolExecutor(max_workers=6) as ex:
                res = dict(ex.map(read_band, BAND_IDS + ["SCL"]))
            stack = np.stack([res[b] for b in BAND_IDS])
            stack[:, np.isnan(stack).any(0)] = np.nan
            hole = np.isnan(refl[0]) & ~np.isnan(stack[0])
            mosaic_fill(refl, stack)
            scl[hole] = res["SCL"][hole]

        if np.isnan(refl[0]).all():
            raise ProviderError("the selected scene has no valid pixels over this area")
        scenes = self.search_ids(items, aoi)
        return SceneData(reflectance=refl, scl=scl, grid=grid, scene=scenes)

    def search_ids(self, items: list, aoi: BaseGeometry) -> SceneInfo:
        foot = unary_union([shape(i.geometry) for i in items])
        clouds = [i.properties.get("eo:cloud_cover") for i in items if i.properties.get("eo:cloud_cover") is not None]
        return SceneInfo(id=",".join(sorted(i.id for i in items)), provider=self.name,
                         datetime=min(i.datetime for i in items).isoformat(),
                         cloud_cover=round(float(np.mean(clouds)), 2) if clouds else None,
                         platform=items[0].properties.get("platform", ""),
                         tiles=sorted({str(i.properties.get("s2:mgrs_tile") or i.properties.get("grid:code") or "") for i in items} - {""}),
                         coverage=round(float(foot.intersection(aoi).area / aoi.area), 4) if aoi.area else None,
                         extras={"collection": self.collection, "processing_baseline": items[0].properties.get("s2:processing_baseline")})

