"""Turn model outputs into map-ready products: PNG overlays (Web Mercator), GeoTIFFs and GeoJSON detections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from ..imagery.geo import Grid, to_crs
from ..ml.engine import MARIDA_CLASSES

WEB_MERCATOR = "EPSG:3857"

# MARIDA class colours (debris = magenta so it never blends with water/imagery)
CLASS_COLORS = [
    (255, 0, 200), (0, 128, 0), (130, 200, 80), (140, 100, 40), (255, 255, 255), (200, 200, 200),
    (20, 60, 160), (170, 130, 60), (230, 230, 180), (190, 160, 100), (60, 170, 200), (150, 200, 255),
    (60, 60, 60), (255, 220, 120), (90, 120, 200),
]

# heatmap ramp: low = pale yellow, high = deep red/magenta
_STOPS = [(0.0, (255, 255, 178)), (0.25, (254, 204, 92)), (0.5, (253, 141, 60)),
          (0.75, (240, 59, 32)), (0.9, (189, 0, 38)), (1.0, (122, 1, 119))]


def _lut() -> np.ndarray:
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, [s[0] for s in _STOPS], [s[1][ch] for s in _STOPS]).astype(np.uint8)
    return lut


HEAT_LUT = _lut()


def to_mercator(arr: np.ndarray, grid: Grid, resampling: Resampling, src_nodata: float) -> tuple[np.ndarray, tuple]:
    """Warp a 2-D array from the analysis grid to Web Mercator. Returns (array, (minx, miny, maxx, maxy))."""
    h, w = grid.shape
    dst_tr, dw, dh = calculate_default_transform(grid.crs, WEB_MERCATOR, w, h, *grid.bounds)
    out = np.full((dh, dw), src_nodata, dtype=arr.dtype)
    reproject(arr, out, src_transform=grid.transform, src_crs=grid.crs, src_nodata=src_nodata,
              dst_transform=dst_tr, dst_crs=WEB_MERCATOR, dst_nodata=src_nodata, resampling=resampling)
    bounds = (dst_tr.c, dst_tr.f + dst_tr.e * dh, dst_tr.c + dst_tr.a * dw, dst_tr.f)
    return out, bounds


def mercator_corners_lonlat(b: tuple) -> list[list[float]]:
    """MapLibre image-source corners: top-left, top-right, bottom-right, bottom-left."""
    tr = Transformer.from_crs(WEB_MERCATOR, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = b
    pts = [(minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]
    return [[round(v, 7) for v in tr.transform(x, y)] for x, y in pts]


def render_heatmap(prob: np.ndarray, valid: np.ndarray, grid: Grid, path: Path, floor: float) -> list:
    p = np.where(valid, prob, -1.0).astype(np.float32)
    pm, b = to_mercator(p, grid, Resampling.bilinear, -1.0)
    ok = pm >= 0
    t = np.clip((pm - floor) / max(1e-6, 1 - floor), 0, 1)
    rgba = np.zeros((*pm.shape, 4), np.uint8)
    rgba[..., :3] = HEAT_LUT[(t * 255).astype(np.uint8)]
    rgba[..., 3] = np.where(ok & (pm >= floor), (60 + 195 * np.sqrt(t)).astype(np.uint8), 0)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
    return mercator_corners_lonlat(b)


def render_rgb(refl: np.ndarray, inside: np.ndarray, grid: Grid, path: Path) -> list:
    rgb = refl[[3, 2, 1]]
    data = inside & ~np.isnan(rgb).any(0)
    vals = rgb[:, data]
    hi = float(np.nanpercentile(vals, 98)) if vals.size else 0.3
    hi = min(max(hi, 0.05), 0.5)
    img = np.clip(np.nan_to_num(rgb) / hi, 0, 1) ** 0.85
    chans = []
    b = None
    for c in range(3):
        m, b = to_mercator(np.where(data, img[c], -1).astype(np.float32), grid, Resampling.bilinear, -1.0)
        chans.append(m)
    alpha = (chans[0] >= 0)
    out = np.zeros((*chans[0].shape, 4), np.uint8)
    for c in range(3):
        out[..., c] = (np.clip(chans[c], 0, 1) * 255).astype(np.uint8)
    out[..., 3] = alpha * 255
    Image.fromarray(out, "RGBA").save(path, optimize=True)
    return mercator_corners_lonlat(b)


def render_classes(cls: np.ndarray, valid: np.ndarray, grid: Grid, path: Path) -> list:
    c = np.where(valid, cls, 255).astype(np.uint8)
    cm, b = to_mercator(c, grid, Resampling.nearest, 255)
    pal = np.zeros((256, 4), np.uint8)
    for i, col in enumerate(CLASS_COLORS):
        pal[i] = (*col, 200)
    Image.fromarray(pal[cm], "RGBA").save(path, optimize=True)
    return mercator_corners_lonlat(b)


def write_geotiff(arr: np.ndarray, grid: Grid, path: Path, nodata: float, descriptions: list[str] | None = None,
                  tags: dict[str, Any] | None = None) -> None:
    a = arr if arr.ndim == 3 else arr[None]
    profile = dict(driver="GTiff", height=grid.height, width=grid.width, count=a.shape[0], dtype=a.dtype.name,
                   crs=grid.crs, transform=grid.transform, nodata=nodata, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, predictor=3 if a.dtype.kind == "f" else 2)
    if grid.width < 256 or grid.height < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize"), profile.pop("blockysize")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a)
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        if min(grid.width, grid.height) >= 512:
            dst.build_overviews([2, 4, 8], Resampling.average if a.dtype.kind == "f" else Resampling.nearest)


def detections(prob: np.ndarray, valid: np.ndarray, cls: np.ndarray, grid: Grid, threshold: float,
               min_px: int, max_features: int = 500) -> dict:
    """Connected debris clusters above `threshold` -> GeoJSON FeatureCollection (EPSG:4326)."""
    hit = (prob >= threshold) & valid
    lab, n = ndimage.label(hit, structure=np.ones((3, 3), bool))
    feats: list[dict] = []
    if n:
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(hit, lab, idx)
        maxp = ndimage.maximum(prob, lab, idx)
        meanp = ndimage.mean(prob, lab, idx)
        keep = [(i, s, mx, mn) for i, s, mx, mn in zip(idx, sizes, maxp, meanp) if s >= min_px]
        keep.sort(key=lambda k: (k[2] * np.log1p(k[1])), reverse=True)
        keep = keep[:max_features]
        keep_ids = {int(k[0]) for k in keep}
        info = {int(k[0]): k for k in keep}
        sub = np.where(np.isin(lab, list(keep_ids)), lab, 0).astype(np.int32)
        polys: dict[int, list] = {}
        for geom, val in shapes(sub, mask=sub > 0, transform=grid.transform, connectivity=8):
            polys.setdefault(int(val), []).append(shape(geom))
        px_area = abs(grid.transform.a * grid.transform.e)
        for rank, (i, s, mx, mn) in enumerate(keep, 1):
            parts = polys.get(int(i), [])
            if not parts:
                continue
            g = unary_union(parts)
            g_ll = to_crs(g, 4326, grid.crs)
            c = g_ll.centroid
            sel = lab == i
            dom = np.bincount(cls[sel], minlength=len(MARIDA_CLASSES))
            feats.append({"type": "Feature", "id": rank, "geometry": mapping(g_ll), "properties": {
                "rank": rank, "pixels": int(s), "area_m2": round(float(s * px_area), 1),
                "max_prob": round(float(mx), 4), "mean_prob": round(float(mn), 4),
                "lon": round(c.x, 6), "lat": round(c.y, 6),
                "dominant_class": MARIDA_CLASSES[int(dom.argmax())]}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"threshold": threshold, "min_cluster_pixels": min_px, "clusters_total": int(n)}}


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
